"""Unit tests for minimax_h3_pdd_acc.

Run from the ComfyUI dir (comfy must be importable):
    PYTHONPATH=/workspace/ComfyUI python3 custom_nodes/minimax_h3_pdd_acc/tests/test_pdd_acc.py

Numerical references:
  - reference_minimax_h3_pdd.py is the verbatim helper shipped in
    alibaba-pai/MiniMax-H3-Acc-LoRAs (apache-2.0) — the ground truth for grid,
    plan and head-fusion math.
  - The structural tests read the REAL safetensors headers from models/ and are
    skipped when the files are absent.
"""

import json
import math
import os
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PACK)

import reference_minimax_h3_pdd as ref  # noqa: E402
from pdd_acc_core import (  # noqa: E402
    AUDIO_SHIFT,
    PARTITION_TOLERANCE,
    VIDEO_SHIFT,
    HeadBank,
    block_boundaries,
    check_partition_pairing,
    convert_pdd_lora,
    fine_sigmas,
    fuse_heads,
    identify_trunk,
    make_pdd_final_forward,
    partition_from_name,
    rebase_adaln_to_curve,
    refit_adaln_basis,
    resolve_partition,
    select_block,
    shifted_sigma,
    split_pdd_state_dict,
    warmup_schedule,
)

PDD_FILE = "/workspace/ComfyUI/models/pdd_acc/MiniMax-H3-FL2VA-Acc-8Step.safetensors"
UNET_FILE = "/workspace/ComfyUI/models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors"

PASS = []
SKIP = []


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", {})
    return hdr, meta


def check(name, fn):
    fn()
    PASS.append(name)
    print(f"  ok: {name}")


# ---------------------------------------------------------------------------

def test_grid_matches_reference():
    for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
        for n in (32, 8):
            ours = fine_sigmas(shift, n)
            t_grid = ref.pdd_time_grid(shift, n)  # ascending t = 1 - sigma
            theirs = (1.0 - t_grid).tolist()
            assert len(ours) == len(theirs) == n + 1
            for a, b in zip(ours, theirs):
                assert abs(a - b) < 1e-12, (shift, n, a, b)
            assert ours[0] == 1.0 and ours[-1] == 0.0


def test_step_sizes_match_reference():
    for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
        fine = fine_sigmas(shift, 32)
        d = [fine[k] - fine[k + 1] for k in range(32)]
        steps = ref.pdd_time_grid(shift, 32).diff().tolist()
        for a, b in zip(d, steps):
            assert abs(a - b) < 1e-12


def test_fusion_matches_reference_head():
    torch.manual_seed(0)
    n, out_dim, in_dim = 32, 6, 10
    bank_w = torch.randn(n, out_dim, in_dim, dtype=torch.float64)
    bank_b = torch.randn(n, out_dim, dtype=torch.float64)
    # fuse_heads is pure math over any partition; envelope policy lives in
    # resolve_partition, so off-envelope sizes are exercised here directly
    partitions = [(8,) * 4, (4,) * 8, (2,) * 16, (1,) * 32]
    partitions.append(resolve_partition(n, 6))            # default 8,8,4,4,4,4
    partitions.append((8, 4, 4, 4, 4, 4, 4))              # custom 7-step
    for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
        for sizes in partitions:
            fine = fine_sigmas(shift, n)
            fw, fb = fuse_heads(bank_w, bank_b, fine, sizes)
            steps = ref.pdd_time_grid(shift, n).diff()
            start = 0
            for b, size in enumerate(sizes):
                plan = ref.pdd_sampling_plan(steps, start, size)  # (1, n)
                want_w = torch.einsum("pn,noi->poi", plan, bank_w).flatten(0, 1)
                want_b = torch.einsum("pn,no->po", plan, bank_b).flatten()
                assert torch.allclose(fw[b].double(), want_w, atol=1e-6), (shift, sizes, b)
                assert torch.allclose(fb[b].double(), want_b, atol=1e-6), (shift, sizes, b)
                start += size


def test_fusion_identity_at_full_nfe():
    torch.manual_seed(1)
    bank_w = torch.randn(32, 4, 5)
    bank_b = torch.randn(32, 4)
    fine = fine_sigmas(VIDEO_SHIFT, 32)
    fw, fb = fuse_heads(bank_w, bank_b, fine, (1,) * 32)
    assert torch.allclose(fw, bank_w, atol=1e-6)
    assert torch.allclose(fb, bank_b, atol=1e-6)


def test_boundaries_match_diffusers_set_timesteps():
    # diffusers MiniMaxH3Scheduler.set_timesteps(N): shift*base/(1+(shift-1)*base),
    # base = linspace(1, 0, N). PDD runs it with N = nfe + 1 points.
    for nfe in (4, 8, 16, 32):
        sizes = resolve_partition(32, nfe) if nfe in (4, 8) else (32 // nfe,) * nfe
        bounds = block_boundaries(32, sizes)
        base = torch.linspace(1.0, 0.0, nfe + 1, dtype=torch.float64)
        want = (VIDEO_SHIFT * base / (1 + (VIDEO_SHIFT - 1) * base)).tolist()
        assert len(bounds) == nfe + 1
        for a, b in zip(bounds, want):
            assert abs(a - b) < 1e-12
        # boundaries are every (32/nfe)-th fine knot
        fine = fine_sigmas(VIDEO_SHIFT, 32)
        assert bounds == fine[:: 32 // nfe]


def test_partition_resolution_and_boundaries():
    # default 6-step partition: knots on the fine grid, sizes in the demonstrated {4,8}
    sizes = resolve_partition(32, 6)
    assert sizes == (8, 8, 4, 4, 4, 4)
    fine = fine_sigmas(VIDEO_SHIFT, 32)
    bounds = block_boundaries(32, sizes)
    assert bounds == [fine[k] for k in (0, 8, 16, 20, 24, 28, 32)]
    assert bounds[0] == 1.0 and bounds[-1] == 0.0
    # every 6-step boundary is also an 8-step (block-4) grid knot
    eight = block_boundaries(32, resolve_partition(32, 8))
    assert set(bounds) <= set(eight) | {0.0}
    # explicit text partition overrides nfe
    assert resolve_partition(32, 8, "8,4,4,4,4,4,4") == (8, 4, 4, 4, 4, 4, 4)
    # OFF-ENVELOPE partitions are rejected: heads only trained on block starts at
    # multiples of 4 with sizes 4/8 — nfe 32 renders as heavy noise (2026-08-27 repro)
    for bad in ("8,8,8", "6,6,6,6,6", "0,32", "abc", "8,8,8,9",
                "7,7,7,7,4", "2," * 15 + "2", "16,16", "12,4,4,4,4,4"):
        try:
            resolve_partition(32, 8, bad)
            raise AssertionError(f"partition '{bad}' should be rejected")
        except ValueError:
            pass
    for bad_nfe in (7, 16, 32):
        try:
            resolve_partition(32, bad_nfe)
            raise AssertionError(f"nfe {bad_nfe} should be rejected")
        except ValueError:
            pass


def test_select_block():
    bounds = block_boundaries(32, resolve_partition(32, 8))
    for b in range(8):
        assert select_block(bounds[b], bounds, "error") == b
        assert select_block(bounds[b] + 5e-7, bounds, "error") == b
        assert select_block(bounds[b] - 5e-7, bounds, "error") == b
    assert select_block(0.0, bounds, "error") == 7  # terminal, never actually evaluated
    mid = (bounds[2] + bounds[3]) / 2
    try:
        select_block(mid, bounds, "error")
        raise AssertionError("off-grid sigma must raise in error mode")
    except ValueError:
        pass
    assert select_block(mid, bounds, "clamp") == 2
    assert select_block(1.5, bounds, "clamp") == 0
    assert select_block(1e-4, bounds, "clamp") == 7


def test_qkv_block_diag_numeric():
    torch.manual_seed(2)
    h, o, r = 12, 8, 3
    x = torch.randn(5, h, dtype=torch.float64)
    downs = [torch.randn(r, h, dtype=torch.float64) for _ in range(3)]
    ups = [torch.randn(o, r, dtype=torch.float64) for _ in range(3)]
    # per-branch deltas, concatenated in comfy's [q;k;v] row order
    want = torch.cat([(x @ d.T) @ u.T for d, u in zip(downs, ups)], dim=-1)
    A = torch.cat(downs, dim=0)
    B = torch.zeros(3 * o, 3 * r, dtype=torch.float64)
    for i in range(3):
        B[i * o:(i + 1) * o, i * r:(i + 1) * r] = ups[i]
    got = (x @ A.T) @ B.T
    assert torch.allclose(got, want, atol=1e-10)


def test_swiglu_half_swap_numeric():
    # diffusers: proj -> chunk = (value, gate) -> value * silu(gate)
    # comfy:     fc1  -> chunk = (gate, value) -> silu(gate) * value
    # comfy fc1 rows = [gate; value] = diffusers rows [f:], [:f] swapped.
    torch.manual_seed(3)
    h, f, r = 10, 6, 2
    x = torch.randn(4, h, dtype=torch.float64)
    Wd = torch.randn(2 * f, h, dtype=torch.float64)      # diffusers layout [value; gate]
    A = torch.randn(r, h, dtype=torch.float64)
    Bd = torch.randn(2 * f, r, dtype=torch.float64)      # lora_up in diffusers layout
    Wd_p = Wd + Bd @ A

    def diffusers_ff(w):
        y = x @ w.T
        value, gate = y.chunk(2, dim=-1)
        return value * torch.nn.functional.silu(gate)

    Wc = torch.cat([Wd[f:], Wd[:f]], dim=0)              # comfy base layout [gate; value]
    Bc = torch.cat([Bd[f:], Bd[:f]], dim=0)              # converted lora_B
    Wc_p = Wc + Bc @ A

    def comfy_ff(w):
        y = x @ w.T
        gate, value = y.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * value

    assert torch.allclose(comfy_ff(Wc_p), diffusers_ff(Wd_p), atol=1e-10)


def test_audio_carried_step_exact():
    # Core carried-audio contract: y = c(sv)*x_a with c = s-(s-1)*sv, forward
    # maps returned v_a to (1-s)*x_a + (s/c_i)*v_a and euler steps y over dsv.
    # Claim: the resulting x_a advance equals dsa * v_a EXACTLY per finite step.
    s = VIDEO_SHIFT / AUDIO_SHIFT
    bounds = block_boundaries(32, resolve_partition(32, 8))
    fine_a = fine_sigmas(AUDIO_SHIFT, 32)
    audio_bounds = fine_a[::4]
    for i in range(8):
        sv_i, sv_j = bounds[i], bounds[i + 1]
        sa_i, sa_j = audio_bounds[i], audio_bounds[i + 1]
        x_a, v_a = 0.7314, -1.2345
        c_i = s - (s - 1) * sv_i
        c_j = s - (s - 1) * sv_j
        y_i = c_i * x_a
        out = (1 - s) * x_a + (1 + (s - 1) * sa_i) * v_a   # model.py:549-551 mapping
        y_j = y_i + (sv_j - sv_i) * out                    # euler on the carried variable
        x_a_j = y_j / c_j if c_j != 0 else y_j             # c_j never 0 for s=4
        want = x_a + (sa_j - sa_i) * v_a                   # audio's own euler step
        assert abs(x_a_j - want) < 1e-12, (i, x_a_j, want)
        # and the sigma identity that makes it exact:
        assert abs(s * (sv_j - sv_i) / (c_i * c_j) - (sa_j - sa_i)) < 1e-12


def test_conversion_structure_against_real_files():
    if not os.path.exists(PDD_FILE):
        SKIP.append("conversion_structure (pdd file missing)")
        return
    hdr, meta = read_header(PDD_FILE)
    # zero tensors with the file's real shapes
    sd = {k: torch.zeros(v["shape"], dtype=torch.bfloat16)
          for k, v in hdr.items() if not k.startswith(("proj_out", "audio_proj_out"))}
    out, leftovers = convert_pdd_lora(sd, float(meta.get("lora_alpha", 64.0)))
    assert not leftovers, sorted(leftovers)[:5]
    modules = [k[: -len(".lora_A.weight")] for k in out if k.endswith(".lora_A.weight")]
    # 50 trunk blocks x 5 modules + 2 refiner blocks x 4 modules
    assert len(modules) == 50 * 5 + 2 * 4, len(modules)
    for mk in modules:
        assert mk + ".lora_B.weight" in out and mk + ".alpha" in out
        A = out[mk + ".lora_A.weight"]
        B = out[mk + ".lora_B.weight"]
        assert A.shape[0] == B.shape[1], mk

    if not os.path.exists(UNET_FILE):
        SKIP.append("conversion_structure UNET half (unet file missing)")
        return
    unet_hdr, _ = read_header(UNET_FILE)
    for mk in modules:
        target = mk[len("diffusion_model."):] + ".weight"
        assert target in unet_hdr, f"converted module has no UNET weight: {target}"
        tshape = unet_hdr[target]["shape"]
        A = out[mk + ".lora_A.weight"]
        B = out[mk + ".lora_B.weight"]
        assert list(tshape) == [B.shape[0], A.shape[1]], (mk, tshape, B.shape, A.shape)
    # heads pair with the fp32 final layer
    assert unet_hdr["final_layer.video_out.weight"]["shape"] == [96, 5376]
    assert unet_hdr["final_layer.audio_out.weight"]["shape"] == [32, 5376]
    assert hdr["proj_out.weight"]["shape"] == [int(meta["pdd_num_steps"]), 96, 5376]
    assert hdr["audio_proj_out.weight"]["shape"] == [int(meta["pdd_num_steps"]), 32, 5376]


def test_alpha_scaling_convention():
    # comfy LoRAAdapter: scale = alpha / lora_A.shape[0]; fused qkv must stay 1.0
    sd = {}
    for p in ("q", "k", "v"):
        sd[f"transformer_blocks.0.attn.to_{p}.lora_down"] = torch.zeros(64, 5376)
        sd[f"transformer_blocks.0.attn.to_{p}.lora_up"] = torch.zeros(7168, 64)
    sd["transformer_blocks.0.attn.to_out.0.lora_down"] = torch.zeros(64, 7168)
    sd["transformer_blocks.0.attn.to_out.0.lora_up"] = torch.zeros(5376, 64)
    sd["transformer_blocks.0.ff.net.0.proj.lora_down"] = torch.zeros(64, 5376)
    sd["transformer_blocks.0.ff.net.0.proj.lora_up"] = torch.zeros(28672, 64)
    sd["transformer_blocks.0.ff.net.2.lora_down"] = torch.zeros(64, 14336)
    sd["transformer_blocks.0.ff.net.2.lora_up"] = torch.zeros(5376, 64)
    out, leftovers = convert_pdd_lora(sd, 64.0)
    assert not leftovers
    A = out["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"]
    alpha = out["diffusion_model.blocks.0.attn.qkv_proj.alpha"]
    assert A.shape[0] == 192
    assert abs(float(alpha) / A.shape[0] - 1.0) < 1e-9


def test_split_both_formats_roundtrip():
    torch.manual_seed(4)
    h, o, ffn, r, n = 6, 4, 5, 2, 32
    sd = {}
    for p in ("q", "k", "v"):
        sd[f"transformer_blocks.0.attn.to_{p}.lora_down"] = torch.randn(r, h)
        sd[f"transformer_blocks.0.attn.to_{p}.lora_up"] = torch.randn(o, r)
    sd["transformer_blocks.0.attn.to_out.0.lora_down"] = torch.randn(r, o)
    sd["transformer_blocks.0.attn.to_out.0.lora_up"] = torch.randn(h, r)
    sd["transformer_blocks.0.ff.net.0.proj.lora_down"] = torch.randn(r, h)
    sd["transformer_blocks.0.ff.net.0.proj.lora_up"] = torch.randn(2 * ffn, r)
    sd["transformer_blocks.0.ff.net.2.lora_down"] = torch.randn(r, ffn)
    sd["transformer_blocks.0.ff.net.2.lora_up"] = torch.randn(h, r)
    heads = {"proj_out.weight": torch.randn(n, o, h), "proj_out.bias": torch.randn(n, o),
             "audio_proj_out.weight": torch.randn(n, 3, h), "audio_proj_out.bias": torch.randn(n, 3)}
    meta = {"pdd_num_steps": "32", "pdd_block_size": "4", "lora_alpha": "64.0"}

    lora1, heads1, cfg1 = split_pdd_state_dict({**sd, **heads}, meta, "orig")
    assert cfg1["source_format"] == "original"
    # re-pack as the converted redistribution format and split again
    lora2, heads2, cfg2 = split_pdd_state_dict({**lora1, **heads}, meta, "conv")
    assert cfg2["source_format"] == "converted"
    assert set(lora1) == set(lora2)
    for k in lora1:
        assert torch.equal(lora1[k], lora2[k]), k
    for a, b in zip(heads1, heads2):
        assert torch.equal(a, b)


def test_warmup_schedule():
    knots = fine_sigmas(VIDEO_SHIFT, 32)[::4]
    sigmas, p2 = warmup_schedule(8, 0.800000)
    assert p2 == 8 and len(sigmas) == 11
    assert abs(sigmas[8] - 0.8) < 1e-9            # handoff lands exactly on the knot
    assert abs(sigmas[9] - knots[7]) < 1e-12      # 0.631579...
    assert sigmas[10] == 0.0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
    # warmup segment is uniform in pre-shift t
    ts = [s / (VIDEO_SHIFT - (VIDEO_SHIFT - 1) * s) for s in sigmas[:9]]
    dts = [a - b for a, b in zip(ts, ts[1:])]
    assert max(dts) - min(dts) < 1e-12
    # phase-2 sigmas are all boundaries of the trained 8-step grid -> arming works
    bounds = block_boundaries(32, resolve_partition(32, 8))
    for s in sigmas[8:-1]:
        assert select_block(s, bounds, "error") >= 0
    # other handoffs
    sig2, p2b = warmup_schedule(6, 0.631579)
    assert p2b == 6 and len(sig2) == 8 and abs(sig2[6] - knots[7]) < 1e-4
    sig3, _ = warmup_schedule(4, 0.878049)
    assert [round(s, 6) for s in sig3[4:]] == [0.878049, 0.8, 0.631579, 0.0]
    # 2 base + 6 PDD: uniform-t warmup evals land exactly on the first trained knots
    sig4, p4 = warmup_schedule(2, 0.972973)
    assert p4 == 2 and len(sig4) == 9
    assert [round(s, 6) for s in sig4] == [1.0, 0.988235, 0.972973, 0.952381,
                                           0.923077, 0.878049, 0.8, 0.631579, 0.0]
    try:
        warmup_schedule(8, 0.9)
        raise AssertionError("off-grid handoff must be rejected")
    except ValueError:
        pass


def test_curve_rebase_equivalence():
    # Build a synthetic "dense curve" u(t) that is exactly affine in a rank-k
    # table (how real pruned checkpoints were made), rebase a random adaln
    # lora, and check dense modulation delta == curve delta at every knot.
    torch.manual_seed(5)
    rows, k, dense_in, out_dim, r = 65, 5, 12, 20, 3
    mean = torch.randn(dense_in, dtype=torch.float64)
    G = torch.randn(rows, k, dtype=torch.float64)        # the table
    H = torch.randn(k, dense_in, dtype=torch.float64)
    u = mean + G @ H                                     # u_j = mean + H^T @ table_j

    # solve the affine basis the same way bake_adaln_basis.py does
    X = torch.cat([torch.ones(rows, 1, dtype=torch.float64), G], dim=1)
    sol = torch.linalg.lstsq(X, u).solution
    c, V = sol[0], sol[1:].T                             # [dense_in], [dense_in, k]

    A = torch.randn(r, dense_in)
    B = torch.randn(out_dim, r)
    sd = {"diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight": A,
          "diffusion_model.blocks.0.adaln_proj.linear.lora_B.weight": B,
          "diffusion_model.blocks.0.adaln_proj.linear.alpha": torch.tensor(float(2 * r)),
          "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": torch.zeros(r, 4),
          "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": torch.zeros(4, r),
          "diffusion_model.blocks.0.attn.out_proj.alpha": torch.tensor(1.0)}
    out, n = rebase_adaln_to_curve(sd, c, V)
    assert n == 1
    assert "diffusion_model.blocks.0.attn.out_proj.lora_A.weight" in out  # untouched
    assert not any("adaln" in key and "lora" in key for key in out)
    dW8 = out["diffusion_model.blocks.0.adaln_proj.linear.diff"].double()
    db = out["diffusion_model.blocks.0.adaln_proj.linear.diff_b"].double()
    assert dW8.shape == (out_dim, k) and db.shape == (out_dim,)

    scale = 2.0                                          # alpha 2r / rank r
    dW = scale * (B.double() @ A.double())
    for j in (0, 1, rows // 2, rows - 1):
        dense = dW @ u[j]
        curve = dW8 @ G[j] + db
        assert torch.allclose(dense, curve, atol=1e-5), j
        # dropping the DC term must break it
        assert not torch.allclose(dense, dW8 @ G[j], atol=1e-3)


def test_shipped_adaln_bases():
    from safetensors import safe_open
    basis_dir = os.path.join(PACK, "adaln_basis")
    if not os.path.isdir(basis_dir):
        SKIP.append("shipped_adaln_bases (dir missing)")
        return
    tables = {}
    for trunk in ("fl2va", "ref2va"):
        p = os.path.join(basis_dir, f"basis_{trunk}.safetensors")
        assert os.path.exists(p), p
        with safe_open(p, framework="pt") as f:
            meta = f.metadata()
            c = f.get_tensor("c")
            V = f.get_tensor("V")
            table = f.get_tensor("adaln_t_table")
        assert c.shape == (2688,) and V.shape == (2688, 8), (c.shape, V.shape)
        assert table.shape == (1025, 8)
        assert float(meta["residual"]) < 5e-3, meta["residual"]
        assert meta["trunk"] == trunk
        tables[trunk] = table
    # the two trunks carry genuinely different tables
    assert not torch.allclose(tables["fl2va"], tables["ref2va"], atol=1e-4)


def test_real_file_full_convert():
    if os.environ.get("PDD_ACC_SLOW") != "1":
        SKIP.append("real_file_full_convert (set PDD_ACC_SLOW=1)")
        return
    from safetensors.torch import load_file
    sd = load_file(PDD_FILE)
    meta_hdr, meta = read_header(PDD_FILE)
    for k in ("proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias"):
        sd.pop(k)
    out, leftovers = convert_pdd_lora(sd, float(meta.get("lora_alpha", 64.0)))
    assert not leftovers
    # spot-check the qkv block-diagonal really carries the branch values
    Bq = sd["transformer_blocks.0.attn.to_q.lora_up"]
    B = out["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]
    assert torch.equal(B[:7168, :64], Bq)
    assert torch.count_nonzero(B[:7168, 64:]) == 0
    # spot-check the fc1 half swap
    Bu = sd["transformer_blocks.0.ff.net.0.proj.lora_up"]
    Bc = out["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"]
    assert torch.equal(Bc[:14336], Bu[14336:]) and torch.equal(Bc[14336:], Bu[:14336])


def test_refit_basis_equivalent_table():
    """A table that is an affine reparameterization of the same trunk's curve
    (repacked / requantized pruned build, issue #1) must refit at ~zero
    residual and produce the same rebased diffs as the original basis."""
    torch.manual_seed(7)
    N, k, out_dim, rank = 65, 8, 48, 4
    c = torch.randn(out_dim)
    V = torch.randn(out_dim, k)
    table = torch.randn(N, k)
    # affine reparameterization: table2 carries the same curve in new coordinates
    M = torch.randn(k, k) + 4.0 * torch.eye(k)  # well-conditioned invertible
    m = torch.randn(k)
    table2 = table @ M.T + m
    c2, V2, rel = refit_adaln_basis(c, V, table, table2)
    assert rel < 1e-6, rel  # fp32 storage of c/V/table bounds the f64 fit
    # curve reconstructed from the refit basis matches the original curve
    y = c[None] + table @ V.T
    y2 = c2[None] + table2 @ V2.T
    assert torch.allclose(y, y2, atol=1e-4), (y - y2).abs().max()
    # rebased LoRA diffs agree between the two bases
    A = torch.randn(rank, out_dim)
    B = torch.randn(6, rank)
    sd = {"m.adaln_proj.linear.lora_A.weight": A,
          "m.adaln_proj.linear.lora_B.weight": B,
          "m.adaln_proj.linear.alpha": torch.tensor(float(rank))}
    r1, n1 = rebase_adaln_to_curve(dict(sd), c, V)
    r2, n2 = rebase_adaln_to_curve(dict(sd), c2, V2)
    assert n1 == n2 == 1
    # the diffs live in different table coordinates and the DC term shifts
    # between bases, so compare the full effect on each grid row:
    # diff_b + diff @ t_i must agree (that is what the adaln module computes)
    e1 = (r1["m.adaln_proj.linear.diff_b"].to(torch.float64)[:, None]
          + r1["m.adaln_proj.linear.diff"].to(torch.float64) @ table.to(torch.float64).T)
    e2 = (r2["m.adaln_proj.linear.diff_b"].to(torch.float64)[:, None]
          + r2["m.adaln_proj.linear.diff"].to(torch.float64) @ table2.to(torch.float64).T)
    assert torch.allclose(e1, e2, atol=1e-3), (e1 - e2).abs().max()


def test_refit_basis_rejects_foreign_table():
    torch.manual_seed(8)
    N, k, out_dim = 65, 8, 48
    c = torch.randn(out_dim)
    V = torch.randn(out_dim, k)
    table = torch.randn(N, k)
    _, _, rel = refit_adaln_basis(c, V, table, torch.randn(N, k))
    assert rel > 5e-2, f"foreign table fit unexpectedly well: {rel}"
    try:
        refit_adaln_basis(c, V, table, torch.randn(N + 1, k))
        raise AssertionError("row-count mismatch should raise")
    except ValueError:
        pass


class _PreModRowFinalLayer(torch.nn.Module):
    """Verbatim replica of core FinalLayer.forward BEFORE #15375 (no _mod_row):
    the delegation patch must work against this signature/body unchanged."""

    def __init__(self, hidden, vdim, adim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden, elementwise_affine=False)
        self.video_out = torch.nn.Linear(hidden, vdim)
        self.audio_out = torch.nn.Linear(hidden, adim)
        self.register_buffer("shift", torch.randn(2, hidden))
        self.register_buffer("scale", torch.randn(2, hidden))

    def adaln_proj(self, t_emb):
        return self.shift, self.scale

    def forward(self, x, t_emb, video_seg, audio_seg):
        shift, scale = self.adaln_proj(t_emb)
        va, vb, vrow = video_seg
        aa, ab, arow = audio_seg
        hv = (self.norm(x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]).to(torch.float32)
        ha = (self.norm(x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]).to(torch.float32)
        return self.video_out(hv), self.audio_out(ha)


def _native_stacked_bank(fl, S):
    vW = fl.video_out.weight.detach()[None].repeat(S, 1, 1).clone()
    vB = fl.video_out.bias.detach()[None].repeat(S, 1).clone()
    aW = fl.audio_out.weight.detach()[None].repeat(S, 1, 1).clone()
    aB = fl.audio_out.bias.detach()[None].repeat(S, 1).clone()
    return vW, vB, aW, aB


def _delegation_checks(fl, x, t_emb, vseg, aseg):
    from types import SimpleNamespace
    native = fl(x, t_emb, vseg, aseg)
    sizes = (4,) * 8
    bounds = block_boundaries(32, sizes)
    vW, vB, aW, aB = _native_stacked_bank(fl, 8)
    # make block 3 distinctive
    vW[3] += 1.0
    vB[3] -= 0.5
    aW[3] *= 2.0
    heads = HeadBank(vW, vB, aW, aB)
    holder = SimpleNamespace(sigma_v=None, blk=None)
    patched = make_pdd_final_forward(fl, heads, holder, bounds, "error", 1.0)

    # unarmed (no wrapper state) -> refuse, and native modules stay in place
    try:
        patched(x, t_emb, vseg, aseg)
        raise AssertionError("expected RuntimeError when unarmed")
    except RuntimeError:
        pass
    assert isinstance(fl.video_out, torch.nn.Linear)

    # block 0: bank == native there -> identical to the unpatched forward
    holder.sigma_v = bounds[0]
    v, a = patched(x, t_emb, vseg, aseg)
    assert torch.allclose(v, native[0], atol=1e-6)
    assert torch.allclose(a, native[1], atol=1e-6)

    # block 3: distinctive weights; (W+1, b-0.5) => native + rowsum(hv) - 0.5
    holder.sigma_v = bounds[3]
    v3, a3 = patched(x, t_emb, vseg, aseg)
    assert not torch.allclose(v3, native[0], atol=1e-4)
    diff = v3 - native[0]
    assert torch.allclose(diff, diff[..., :1].expand_as(diff), atol=1e-5), \
        "W+1 head must shift all outputs of a row by the same amount (rowsum(hv) - 0.5)"
    assert torch.allclose(a3, native[1] * 2.0 - fl.audio_out.bias, atol=1e-5)

    # head_strength 0.0 -> pure native even on the distinctive block
    patched0 = make_pdd_final_forward(fl, heads, holder, bounds, "error", 0.0)
    holder.sigma_v = bounds[3]
    v0, a0 = patched0(x, t_emb, vseg, aseg)
    assert torch.allclose(v0, native[0], atol=1e-6)
    assert torch.allclose(a0, native[1], atol=1e-6)

    # projections restored after every call
    assert isinstance(fl.video_out, torch.nn.Linear)
    assert isinstance(fl.audio_out, torch.nn.Linear)
    again = fl(x, t_emb, vseg, aseg)
    assert torch.allclose(again[0], native[0], atol=0.0)


def test_final_forward_delegation_pre_modrow_core():
    torch.manual_seed(9)
    fl = _PreModRowFinalLayer(16, 6, 3)
    x = torch.randn(10, 16)
    _delegation_checks(fl, x, torch.randn(2, 8), (0, 7, 0), (7, 10, 1))


def test_final_forward_delegation_current_core():
    try:
        import comfy.ops
        from comfy.ldm.minimax.model import FinalLayer
    except Exception:
        SKIP.append("final_forward_current_core (comfy not importable)")
        return
    torch.manual_seed(10)
    fl = FinalLayer(16, 8, 6, 3, eps=1e-6, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for p in fl.parameters():
            torch.nn.init.normal_(p, std=0.3)
    x = torch.randn(10, 16)
    _delegation_checks(fl, x, torch.randn(2, 8), (0, 7, 0), (7, 10, 1))


def test_apply_bypass():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "minimax_h3_pdd_acc_testpkg", os.path.join(PACK, "__init__.py"),
            submodule_search_locations=[PACK])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["minimax_h3_pdd_acc_testpkg"] = pkg
        spec.loader.exec_module(pkg)
    except Exception as e:
        SKIP.append(f"apply_bypass (comfy not importable: {type(e).__name__})")
        return
    node = pkg.NODE_CLASS_MAPPINGS["MiniMaxH3PDDAccApply"]()
    model = object()
    sig = torch.tensor([1.0, 0.5, 0.0])
    m, s, info = node.apply(model=model, pdd_file="missing.safetensors", nfe="8",
                            lora_strength=1.0, head_strength=1.0, on_off_grid="error",
                            enabled=False, bypass_sigmas=sig)
    assert m is model and s is sig and "BYPASS" in info
    try:
        node.apply(model=model, pdd_file="missing.safetensors", nfe="8", lora_strength=1.0,
                   head_strength=1.0, on_off_grid="error", enabled=False)
        raise AssertionError("bypass without bypass_sigmas should raise")
    except ValueError:
        pass


class _PR15908FinalLayer(_PreModRowFinalLayer):
    """FinalLayer with the shape core PR #15908 proposes: a widened forward
    signature AND an n-head probe that reads video_out.weight/out_features
    BEFORE calling the projections. Our swapped-in head modules must expose the
    NATIVE attributes so the probe computes n == 1 and takes the stock path."""

    def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
        n = self.video_out.weight.shape[0] // self.video_out.out_features
        assert n == 1, f"probe saw fused-checkpoint head (n={n}) through the swap"
        assert self.video_out.bias is not None and self.video_out.in_features > 0
        return super().forward(x, t_emb, video_seg, audio_seg)


def test_head_swap_survives_pr15908_probe():
    from types import SimpleNamespace
    torch.manual_seed(11)
    fl = _PR15908FinalLayer(16, 6, 3)
    x = torch.randn(10, 16)
    t_emb, vseg, aseg = torch.randn(2, 8), (0, 7, 0), (7, 10, 1)
    extra = (0.5, torch.linspace(1, 0, 9), (12.0, 3.0))   # sigma, sample_sigmas, shifts
    native = fl(x, t_emb, vseg, aseg, *extra)
    sizes = (4,) * 8
    bounds = block_boundaries(32, sizes)
    vW, vB, aW, aB = _native_stacked_bank(fl, 8)
    vW[3] += 1.0
    heads = HeadBank(vW, vB, aW, aB)
    holder = SimpleNamespace(sigma_v=None, blk=None)
    patched = make_pdd_final_forward(fl, heads, holder, bounds, "error", 1.0)
    # extra positional args must reach the widened forward untouched
    holder.sigma_v = bounds[0]
    v, a = patched(x, t_emb, vseg, aseg, *extra)
    assert torch.allclose(v, native[0], atol=1e-6) and torch.allclose(a, native[1], atol=1e-6)
    holder.sigma_v = bounds[3]
    v3, _ = patched(x, t_emb, vseg, aseg, *extra)
    assert not torch.allclose(v3, native[0], atol=1e-4), "armed head must reach the output"


def test_partition_helpers():
    assert partition_from_name("MiniMax-H3-FL2VA-Acc-8Step.safetensors") == "fl2va"
    assert partition_from_name("MiniMax-H3-Ref2VA-Acc-8Step.safetensors") == "ref2va"
    assert partition_from_name("minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors") == "fl2va"
    assert partition_from_name(None, "", "no_trunk_here.safetensors") is None
    assert partition_from_name("fl2va_vs_ref2va_merge.safetensors") is None  # ambiguous
    assert partition_from_name(None, "ref2va", "ignored_fl2va") == "ref2va"  # first hit wins

    torch.manual_seed(12)
    fp_a = torch.randn(8, 16)
    fp_b = fp_a + 0.05 * fp_a.norm() / math.sqrt(fp_a.numel()) * torch.randn(8, 16) * 4
    assert float((fp_a - fp_b).norm() / fp_a.norm()) > 2 * PARTITION_TOLERANCE
    fps = {"fl2va": fp_a.half(), "ref2va": fp_b.half()}
    live = fp_a + 1e-3 * torch.randn(8, 16)   # cast-scale noise
    trunk, dists = identify_trunk(live, fps)
    assert trunk == "fl2va" and dists["fl2va"] < PARTITION_TOLERANCE

    ok = check_partition_pairing(live, fps, "fl2va", "f.safetensors")
    assert ok.startswith("partition check ok")
    try:
        check_partition_pairing(live, fps, "ref2va", "f.safetensors")
        raise AssertionError("confident mismatch must raise")
    except ValueError as e:
        assert "silently wrong" in str(e)
    warned = check_partition_pairing(live, fps, "ref2va", "f.safetensors", mode="warn")
    assert "MISMATCH" in warned
    far = torch.randn(8, 16) * 10
    assert "INCONCLUSIVE" in check_partition_pairing(far, fps, "fl2va", "f")
    assert "INCONCLUSIVE" in check_partition_pairing(live, fps, None, "f")
    assert "skipped" in check_partition_pairing(live, {}, "fl2va", "f")


def test_shipped_partition_fingerprints():
    fp_dir = os.path.join(PACK, "partition_fingerprints")
    paths = {t: os.path.join(fp_dir, f"video_out_{t}.safetensors")
             for t in ("fl2va", "ref2va")}
    if not all(os.path.exists(p) for p in paths.values()):
        SKIP.append("shipped_partition_fingerprints (files absent)")
        return
    from safetensors import safe_open
    fps = {}
    for t, p in paths.items():
        with safe_open(p, framework="pt") as f:
            fps[t] = f.get_tensor("video_out_weight")
            assert f.metadata()["partition"] == t
    gap = float((fps["fl2va"].float() - fps["ref2va"].float()).norm()
                / fps["ref2va"].float().norm())
    assert gap > 0.04, f"trunk gap {gap} suspiciously small"
    # against the real checkpoints when present
    ckpts = {"fl2va": UNET_FILE,
             "ref2va": UNET_FILE.replace("fl2va", "ref2va")}
    for t, ck in ckpts.items():
        if not os.path.exists(ck):
            SKIP.append(f"fingerprint vs real checkpoint ({t} absent)")
            continue
        with safe_open(ck, framework="pt") as f:
            live = f.get_tensor("final_layer.video_out.weight")
        got, dists = identify_trunk(live, fps)
        assert got == t, f"{ck} identified as {got} ({dists})"
        assert dists[t] < 1e-3, f"self-distance {dists[t]} too large"


def test_av_latent_upscale_node():
    try:
        import comfy.nested_tensor
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "minimax_h3_pdd_acc_testpkg2", os.path.join(PACK, "__init__.py"),
            submodule_search_locations=[PACK])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["minimax_h3_pdd_acc_testpkg2"] = pkg
        spec.loader.exec_module(pkg)
    except Exception as e:
        SKIP.append(f"av_latent_upscale (comfy not importable: {type(e).__name__})")
        return
    node = pkg.NODE_CLASS_MAPPINGS["MiniMaxH3AVLatentUpscaleBy"]()
    torch.manual_seed(13)
    video = torch.randn(1, 24, 7, 32, 56)      # 896x512 -> latent 56x32
    audio = torch.randn(1, 32, 2, 88)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio)),
              "noise_mask": torch.ones(1, 1, 7, 32, 56), "extra": "kept"}
    (out,) = node.upscale(latent, "bicubic", 1.5)
    v2, a2 = out["samples"].tensors
    assert tuple(v2.shape) == (1, 24, 7, 48, 84), f"got {tuple(v2.shape)}"   # 1344x768
    assert torch.equal(a2, audio), "audio half must pass through untouched"
    assert "noise_mask" not in out and out["extra"] == "kept"
    # odd result snaps to even (2x2 spatial patches)
    (out2,) = node.upscale({"samples": comfy.nested_tensor.NestedTensor((video, audio))},
                           "bilinear", 1.4)
    assert all(s % 2 == 0 for s in out2["samples"].tensors[0].shape[-2:])
    # input latent unchanged (copy semantics)
    assert torch.equal(latent["samples"].tensors[0], video)
    # a plain (non-nested) latent is refused
    try:
        node.upscale({"samples": torch.randn(1, 4, 32, 56)}, "bicubic", 1.5)
        raise AssertionError("plain latent should be refused")
    except ValueError:
        pass


def test_scheduler_denoise_slice_keeps_resume_sigma():
    """Regression: _sigmas_tensor force-snapped sigmas[0] to 1.0, so a
    denoise-sliced schedule silently became a full-noise start and the resume
    latent was discarded (water-droplet garble in two-pass upscale flows)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "minimax_h3_pdd_acc_testpkg3", os.path.join(PACK, "__init__.py"),
            submodule_search_locations=[PACK])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["minimax_h3_pdd_acc_testpkg3"] = pkg
        spec.loader.exec_module(pkg)
    except Exception as e:
        SKIP.append(f"scheduler_denoise_slice (comfy not importable: {type(e).__name__})")
        return
    sched = pkg.NODE_CLASS_MAPPINGS["MiniMaxH3PDDAccScheduler"]()
    (full,) = sched.get_sigmas("8", 1.0)
    assert float(full[0]) == 1.0 and float(full[-1]) == 0.0 and full.numel() == 9
    (s25,) = sched.get_sigmas("8", 0.25)
    assert s25.numel() == 3 and abs(float(s25[0]) - 0.8) < 1e-5, \
        f"denoise 0.25 must resume at 0.8, got {float(s25[0])}"
    (s50,) = sched.get_sigmas("8", 0.5)
    assert s50.numel() == 5 and abs(float(s50[0]) - 0.923077) < 1e-5
    assert float(s25[-1]) == 0.0 and float(s50[-1]) == 0.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(PASS)} passed" + (f", skipped: {SKIP}" if SKIP else ""))

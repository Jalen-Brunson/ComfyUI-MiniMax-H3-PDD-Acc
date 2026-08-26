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
    VIDEO_SHIFT,
    block_boundaries,
    convert_pdd_lora,
    fine_sigmas,
    fuse_heads,
    select_block,
    shifted_sigma,
    split_pdd_state_dict,
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
    for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
        for nfe in (4, 8, 16, 32):
            L = n // nfe
            fine = fine_sigmas(shift, n)
            fw, fb = fuse_heads(bank_w, bank_b, fine, nfe)
            steps = ref.pdd_time_grid(shift, n).diff()
            for b in range(nfe):
                plan = ref.pdd_sampling_plan(steps, b * L, L)  # (1, n)
                want_w = torch.einsum("pn,noi->poi", plan, bank_w).flatten(0, 1)
                want_b = torch.einsum("pn,no->po", plan, bank_b).flatten()
                assert torch.allclose(fw[b].double(), want_w, atol=1e-6), (shift, nfe, b)
                assert torch.allclose(fb[b].double(), want_b, atol=1e-6), (shift, nfe, b)


def test_fusion_identity_at_full_nfe():
    torch.manual_seed(1)
    bank_w = torch.randn(32, 4, 5)
    bank_b = torch.randn(32, 4)
    fine = fine_sigmas(VIDEO_SHIFT, 32)
    fw, fb = fuse_heads(bank_w, bank_b, fine, 32)
    assert torch.allclose(fw, bank_w, atol=1e-6)
    assert torch.allclose(fb, bank_b, atol=1e-6)


def test_boundaries_match_diffusers_set_timesteps():
    # diffusers MiniMaxH3Scheduler.set_timesteps(N): shift*base/(1+(shift-1)*base),
    # base = linspace(1, 0, N). PDD runs it with N = nfe + 1 points.
    for nfe in (4, 8, 16, 32):
        bounds = block_boundaries(32, nfe)
        base = torch.linspace(1.0, 0.0, nfe + 1, dtype=torch.float64)
        want = (VIDEO_SHIFT * base / (1 + (VIDEO_SHIFT - 1) * base)).tolist()
        assert len(bounds) == nfe + 1
        for a, b in zip(bounds, want):
            assert abs(a - b) < 1e-12
        # boundaries are every (32/nfe)-th fine knot
        fine = fine_sigmas(VIDEO_SHIFT, 32)
        assert bounds == fine[:: 32 // nfe]


def test_select_block():
    bounds = block_boundaries(32, 8)
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
    bounds = block_boundaries(32, 8)
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


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(PASS)} passed" + (f", skipped: {SKIP}" if SKIP else ""))

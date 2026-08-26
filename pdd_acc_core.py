"""Comfy-independent core for the MiniMax-H3 PDD Acc LoRAs (alibaba-pai).

Pure torch: grid math, per-block head fusion, sigma->block selection, and the
diffusers->ComfyUI LoRA key conversion. Used by nodes.py, the standalone
convert_pdd_acc.py CLI, and the tests.

Source files (alibaba-pai/MiniMax-H3-Acc-LoRAs) carry two things:
  1. rank-64 trunk LoRA in diffusers naming (bare lora_down/lora_up suffixes,
     alpha only in file metadata),
  2. a PDD head bank: proj_out [N,96,5376] / audio_proj_out [N,32,5376]
     (+ biases) = one final-layer projection per fine interval of an N-point
     grid (uniform linspace(1,0,N+1) under sigma shift 12 video / 3 audio).

A "converted" file (this project's ComfyUI redistribution format,
format = minimax_h3_pdd_acc_comfyui_v1) carries the SAME head bank keys plus
the LoRA already renamed to ComfyUI keys (diffusion_model.*.lora_A/B.weight
+ .alpha). Both formats load through split_pdd_state_dict().
"""

import hashlib
import re

import torch

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
KNOT_TOLERANCE = 1e-6
CONVERTED_FORMAT = "minimax_h3_pdd_acc_comfyui_v1"

HEAD_KEYS = ("proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias")


# ---------------------------------------------------------------------------
# grid math (float64 throughout; endpoints are analytically exact 1.0 / 0.0)
# ---------------------------------------------------------------------------

def shifted_sigma(shift, t):
    return shift * t / (1.0 + (shift - 1.0) * t)


def fine_sigmas(shift, num_steps):
    """Descending [1.0 ... 0.0], num_steps+1 knots, python floats (float64)."""
    return [shifted_sigma(shift, (num_steps - i) / num_steps) for i in range(num_steps + 1)]


# Non-uniform step counts that don't divide the fine grid get a default
# partition (block sizes in fine steps). 6: merge the early high-sigma blocks
# (they span almost no sigma, so block-start feature staleness is minimal) and
# keep the heavyweight late blocks at the trained size 4; the size-8 blocks
# start at multiples of 8, exactly like the official 4-NFE regrouping.
DEFAULT_PARTITIONS = {6: (8, 8, 4, 4, 4, 4)}


def resolve_partition(num_steps, nfe, partition_text="", trained_block=4):
    """Return the tuple of block sizes (in fine steps) for an nfe / partition spec.

    Sizes are restricted to {trained_block, 2*trained_block} (4 and 8 for the
    released files): PDD heads are conditioned on trunk features from block
    STARTS at multiples of L_min=trained_block with sizes up to L_max=2*L_min —
    the two officially demonstrated groupings. Evaluating the trunk anywhere
    else feeds the heads features they never trained on; empirically (fl2va,
    nfe 32, 2026-08-27 reports + local repro) that renders as heavy noise, so
    off-envelope partitions are rejected rather than allowed to degrade.
    """
    text = (partition_text or "").strip()
    if text:
        try:
            sizes = tuple(int(p) for p in text.replace(" ", "").split(",") if p)
        except ValueError:
            raise ValueError(f"partition '{partition_text}' is not a comma-separated int list")
        if any(s < 1 for s in sizes) or sum(sizes) != num_steps:
            raise ValueError(f"partition {sizes} must be positive block sizes summing to "
                             f"{num_steps} (got sum {sum(sizes)})")
    elif num_steps % nfe == 0:
        sizes = (num_steps // nfe,) * nfe
    elif nfe in DEFAULT_PARTITIONS and sum(DEFAULT_PARTITIONS[nfe]) == num_steps:
        sizes = DEFAULT_PARTITIONS[nfe]
    else:
        raise ValueError(f"nfe {nfe} does not divide pdd_num_steps {num_steps} and has no "
                         f"default partition — supply one via the partition field (block sizes "
                         f"summing to {num_steps}, e.g. '8,8,4,4,4,4')")
    allowed = (trained_block, 2 * trained_block)
    bad = sorted({s for s in sizes if s not in allowed})
    if bad:
        raise ValueError(
            f"partition {sizes}: block sizes {bad} are outside the trained envelope "
            f"{allowed}. PDD heads only ever saw trunk features from block starts on the "
            f"L_min={trained_block} grid with blocks of {allowed[0]} or {allowed[1]} fine "
            f"steps; anything else renders as heavy noise (verified on fl2va). Use nfe "
            f"8/6/4 or a partition made of {allowed[0]}s and {allowed[1]}s.")
    return sizes


def partition_starts(sizes):
    starts = [0]
    for s in sizes[:-1]:
        starts.append(starts[-1] + s)
    return starts


def block_boundaries(num_steps, sizes):
    """Video-sigma boundaries of a partition: descending, len(sizes)+1, ends at 0."""
    fine = fine_sigmas(VIDEO_SHIFT, num_steps)
    knots = partition_starts(sizes) + [num_steps]
    return [fine[k] for k in knots]


def fuse_heads(bank_w, bank_b, fine, sizes):
    """Fuse per-fine-interval velocity heads into one head per partition block.

    Plan weights are the block's fine step sizes normalized to the block span,
    per modality — identical to the reference einsum('pn,noi->poi', plan, bank).
    """
    d = [fine[k] - fine[k + 1] for k in range(bank_w.shape[0])]
    w64 = bank_w.to(torch.float64)
    b64 = bank_b.to(torch.float64)
    fused_w = []
    fused_b = []
    for start, size in zip(partition_starts(sizes), sizes):
        ks = list(range(start, start + size))
        span = sum(d[k] for k in ks)
        fw = sum((d[k] / span) * w64[k] for k in ks)
        fb = sum((d[k] / span) * b64[k] for k in ks)
        fused_w.append(fw.to(torch.float32))
        fused_b.append(fb.to(torch.float32))
    return torch.stack(fused_w).contiguous(), torch.stack(fused_b).contiguous()


def select_block(sigma, bounds, on_off_grid):
    """Map a step-start sigma to its trained block. bounds is descending, len nfe+1."""
    nfe = len(bounds) - 1
    for b in range(nfe):
        if abs(sigma - bounds[b]) <= KNOT_TOLERANCE:
            return b
    if abs(sigma - bounds[-1]) <= KNOT_TOLERANCE:
        return nfe - 1
    if on_off_grid == "error":
        pretty = ", ".join(f"{s:.6f}" for s in bounds)
        raise ValueError(
            f"MiniMaxH3PDDAccApply: model evaluated at sigma {sigma:.6f}, which is not a trained "
            f"PDD block boundary [{pretty}]. Use the `sigmas` output of the Apply node (or the "
            f"MiniMaxH3PDDAccScheduler) with the plain `euler` sampler — multi-stage samplers "
            f"(er_sde, dpmpp, res_*) evaluate off-grid and cannot drive PDD heads. "
            f"Set on_off_grid=clamp to force nearest-block instead (degraded output)."
        )
    if sigma >= bounds[0]:
        return 0
    for b in range(nfe):
        if sigma > bounds[b + 1]:
            return b
    return nfe - 1


# ---------------------------------------------------------------------------
# LoRA key conversion: alibaba/diffusers naming -> ComfyUI H3 naming
# ---------------------------------------------------------------------------

def convert_pdd_lora(sd, alpha):
    """Return (comfy_lora_sd, leftover_keys). Consumes lora_down/lora_up pairs from sd.

    Verified transforms (comfy/ldm/minimax/model.py vs diffusers transformer_minimax_h3.py):
      - to_q/to_k/to_v -> attn.qkv_proj: comfy splits [q;k;v] in row order, so
        lora_A rows concatenate and lora_B goes block-diagonal; alpha*3 keeps the
        per-branch scale alpha/rank exact (comfy computes scale = alpha/A.shape[0]).
      - ff.net.0.proj -> mlp.fc1: diffusers SwiGLU chunks [value;gate], comfy
        _swiglu_eager chunks [gate;value] -> swap the two lora_B row halves.
      - ff.net.2 -> mlp.fc2, to_out.0 -> attn.out_proj: straight copy.
      - adaln_proj.linear: identical layout in both impls (modality-outer x
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)) -> copy.
      - token_refiner.refiner_blocks.N -> token_refiner.blocks.N (no adaln there).
    """
    out = {}
    consumed = set()

    def take(k):
        consumed.add(k)
        return sd[k]

    def emit(dst_module, A, B, alpha_val):
        out[dst_module + ".lora_A.weight"] = A.contiguous()
        out[dst_module + ".lora_B.weight"] = B.contiguous()
        out[dst_module + ".alpha"] = torch.tensor(float(alpha_val))

    def qkv(src_prefix, dst_module):
        parts = [(take(f"{src_prefix}.attn.to_{p}.lora_down"),
                  take(f"{src_prefix}.attn.to_{p}.lora_up")) for p in ("q", "k", "v")]
        r = parts[0][0].shape[0]
        o = parts[0][1].shape[0]
        A = torch.cat([p[0] for p in parts], dim=0)
        B = torch.zeros(o * 3, r * 3, dtype=parts[0][1].dtype)
        for i, (_, up) in enumerate(parts):
            B[i * o:(i + 1) * o, i * r:(i + 1) * r] = up
        emit(dst_module + ".attn.qkv_proj", A, B, alpha * 3.0)

    def plain(src_module, dst_module, half_swap=False):
        A = take(src_module + ".lora_down")
        B = take(src_module + ".lora_up")
        if half_swap:
            f = B.shape[0] // 2
            B = torch.cat([B[f:], B[:f]], dim=0)
        emit(dst_module, A, B, alpha)

    trunk = sorted({int(m.group(1)) for k in sd
                    if (m := re.match(r"transformer_blocks\.(\d+)\.", k))})
    refiner = sorted({int(m.group(1)) for k in sd
                      if (m := re.match(r"token_refiner\.refiner_blocks\.(\d+)\.", k))})

    for i in trunk:
        s = f"transformer_blocks.{i}"
        d = f"diffusion_model.blocks.{i}"
        qkv(s, d)
        plain(f"{s}.attn.to_out.0", f"{d}.attn.out_proj")
        plain(f"{s}.ff.net.0.proj", f"{d}.mlp.fc1", half_swap=True)
        plain(f"{s}.ff.net.2", f"{d}.mlp.fc2")
        if f"{s}.adaln_proj.linear.lora_down" in sd:
            plain(f"{s}.adaln_proj.linear", f"{d}.adaln_proj.linear")
    for i in refiner:
        s = f"token_refiner.refiner_blocks.{i}"
        d = f"diffusion_model.token_refiner.blocks.{i}"
        qkv(s, d)
        plain(f"{s}.attn.to_out.0", f"{d}.attn.out_proj")
        plain(f"{s}.ff.net.0.proj", f"{d}.mlp.fc1", half_swap=True)
        plain(f"{s}.ff.net.2", f"{d}.mlp.fc2")

    leftovers = set(sd.keys()) - consumed
    return out, leftovers


# ---------------------------------------------------------------------------
# pruned (curve-form adaln) support
# ---------------------------------------------------------------------------

def table_sha(table):
    t = table.detach().to(torch.float32).contiguous().cpu()
    return hashlib.sha256(str(tuple(t.shape)).encode() + t.numpy().tobytes()).hexdigest()[:16]


def rebase_adaln_to_curve(lora_sd, c, V):
    """Rewrite dense adaln LoRA modules as curve-form weight+bias diffs.

    A pruned checkpoint's adaln consumes shared-table curve coordinates instead
    of silu(t_emb(t)); with the affine basis silu(t_emb(t)) ~= c + V @ table(t)
    (see bake_adaln_basis.py) the dense delta dW = scale*B@A becomes

        weight diff  dW @ V = scale * B @ (A @ V)     [out, k]
        bias   diff  dW @ c = scale * B @ (A @ c)     [out]   (DC term, mandatory)

    contracted A-first so the [out, 2688] dense delta is never materialized.
    Returns (new_lora_sd, num_rebased_modules).
    """
    out = dict(lora_sd)
    c64 = c.to(torch.float64)
    V64 = V.to(torch.float64)
    n = 0
    for k in list(out):
        if not k.endswith(".adaln_proj.linear.lora_A.weight"):
            continue
        mod = k[: -len(".lora_A.weight")]
        A = out.pop(mod + ".lora_A.weight").to(torch.float64)
        B = out.pop(mod + ".lora_B.weight").to(torch.float64)
        alpha = out.pop(mod + ".alpha", None)
        scale = float(alpha) / A.shape[0] if alpha is not None else 1.0
        out[mod + ".diff"] = (scale * (B @ (A @ V64))).to(torch.float32).contiguous()
        out[mod + ".diff_b"] = (scale * (B @ (A @ c64))).to(torch.float32).contiguous()
        n += 1
    return out, n


# ---------------------------------------------------------------------------
# file splitting (both formats)
# ---------------------------------------------------------------------------

def split_pdd_state_dict(sd, metadata, filename="<file>"):
    """Split a PDD Acc state dict into (lora_sd, heads, config).

    Accepts the original alibaba-pai format (diffusers lora keys, converted
    here) and the pre-converted ComfyUI redistribution format (diffusion_model.*
    lora keys used as-is). heads = (vw, vb, aw, ab). config = dict with
    num_steps, block_size, alpha, source_format.
    """
    metadata = metadata or {}
    config = {
        "num_steps": int(metadata.get("pdd_num_steps", 32)),
        "block_size": int(metadata.get("pdd_block_size", 4)),
        "alpha": float(metadata.get("lora_alpha", 64.0)),
    }

    for k in HEAD_KEYS:
        if k not in sd:
            raise ValueError(f"{filename} has no '{k}' — not a PDD Acc file from "
                             f"alibaba-pai/MiniMax-H3-Acc-LoRAs (or a converted copy)?")
    heads = tuple(sd.pop(k) for k in HEAD_KEYS)
    vw, vb, aw, ab = heads
    if vw.ndim != 3 or vw.shape[0] != config["num_steps"] or aw.shape[0] != config["num_steps"]:
        raise ValueError(f"head bank shape mismatch: proj_out {list(vw.shape)}, "
                         f"audio_proj_out {list(aw.shape)}, pdd_num_steps {config['num_steps']}")

    if any(k.startswith("diffusion_model.") for k in sd):
        config["source_format"] = "converted"
        bad = [k for k in sd if not (k.startswith("diffusion_model.") and
                                     k.endswith((".lora_A.weight", ".lora_B.weight", ".alpha")))]
        if bad:
            raise ValueError(f"{filename}: unexpected keys in converted-format file, e.g. "
                             f"{sorted(bad)[:4]}")
        lora_sd = dict(sd)
    else:
        config["source_format"] = "original"
        lora_sd, leftovers = convert_pdd_lora(sd, config["alpha"])
        if leftovers:
            raise ValueError(f"{filename}: {len(leftovers)} unrecognized keys, e.g. "
                             f"{sorted(leftovers)[:4]} — format drift, refusing to load.")
    return lora_sd, heads, config

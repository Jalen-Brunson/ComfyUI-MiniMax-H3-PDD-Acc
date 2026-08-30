#!/usr/bin/env python3
"""Bake the PDD Acc trunk LoRA into a quantized MiniMax-H3 checkpoint (issue #4).

WHY. ComfyUI merges LoRA patches into weights ONCE at load — but only for
modules that fit in VRAM. Modules that get offloaded receive a per-forward
`LowVramPatch` instead: the LoRA (plus a dequantize) is re-applied on EVERY
step. On cards where the trunk sits at the VRAM edge (issue #4: RTX 5090 32GB,
~2x s/it at 864x480) that fixed per-step cost eats much of the 20->8 NFE win.
Baking the trunk LoRA into the checkpoint offline removes those patches
entirely: the runtime then carries only the head-bank object patch, which is
per-fine-interval and cannot be baked.

WHAT. For every module the trunk LoRA targets (~258): dequantize int8-convrot
-> fp32 -> add `strength * (alpha/rank) * B @ A` (comfy's exact scale
convention, `comfy/weight_adapter/lora.py`) -> requantize with the same
comfy-kitchen kernels ComfyUI itself uses (`comfy-kitchen` is a pinned ComfyUI
dependency, so no extra install). Unquantized targets (token_refiner blocks
are plain bf16; pruned-base adaln is fp16 curve-form) get a plain add. On a
PRUNED base the adaln LoRA is first rebased into curve space against the
shipped basis — the same `rebase_adaln_to_curve` path the Apply node runs.

The output keeps every tensor's dtype, shape and byte length IDENTICAL to the
input, so the file is written STREAMING: the input's tensor layout is reused
verbatim and untouched tensors are raw byte copies. Peak RAM is one module,
not one checkpoint — a 34GB bake needs a few GB, not 34.

Round-trip fidelity, measured before shipping: dequant->requant of an
UNTOUCHED real weight reproduces 100% of int8 codes (scale rel err ~1e-7), so
the bake's only real error is requantizing the LoRA delta — same class as the
runtime merge, which also requantizes (stochastic rounding there, nearest
here).

USE.
    python3 bake_pdd_trunk.py --base <trunk .safetensors> --pdd <acc file> \
        --out <baked .safetensors> [--strength 1.0] [--check] [--sample 8]

    --check   audit only (no write): sample modules, report requant error of
              (dequant(base)+delta) through the int8 round trip.
    After a write, the same audit runs against the OUTPUT file automatically.

Then load the baked checkpoint with a normal UNETLoader and set the Apply
node's `lora_strength` to 0.0 (baked-trunk mode: trunk patching skipped, head
bank / sigmas / guards unchanged). The partition fingerprint survives baking —
the trunk LoRA never touches `final_layer.video_out`.

CAVEATS. Strength is frozen into the file (re-bake to change it). GGUF and
non-convrot quant formats are refused, not guessed. Baking helps ONLY when the
model does not fully fit in VRAM; a full-load card already pays nothing per
step for runtime patches.
"""

import argparse
import datetime
import hashlib
import json
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdd_acc_core import (  # noqa: E402
    rebase_adaln_to_curve,
    refit_adaln_basis,
    split_pdd_state_dict,
    table_sha,
)

torch.set_num_threads(min(8, os.cpu_count() or 8))

COPY_CHUNK = 64 << 20


# ---------------------------------------------------------------------------
# safetensors plumbing (header + raw byte access; no full-file load)
# ---------------------------------------------------------------------------

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
           "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16,
           "I32": torch.int32, "I64": torch.int64, "F64": torch.float64}
# bf16 has no numpy type; serialize through a same-width integer view
_VIEW_FOR_BYTES = {torch.bfloat16: torch.int16}


def read_tensor(f, entry, data_start):
    b0, b1 = entry["data_offsets"]
    f.seek(data_start + b0)
    raw = f.read(b1 - b0)
    dt = _DTYPES[entry["dtype"]]
    t = torch.frombuffer(bytearray(raw), dtype=_VIEW_FOR_BYTES.get(dt, dt))
    if dt in _VIEW_FOR_BYTES:
        t = t.view(dt)
    return t.reshape(entry["shape"]).clone()


def tensor_bytes(t):
    t = t.contiguous()
    if t.dtype in _VIEW_FOR_BYTES:
        t = t.view(_VIEW_FOR_BYTES[t.dtype])
    return t.numpy().tobytes()


# ---------------------------------------------------------------------------
# quant round trip (per-tensor format read from its own comfy_quant marker —
# group size VARIES by module in the published files: qkv 256, adaln 64)
# ---------------------------------------------------------------------------

def _quant_meta(f, header, data_start, base_key):
    mk = base_key + ".comfy_quant"
    if mk not in header:
        return None
    meta = json.loads(bytes(read_tensor(f, header[mk], data_start).tolist()).decode())
    if meta.get("format") != "int8_tensorwise" or not meta.get("convrot"):
        raise ValueError(
            f"{base_key}: quant format {meta} is not int8_tensorwise+convrot — this "
            f"tool bakes only the published Comfy-Org int8_convrot layout (GGUF and "
            f"other formats: dequantize to a plain checkpoint first, or bake there).")
    return int(meta.get("convrot_groupsize", 64))


def _kitchen():
    try:
        from comfy_kitchen.backends.eager.quantization import (
            dequantize_int8_convrot_weight, quantize_int8_convrot_weight)
    except ImportError as e:
        raise ImportError(
            "comfy-kitchen is required (it ships with ComfyUI — run inside the "
            "same python environment ComfyUI uses).") from e
    return dequantize_int8_convrot_weight, quantize_int8_convrot_weight


# ---------------------------------------------------------------------------
# deltas from the PDD file
# ---------------------------------------------------------------------------

def build_deltas(lora_sd, strength):
    """{disk module key: {"weight": callable -> fp32 delta [out,in],
                          "bias": fp32 delta or None}}.

    Handles both patch kinds the runtime path can produce: lora_A/lora_B/alpha
    triples (delta = strength * alpha/rank * B@A — comfy computes scale as
    alpha / lora_A.shape[0], `comfy/weight_adapter/lora.py`) and diff/diff_b
    pairs (the pruned-base curve rebase output; plain strength-scaled adds).
    Disk keys drop the `diffusion_model.` prefix the lora keys carry.

    Weight deltas are LAZY (a thunk expanding B@A on demand): materializing all
    258 fp32 deltas up front is ~140GB — the first version did exactly that and
    would OOM the 64GB boxes this tool exists for. The factors are megabytes;
    one expanded delta at a time is the peak."""
    deltas = {}

    def slot(key):
        assert key.startswith("diffusion_model."), key
        return deltas.setdefault(key[len("diffusion_model."):],
                                 {"weight": None, "bias": None})

    def lora_thunk(mod):
        def expand():
            A = lora_sd[mod + ".lora_A.weight"].to(torch.float32)
            B = lora_sd[mod + ".lora_B.weight"].to(torch.float32)
            alpha = float(lora_sd[mod + ".alpha"])
            return (strength * alpha / A.shape[0]) * (B @ A)
        return expand

    for k in lora_sd:
        if k.endswith(".lora_A.weight"):
            mod = k[:-len(".lora_A.weight")]
            slot(mod)["weight"] = lora_thunk(mod)
        elif k.endswith(".diff"):
            mod = k[:-len(".diff")]
            t = strength * lora_sd[k].to(torch.float32)
            slot(mod)["weight"] = lambda t=t: t
        elif k.endswith(".diff_b"):
            mod = k[:-len(".diff_b")]
            slot(mod)["bias"] = strength * lora_sd[k].to(torch.float32)
        elif not k.endswith((".lora_B.weight", ".alpha")):
            raise ValueError(f"unexpected lora key {k}")
    return deltas


def load_pdd_deltas(pdd_path, base_header, base_path, strength):
    """Split the PDD file, run the pruned-base curve rebase when the base is
    curve-form, and return (deltas, config, notes)."""
    from safetensors import safe_open
    notes = []
    with safe_open(pdd_path, framework="pt") as f:
        meta = dict(f.metadata() or {})
        sd = {k: f.get_tensor(k) for k in f.keys()}
    lora_sd, _heads, config = split_pdd_state_dict(sd, meta, os.path.basename(pdd_path))

    pruned = "adaln_t_table" in base_header
    if pruned:
        header, data_start = base_header, None
        with open(base_path, "rb") as bf:
            n = struct.unpack("<Q", bf.read(8))[0]
            data_start = 8 + n
            table = read_tensor(bf, header["adaln_t_table"], data_start).to(torch.float32)
        c, V, note = _find_basis(table)
        notes.append(note)
        lora_sd, nre = rebase_adaln_to_curve(lora_sd, c, V)
        notes.append(f"pruned base: {nre} adaln modules rebased to curve space")
    return build_deltas(lora_sd, strength), config, notes


def _find_basis(table):
    """Match the base's adaln_t_table to a shipped basis (exact bytes, or an
    f64 refit for a repacked same-shape table) — the offline twin of the Apply
    node's `_curve_rebase_if_pruned`."""
    from safetensors import safe_open
    basis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaln_basis")
    sha = table_sha(table)
    candidates = []
    for fn in sorted(os.listdir(basis_dir)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(basis_dir, fn), framework="pt") as f:
            data = {k: f.get_tensor(k) for k in f.keys()}
        bt = data["adaln_t_table"].to(torch.float32)
        if table_sha(bt) == sha:
            return data["c"], data["V"], f"adaln basis {fn} (exact table match)"
        if bt.shape == table.shape:
            candidates.append((fn, data, bt))
    for fn, data, bt in candidates:
        c2, V2, rel = refit_adaln_basis(data["c"], data["V"], bt, table)
        if rel < 5e-3:
            return c2, V2, f"adaln basis {fn} (refit, residual {rel:.2e})"
    raise ValueError(
        "this base is PRUNED (curve-form adaln) but its adaln_t_table matches no "
        "shipped basis in adaln_basis/ — bake one with bake_adaln_basis.py first.")


# ---------------------------------------------------------------------------
# bake one module (returns replacement bytes for the keys it changes)
# ---------------------------------------------------------------------------

def bake_module(f, header, data_start, mod, delta):
    """{key: bytes} for the module's changed tensors. Byte lengths must equal
    the originals — asserted, since the streaming writer reuses the layout."""
    wkey = mod + ".weight"
    if wkey not in header:
        raise ValueError(f"PDD file targets {mod} but the base has no {wkey} — "
                         f"wrong trunk file, or a checkpoint layout this tool predates.")
    out = {}
    entry = header[wkey]
    w = read_tensor(f, entry, data_start)
    gs = _quant_meta(f, header, data_start, mod)
    dw = delta["weight"]()
    if gs is not None:
        deq, req = _kitchen()
        scale = read_tensor(f, header[mod + ".weight_scale"], data_start)
        w_f = deq(w, scale, gs)
        q, sc = req((w_f + dw).float(), gs)
        assert q.dtype == w.dtype and tuple(q.shape) == tuple(w.shape)
        out[wkey] = tensor_bytes(q)
        out[mod + ".weight_scale"] = tensor_bytes(sc.to(scale.dtype))
    else:
        out[wkey] = tensor_bytes((w.to(torch.float32) + dw).to(w.dtype))
    if delta["bias"] is not None:
        bkey = mod + ".bias"
        b = read_tensor(f, header[bkey], data_start)
        out[bkey] = tensor_bytes((b.to(torch.float32) + delta["bias"]).to(b.dtype))
    for k, raw in out.items():
        b0, b1 = header[k]["data_offsets"]
        assert len(raw) == b1 - b0, f"{k}: baked {len(raw)}B vs slot {b1 - b0}B"
    return out


# ---------------------------------------------------------------------------
# streaming write
# ---------------------------------------------------------------------------

def bake_stream(base_path, pdd_path, out_path, strength):
    header, data_start = read_header(base_path)
    metadata = dict(header.pop("__metadata__", {}) or {})
    deltas, config, notes = load_pdd_deltas(pdd_path, header, base_path, strength)
    for n in notes:
        print(f"  {n}")
    missing = [m for m in deltas if m + ".weight" not in header]
    if missing:
        raise ValueError(f"{len(missing)} PDD modules absent from the base, e.g. "
                         f"{sorted(missing)[:3]} — wrong trunk file?")
    print(f"  {len(deltas)} modules to bake @ strength {strength} "
          f"({config['source_format']} format, alpha {config['alpha']})")

    with open(pdd_path, "rb") as f:
        pdd_sha = hashlib.sha256(f.read()).hexdigest()
    metadata.update({
        "pdd_acc_baked": "true",
        "pdd_acc_bake_source": os.path.basename(pdd_path),
        "pdd_acc_bake_source_sha256": pdd_sha,
        "pdd_acc_bake_strength": repr(float(strength)),
        "pdd_acc_bake_base": f"{os.path.basename(base_path)} "
                             f"({os.path.getsize(base_path)} bytes)",
        "pdd_acc_bake_date": datetime.date.today().isoformat(),
        "pdd_acc_bake_note": "trunk LoRA merged; load with MiniMaxH3PDDAccApply "
                             "lora_strength 0.0 (heads still apply at runtime)",
    })

    # header out: identical tensor table (offsets unchanged — no shape/dtype
    # moves), only __metadata__ differs
    header_out = {"__metadata__": metadata}
    header_out.update(header)
    hjson = json.dumps(header_out, separators=(",", ":")).encode()
    ordered = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])

    replaced = 0
    pending = {}   # mod -> {key: bytes} not yet written (weight+scale share one bake)
    with open(base_path, "rb") as fin, open(out_path, "wb") as fout:
        fout.write(struct.pack("<Q", len(hjson)))
        fout.write(hjson)
        pos = 0
        for key, entry in ordered:
            b0, b1 = entry["data_offsets"]
            assert b0 == pos, f"non-contiguous tensor layout at {key}"
            mod = key.rsplit(".", 1)[0]
            baked = None
            if mod in deltas:
                which = key.rsplit(".", 1)[1]
                if which in ("weight", "weight_scale") or \
                        (which == "bias" and deltas[mod]["bias"] is not None):
                    if mod not in pending:
                        pending[mod] = bake_module(fin, header, data_start, mod, deltas[mod])
                    baked = pending[mod].pop(key)
                    if not pending[mod]:
                        del pending[mod]
            if baked is not None:
                fout.write(baked)
                replaced += 1
                print(f"  baked {key} ({replaced})")
            else:
                fin.seek(data_start + b0)
                left = b1 - b0
                while left:
                    chunk = fin.read(min(COPY_CHUNK, left))
                    fout.write(chunk)
                    left -= len(chunk)
            pos = b1
    print(f"written {out_path} ({os.path.getsize(out_path) / 1e9:.1f} GB), "
          f"{replaced} tensors baked")
    return deltas


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit(base_path, pdd_path, strength, sample, baked_path=None):
    """Relative error of the baked weight vs the exact fp32 merge, per module.

    baked_path=None audits the round trip in memory (--check); otherwise the
    written file is read back and compared against base+delta."""
    from math import ceil
    header, data_start = read_header(base_path)
    header.pop("__metadata__", None)
    deltas, _cfg, _notes = load_pdd_deltas(pdd_path, header, base_path, strength)
    mods = sorted(deltas)
    step = max(1, ceil(len(mods) / sample))
    picked = mods[::step][:sample]
    deq, req = _kitchen()
    worst = (0.0, None)
    if baked_path:
        bheader, bstart = read_header(baked_path)
        bheader.pop("__metadata__", None)
    with open(base_path, "rb") as f:
        fb = open(baked_path, "rb") if baked_path else None
        for mod in picked:
            w = read_tensor(f, header[mod + ".weight"], data_start)
            gs = _quant_meta(f, header, data_start, mod)
            if gs is not None:
                scale = read_tensor(f, header[mod + ".weight_scale"], data_start)
                exact = deq(w, scale, gs) + deltas[mod]["weight"]()
                if fb:
                    q2 = read_tensor(fb, bheader[mod + ".weight"], bstart)
                    s2 = read_tensor(fb, bheader[mod + ".weight_scale"], bstart)
                else:
                    q2, s2 = req(exact.float(), gs)
                got = deq(q2, s2, gs)
            else:
                exact = w.to(torch.float32) + deltas[mod]["weight"]()
                if fb:
                    got = read_tensor(fb, bheader[mod + ".weight"], bstart).to(torch.float32)
                else:
                    got = exact.to(w.dtype).to(torch.float32)
            rel = float((got - exact).norm() / exact.norm())
            print(f"  {mod:<50s} rel {rel:.3e} [{'int8cr g' + str(gs) if gs else str(w.dtype)}]")
            if rel > worst[0]:
                worst = (rel, mod)
        if fb:
            fb.close()
    print(f"audit: {len(picked)}/{len(mods)} modules sampled, worst rel "
          f"{worst[0]:.3e} at {worst[1]}")
    return worst[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="int8_convrot trunk checkpoint")
    ap.add_argument("--pdd", required=True, help="PDD Acc file (original or converted)")
    ap.add_argument("--out", help="output path (required unless --check)")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--check", action="store_true", help="audit only, no write")
    ap.add_argument("--sample", type=int, default=8, help="modules to audit")
    a = ap.parse_args()
    if a.check:
        audit(a.base, a.pdd, a.strength, a.sample)
    else:
        if not a.out:
            ap.error("--out is required (or pass --check)")
        if os.path.abspath(a.out) == os.path.abspath(a.base):
            ap.error("refusing to overwrite the base in place")
        bake_stream(a.base, a.pdd, a.out, a.strength)
        print("post-write audit (baked file vs exact merge):")
        audit(a.base, a.pdd, a.strength, a.sample, baked_path=a.out)

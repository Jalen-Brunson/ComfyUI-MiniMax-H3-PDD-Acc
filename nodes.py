"""ComfyUI nodes for the MiniMax-H3 PDD Acc LoRAs (alibaba-pai/MiniMax-H3-Acc-LoRAs).

See pdd_acc_core.py for the format description and conversion math, README.md
for the required sampling recipe (euler + the node's sigmas output, CFG 1.0,
SigmaShift 12/3).

Audio needs no extra conversion on current core: forward() un-carries the audio
latent before the wrappers, treats the returned audio tensor as the stream's own
velocity, and its carried-variable mapping integrates a mean block velocity
EXACTLY over finite Euler steps (c(sigma) is linear, so s*dsig_v/(c_i*c_j) ==
dsig_a for any step) — the same first-order-exact contract the fused heads were
trained for in diffusers. Sign convention is handled by core's own negation of
the final-layer outputs.
"""

import logging
import math
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.patcher_extension
import comfy.utils
import folder_paths

from .pdd_acc_core import (
    AUDIO_SHIFT,
    VIDEO_SHIFT,
    block_boundaries,
    fine_sigmas,
    fuse_heads,
    rebase_adaln_to_curve,
    resolve_partition,
    select_block,
    split_pdd_state_dict,
    table_sha,
)

WRAPPER_KEY = "minimax_h3_pdd_acc"

_pdd_dir = os.path.join(folder_paths.models_dir, "pdd_acc")
os.makedirs(_pdd_dir, exist_ok=True)
folder_paths.add_model_folder_path("pdd_acc", _pdd_dir, is_default=True)


class HeadBank:
    def __init__(self, vw, vb, aw, ab):
        self.cpu = (vw, vb, aw, ab)
        self._cache = {}

    def for_device(self, device):
        d = torch.device(device)
        if d.type == "cpu":
            return self.cpu
        got = self._cache.get(d)
        if got is None:
            got = tuple(t.to(d) for t in self.cpu)
            self._cache[d] = got
        return got


def make_wrapper(holder):
    def pdd_acc_wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        dm = executor.class_obj
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video",
                                                getattr(dm, "sigma_shift_video", VIDEO_SHIFT)))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio",
                                                getattr(dm, "sigma_shift_audio", AUDIO_SHIFT)))
        if not (math.isclose(shift_v, VIDEO_SHIFT, abs_tol=1e-6)
                and math.isclose(shift_a, AUDIO_SHIFT, abs_tol=1e-6)):
            raise ValueError(
                f"MiniMaxH3PDDAccApply: PDD Acc heads are trained on SigmaShift "
                f"{VIDEO_SHIFT}/{AUDIO_SHIFT}, got {shift_v}/{shift_a}.")
        payload = kwargs.get("minimax_payload") or {}
        scale = float(payload.get("audio_scale", VIDEO_SHIFT / AUDIO_SHIFT))
        if not math.isclose(scale, VIDEO_SHIFT / AUDIO_SHIFT, abs_tol=1e-6):
            raise ValueError(
                f"MiniMaxH3PDDAccApply: payload audio_scale {scale} != "
                f"{VIDEO_SHIFT / AUDIO_SHIFT} — mismatched sampling shifts.")
        holder.sigma_v = float(timestep.flatten()[0]) / 1000.0
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            holder.sigma_v = None
    return pdd_acc_wrapper


def make_pdd_final_forward(final_layer, heads, holder, bounds, on_off_grid, strength):
    from comfy.ldm.minimax.model import _mod_row

    def pdd_final_forward(self, x, t_emb, video_seg, audio_seg):
        # line-for-line FinalLayer.forward through the fp32 casts; only the two
        # projections are replaced by the armed per-block fused heads
        if holder.sigma_v is None:
            raise RuntimeError(
                "MiniMaxH3PDDAccApply: final-layer patch invoked without active diffusion-call "
                "state. Do not stack caching packs (T8 blockcache / EasyCache / Spectrum) on a "
                "PDD-patched model.")
        blk = select_block(holder.sigma_v, bounds, on_off_grid)
        shift, scale = self.adaln_proj(t_emb)

        def mod(seg):
            a, b, row = seg
            return (self.norm(x[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype))
                    + _mod_row(shift, row, shift.dtype)).to(torch.float32)

        hv = mod(video_seg)
        ha = mod(audio_seg)
        vW, vB, aW, aB = heads.for_device(hv.device)
        v = F.linear(hv, vW[blk], vB[blk])
        a = F.linear(ha, aW[blk], aB[blk])
        if strength != 1.0:
            nv = self.video_out(hv)
            na = self.audio_out(ha)
            v = nv + (v - nv) * strength
            a = na + (a - na) * strength
        return v, a

    return pdd_final_forward.__get__(final_layer, final_layer.__class__)


def _sigmas_tensor(bounds):
    t = torch.tensor(bounds, dtype=torch.float32)
    t[0] = 1.0
    t[-1] = 0.0
    return t


def _curve_rebase_if_pruned(model, lora_sd, pdd_file):
    """On a pruned (curve-form adaln) model, rebase the dense adaln LoRA modules
    onto the model's curve basis. Returns (lora_sd, note) — note is "" on dense."""
    dm = model.get_model_object("diffusion_model")
    if not getattr(dm, "use_adaln_curves", False):
        return lora_sd, ""
    if not any(k.endswith(".adaln_proj.linear.lora_A.weight") for k in lora_sd):
        return lora_sd, "pruned model, no adaln lora modules to rebase"

    table = dm.adaln_t_table.detach().to(torch.float32).cpu()
    basis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaln_basis")
    from safetensors.torch import load_file as _load
    match = None
    seen = []
    for name in sorted(os.listdir(basis_dir)) if os.path.isdir(basis_dir) else []:
        if not name.startswith("basis_") or not name.endswith(".safetensors"):
            continue
        data = _load(os.path.join(basis_dir, name))
        seen.append(name)
        bt = data["adaln_t_table"]
        if bt.shape == table.shape and torch.allclose(bt, table, atol=1e-6):
            match = (name, data)
            break
    if match is None:
        raise ValueError(
            f"MiniMaxH3PDDAccApply: this model is PRUNED (curve-form adaln, table "
            f"{list(table.shape)} sha {table_sha(table)}) but its adaln_t_table matches none of "
            f"the shipped bases ({seen}). Bake one from a matching FULL checkpoint:\n"
            f"  python3 bake_adaln_basis.py <full_ckpt> <table source> "
            f"adaln_basis/basis_<name>.safetensors --trunk <name>")
    name, data = match
    trunk = name[len("basis_"):-len(".safetensors")]
    if trunk not in pdd_file.lower():
        logging.warning("MiniMaxH3PDDAccApply: model's adaln table is the %s trunk but the PDD "
                        "file is %s — trunk mismatch? Pair FL2VA with fl2va, Ref2VA with ref2va.",
                        trunk, pdd_file)
    lora_sd, n = rebase_adaln_to_curve(lora_sd, data["c"], data["V"])
    return lora_sd, f"pruned model: {n} adaln modules rebased onto the {trunk} curve basis"


class MiniMaxH3PDDAccApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "pdd_file": (folder_paths.get_filename_list("pdd_acc"),
                         {"tooltip": "PDD Acc file from models/pdd_acc — the original "
                                     "alibaba-pai release or a converted ComfyUI copy, both "
                                     "load. Pair FL2VA with an fl2va UNET, Ref2VA with ref2va."}),
            "nfe": (["8", "4", "6", "16", "32"],
                    {"default": "8",
                     "tooltip": "Model evaluations (sampler steps). 8 = trained block size 4. "
                                "4 regroups two blocks per step (faster, paper-sanctioned); "
                                "6 uses the non-uniform default partition 8,8,4,4,4,4; "
                                "16/32 use shorter blocks (features slightly off-distribution)."}),
            "lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01,
                                        "tooltip": "Trunk LoRA strength. Trained at 1.0."}),
            "head_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                        "tooltip": "PDD head blend: native + s*(pdd - native). "
                                                   "1.0 = trained path (native head not evaluated)."}),
            "on_off_grid": (["error", "clamp"],
                            {"default": "error",
                             "tooltip": "What to do when the model is evaluated at a sigma that is "
                                        "not a trained block boundary (wrong sampler/scheduler)."}),
        }, "optional": {
            "partition": ("STRING", {"default": "",
                                     "tooltip": "Custom block sizes in fine steps, comma-separated, "
                                                "summing to 32 (e.g. '8,8,4,4,4,4' = 6 steps, or "
                                                "'8,4,4,4,4,4,4' = 7). Overrides nfe. Sizes 4 and 8 "
                                                "on multiple-of-4 starts stay inside the "
                                                "officially demonstrated envelope."}),
        }}

    RETURN_TYPES = ("MODEL", "SIGMAS", "STRING")
    RETURN_NAMES = ("model", "sigmas", "info")
    FUNCTION = "apply"
    CATEGORY = "model/patch/minimax"
    DESCRIPTION = ("Applies a MiniMax-H3 PDD Acc LoRA (alibaba-pai): trunk LoRA (converted to "
                   "ComfyUI naming if needed) + per-step fused final-layer head bank. Wire the "
                   "sigmas output to SamplerCustomAdvanced and sample with euler, CFG 1.0, "
                   "SigmaShift 12/3. Remove other distill LoRAs (turbo) and cache nodes.")

    def apply(self, model, pdd_file, nfe, lora_strength, head_strength, on_off_grid, partition=""):
        nfe = int(nfe)
        path = folder_paths.get_full_path_or_raise("pdd_acc", pdd_file)
        sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        lora_sd, (vw, vb, aw, ab), config = split_pdd_state_dict(sd, metadata, pdd_file)
        num_steps = config["num_steps"]
        sizes = resolve_partition(num_steps, nfe, partition)

        # ---- trunk LoRA (normal quant-aware patch path) ----
        lora_sd, curve_note = _curve_rebase_if_pruned(model, lora_sd, pdd_file)
        expected_keys = sum(1 for k in lora_sd
                            if k.endswith((".lora_A.weight", ".diff", ".diff_b")))
        expected_modules = sum(1 for k in lora_sd if k.endswith((".lora_A.weight", ".diff")))
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        loaded = comfy.lora.load_lora(lora_sd, key_map)

        m = model.clone()
        applied = m.add_patches(loaded, lora_strength)
        if len(applied) != expected_keys:
            raise ValueError(
                f"MiniMaxH3PDDAccApply: only {len(applied)}/{expected_keys} LoRA patch keys "
                f"matched the loaded model — is this a MiniMax-H3 UNET of the matching trunk?")

        # ---- head bank ----
        final_layer = m.get_model_object("diffusion_model.final_layer")
        for name, bank in (("video_out", vw), ("audio_out", aw)):
            native = getattr(final_layer, name).weight
            if tuple(native.shape) != tuple(bank.shape[1:]):
                raise ValueError(f"head bank {name} {list(bank.shape[1:])} does not match model "
                                 f"final_layer.{name} {list(native.shape)}")

        fine_v = fine_sigmas(VIDEO_SHIFT, num_steps)
        fine_a = fine_sigmas(AUDIO_SHIFT, num_steps)
        vW, vB = fuse_heads(vw, vb, fine_v, sizes)
        aW, aB = fuse_heads(aw, ab, fine_a, sizes)
        heads = HeadBank(vW, vB, aW, aB)
        bounds = block_boundaries(num_steps, sizes)

        holder = SimpleNamespace(sigma_v=None)
        m.add_object_patch(
            "diffusion_model.final_layer.forward",
            make_pdd_final_forward(final_layer, heads, holder, bounds, on_off_grid, head_strength))
        m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                               make_wrapper(holder))

        sigmas = _sigmas_tensor(bounds)
        info = (
            f"PDD Acc: {pdd_file} ({config['source_format']} format)\n"
            f"grid {num_steps} fine steps (trained block {config['block_size']}) -> "
            f"{len(sizes)} steps, blocks {','.join(map(str, sizes))}\n"
            f"lora: {expected_modules} modules @ strength {lora_strength} "
            f"(alpha {config['alpha']})"
            + (f"\n{curve_note}" if curve_note else "") + "\n"
            f"heads: video {list(vW.shape)}, audio {list(aW.shape)} fp32, "
            f"strength {head_strength}\n"
            f"sigmas: {', '.join(f'{s:.6f}' for s in bounds)}\n"
            f"recipe: euler + these sigmas, CFG 1.0, SigmaShift 12/3. No turbo LoRA, no "
            f"T8/EasyCache/Spectrum, no PDD on a hybrid-merged trunk (untested)."
        )
        logging.info("MiniMaxH3PDDAccApply: %s (%s) steps=%d blocks=%s, %d lora modules, "
                     "heads fused", pdd_file, config["source_format"], len(sizes),
                     ",".join(map(str, sizes)), expected_modules)
        return (m, sigmas, info)


class MiniMaxH3PDDAccScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "nfe": (["8", "4", "6", "16", "32"], {"default": "8"}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Keep only the last round(steps*denoise) blocks "
                                             "(v2v-style partial denoise on the trained grid)."}),
        }, "optional": {
            "partition": ("STRING", {"default": "",
                                     "tooltip": "Custom block sizes summing to 32, overrides nfe — "
                                                "must match the Apply node's partition."}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"
    DESCRIPTION = ("Trained PDD block-boundary sigmas (shift 12 grid). The Apply node's sigmas "
                   "output is the same thing at denoise 1.0 — use this one for partial-denoise "
                   "or split-sigma workflows. nfe must match the Apply node.")

    def get_sigmas(self, nfe, denoise, partition=""):
        sizes = resolve_partition(32, int(nfe), partition)
        bounds = block_boundaries(32, sizes)
        if denoise < 1.0:
            keep = max(1, int(round(len(sizes) * denoise)))
            bounds = bounds[-(keep + 1):]
        return (_sigmas_tensor(bounds),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PDDAccApply": MiniMaxH3PDDAccApply,
    "MiniMaxH3PDDAccScheduler": MiniMaxH3PDDAccScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PDDAccApply": "MiniMax H3 PDD Acc LoRA (Apply)",
    "MiniMaxH3PDDAccScheduler": "MiniMax H3 PDD Acc Scheduler",
}

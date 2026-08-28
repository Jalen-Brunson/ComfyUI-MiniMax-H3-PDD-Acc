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

import inspect
import logging
import math
import os
from types import SimpleNamespace

import torch

import comfy.lora
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
import folder_paths

from .pdd_acc_core import (
    AUDIO_SHIFT,
    VIDEO_SHIFT,
    HeadBank,
    block_boundaries,
    check_partition_pairing,
    fine_sigmas,
    fuse_heads,
    make_pdd_final_forward,
    partition_from_name,
    rebase_adaln_to_curve,
    refit_adaln_basis,
    resolve_partition,
    split_pdd_state_dict,
    table_sha,
    warmup_schedule,
)

WRAPPER_KEY = "minimax_h3_pdd_acc"

_partition_fingerprints = None


def _load_partition_fingerprints():
    """{trunk: fp16 video_out weight} from partition_fingerprints/, cached."""
    global _partition_fingerprints
    if _partition_fingerprints is None:
        _partition_fingerprints = {}
        fp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "partition_fingerprints")
        if os.path.isdir(fp_dir):
            from safetensors.torch import load_file
            for name in sorted(os.listdir(fp_dir)):
                if name.startswith("video_out_") and name.endswith(".safetensors"):
                    trunk = name[len("video_out_"):-len(".safetensors")]
                    _partition_fingerprints[trunk] = load_file(
                        os.path.join(fp_dir, name))["video_out_weight"]
    return _partition_fingerprints

_pdd_dir = os.path.join(folder_paths.models_dir, "pdd_acc")
os.makedirs(_pdd_dir, exist_ok=True)
folder_paths.add_model_folder_path("pdd_acc", _pdd_dir, is_default=True)


def _check_core_supported():
    """Fail closed on ComfyUI cores that predate the MiniMax-H3 carried-audio
    rework (#15243): heads emit mean block velocities and rely on core's
    carried-variable audio mapping — the older stock-world mechanics would
    silently mis-integrate audio."""
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
        src = inspect.getsource(MiniMaxH3Model.forward)
    except Exception:
        return  # can't probe (source unavailable) — don't block on that alone
    if "audio_scale" not in src:
        raise RuntimeError(
            "MiniMaxH3PDDAccApply: this ComfyUI predates the MiniMax-H3 audio-mechanics rework "
            "(comfyanonymous/ComfyUI#15243), so the PDD heads would mis-integrate audio. "
            "Update ComfyUI to v0.33.0 or newer.")


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


def _sigmas_tensor(bounds):
    # Snap the float32 endpoints to their analytic values — but ONLY when they
    # are already those endpoints. Unconditional t[0] = 1.0 silently rewrote a
    # denoise-sliced schedule ([0.8, 0.632, 0] -> [1.0, 0.632, 0]): the resume
    # latent was discarded at full noise and block 0's head was integrated
    # across a 37% sigma span — structured "water droplet" garbage (found via
    # the latent-upscale two-pass flow, 2026-08-28).
    t = torch.tensor(bounds, dtype=torch.float32)
    if abs(float(t[0]) - 1.0) < 1e-4:
        t[0] = 1.0
    if abs(float(t[-1])) < 1e-4:
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
    candidates = []
    for name in sorted(os.listdir(basis_dir)) if os.path.isdir(basis_dir) else []:
        if not name.startswith("basis_") or not name.endswith(".safetensors"):
            continue
        data = _load(os.path.join(basis_dir, name))
        seen.append(name)
        bt = data["adaln_t_table"]
        if bt.shape == table.shape and torch.allclose(bt, table, atol=1e-6):
            match = (name, data["c"], data["V"], "")
            break
        candidates.append((name, data))
    if match is None:
        # Not a byte-identical table. The same trunk's curve can be carried by a
        # different (affinely equivalent, or merely re-rounded) table in a
        # repacked / requantized pruned build — every table samples the same
        # fixed timestep grid, so refit each shipped basis onto this table and
        # accept the best fit if its residual is same-trunk small (known-good
        # equivalence fits are ~1e-5; a different trunk lands around 1e-1).
        fits = []
        for name, data in candidates:
            if data["adaln_t_table"].shape[0] != table.shape[0]:
                continue
            c2, V2, rel = refit_adaln_basis(data["c"], data["V"], data["adaln_t_table"], table)
            fits.append((rel, name, c2, V2))
        fits.sort(key=lambda f: f[0])
        if fits and fits[0][0] < 5e-3:
            rel, name, c2, V2 = fits[0]
            match = (name, c2, V2, f", auto-refit onto its table (residual {rel:.2e})")
            logging.info("MiniMaxH3PDDAccApply: model's adaln_t_table (sha %s) is not "
                         "byte-identical to %s but is curve-equivalent — basis auto-refit, "
                         "residual %.2e", table_sha(table), name, rel)
        else:
            detail = "; ".join(f"{n}: refit residual {r:.2e}" for r, n, _, _ in fits) \
                     or "no shape-compatible shipped basis"
            raise ValueError(
                f"MiniMaxH3PDDAccApply: this model is PRUNED (curve-form adaln, table "
                f"{list(table.shape)} sha {table_sha(table)}) but its adaln_t_table matches none "
                f"of the shipped bases ({seen}; {detail}) — it looks like a different finetune, "
                f"not a repack of a known trunk. Bake a basis from a matching FULL checkpoint:\n"
                f"  python3 bake_adaln_basis.py <full_ckpt> <table source> "
                f"adaln_basis/basis_<name>.safetensors --trunk <name>\n"
                f"or open an issue naming the exact checkpoint file/source so a basis can be "
                f"shipped.")
    name, c, V, extra = match
    trunk = name[len("basis_"):-len(".safetensors")]
    if trunk not in pdd_file.lower():
        logging.warning(
            "MiniMaxH3PDDAccApply: model's adaln table is the %s trunk but the PDD file is %s. "
            "On a PLAIN trunk this is a real mispairing (FL2VA pairs with fl2va, Ref2VA with "
            "ref2va). On a %s-BASED HYBRID (e.g. fl2va+ref2va block merges) the table always "
            "comes from the base trunk, so this pairing may be intentional — PDD still applies "
            "(see the info output), but hybrids are off-label/untested for PDD.",
            trunk, pdd_file, trunk)
    lora_sd, n = rebase_adaln_to_curve(lora_sd, c, V)
    return lora_sd, f"pruned model: {n} adaln modules rebased onto the {trunk} curve basis{extra}"


class MiniMaxH3PDDAccApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "pdd_file": (folder_paths.get_filename_list("pdd_acc"),
                         {"tooltip": "PDD Acc file from models/pdd_acc — the original "
                                     "alibaba-pai release or a converted ComfyUI copy, both "
                                     "load. Pair FL2VA with an fl2va UNET, Ref2VA with ref2va."}),
            "nfe": (["8", "4", "6"],
                    {"default": "8",
                     "tooltip": "Model evaluations (sampler steps). 8 = trained block size 4. "
                                "4 regroups two blocks per step (faster, official); 6 uses the "
                                "non-uniform default partition 8,8,4,4,4,4. Higher step counts "
                                "are OFF the training envelope and render as noise."}),
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
                                                "'8,4,4,4,4,4,4' = 7). Overrides nfe. Only sizes 4 "
                                                "and 8 are accepted — the trained envelope; other "
                                                "sizes render as noise and are rejected."}),
            "enabled": ("BOOLEAN", {"default": True,
                                    "tooltip": "False = full bypass: the input model and "
                                               "bypass_sigmas pass through untouched (nothing is "
                                               "loaded or patched). Wire bypass_sigmas with the "
                                               "schedule for the un-distilled model."}),
            "bypass_sigmas": ("SIGMAS", {"tooltip": "Returned as the sigmas output when "
                                                    "enabled=False (e.g. a BasicScheduler for the "
                                                    "base model). Ignored when enabled."}),
            "partition_check": (["error", "warn"],
                                {"default": "error",
                                 "tooltip": "fl2va and ref2va share identical key sets, so an "
                                            "FL2VA file on a ref2va UNET (or vice versa) applies "
                                            "cleanly and renders silently wrong. The loaded "
                                            "model's trunk is identified from its "
                                            "final_layer.video_out fingerprint; a confident "
                                            "mismatch errors. 'warn' continues anyway "
                                            "(deliberate cross-trunk experiments)."}),
        }}

    RETURN_TYPES = ("MODEL", "SIGMAS", "STRING")
    RETURN_NAMES = ("model", "sigmas", "info")
    FUNCTION = "apply"
    CATEGORY = "model/patch/minimax"
    DESCRIPTION = ("Applies a MiniMax-H3 PDD Acc LoRA (alibaba-pai): trunk LoRA (converted to "
                   "ComfyUI naming if needed) + per-step fused final-layer head bank. Wire the "
                   "sigmas output to SamplerCustomAdvanced and sample with euler, CFG 1.0, "
                   "SigmaShift 12/3. Remove other distill LoRAs (turbo) and cache nodes.")

    def apply(self, model, pdd_file, nfe, lora_strength, head_strength, on_off_grid, partition="",
              enabled=True, bypass_sigmas=None, partition_check="error"):
        if not enabled:
            if bypass_sigmas is None:
                raise ValueError(
                    "MiniMaxH3PDDAccApply: enabled=False needs bypass_sigmas wired (the schedule "
                    "for the un-distilled model) — the PDD block-boundary sigmas would be wrong "
                    "for an unpatched model.")
            return (model, bypass_sigmas,
                    "PDD Acc BYPASSED (enabled=False): model and bypass_sigmas passed through "
                    "unchanged. Remember the un-distilled recipe (CFG, sampler, steps) differs "
                    "from the PDD one.")
        _check_core_supported()
        nfe = int(nfe)
        path = folder_paths.get_full_path_or_raise("pdd_acc", pdd_file)
        sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        lora_sd, (vw, vb, aw, ab), config = split_pdd_state_dict(sd, metadata, pdd_file)
        num_steps = config["num_steps"]
        sizes = resolve_partition(num_steps, nfe, partition)

        # ---- partition fingerprint (before any patching: fail on a mispair
        # early). fl2va/ref2va key sets are identical, so a crossed pairing
        # would otherwise apply cleanly and render silently wrong. ----
        meta = metadata or {}
        file_partition = partition_from_name(
            meta.get("pdd_partition"), meta.get("source_file"), pdd_file)
        fp_note = check_partition_pairing(
            model.get_model_object("diffusion_model.final_layer").video_out.weight,
            _load_partition_fingerprints(), file_partition, pdd_file,
            mode=partition_check)
        logging.info("MiniMaxH3PDDAccApply: %s", fp_note)

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

        holder = SimpleNamespace(sigma_v=None, blk=None)
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
            f"(alpha {config['alpha']})\n"
            f"{fp_note}"
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
        if curve_note:
            logging.info("MiniMaxH3PDDAccApply: %s", curve_note)
        return (m, sigmas, info)


class MiniMaxH3PDDAccScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "nfe": (["8", "4", "6"], {"default": "8"}),
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


class MiniMaxH3PDDAccWarmupScheduler:
    """Two-phase schedule: undistilled-base warmup for identity/structure, PDD tail.

    Wire sigmas into the sampler, phase2_start_step into MMH3LoopingSampler's
    phase2_start_step (or SplitSigmas for a two-pass SamplerCustomAdvanced
    graph). Phase 1 runs a guider on the BASE model (no PDD, no distill loras);
    phase 2 runs the PDD-patched model's guider from the handoff boundary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "warmup_steps": ("INT", {"default": 8, "min": 1, "max": 32,
                                     "tooltip": "Base-model steps from sigma 1.0 to the handoff "
                                                "(uniform in t; any sigmas are legal for the "
                                                "undistilled base)."}),
            "handoff_sigma": (["0.800000", "0.631579", "0.878049", "0.923077",
                               "0.952381", "0.972973", "0.988235"],
                              {"default": "0.800000",
                               "tooltip": "PDD boundary where phase 2 takes over = how many PDD "
                                          "evals finish the run: 0.631579->1, 0.8->2, "
                                          "0.878049->3, 0.923077->4, 0.952381->5, 0.972973->6, "
                                          "0.988235->7. Keep the Apply node at nfe 8 (the tail "
                                          "is on the block-4 grid)."}),
        }}

    RETURN_TYPES = ("SIGMAS", "INT", "STRING")
    RETURN_NAMES = ("sigmas", "phase2_start_step", "info")
    FUNCTION = "get_schedule"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_schedule(self, warmup_steps, handoff_sigma):
        sigmas, p2 = warmup_schedule(warmup_steps, float(handoff_sigma))
        info = (f"phase 1: {p2} base steps {sigmas[0]:.3f}->{sigmas[p2]:.6f} | "
                f"phase 2: {len(sigmas) - 1 - p2} PDD blocks "
                f"{', '.join(f'{s:.6f}' for s in sigmas[p2:])}\n"
                f"phase-1 guider = BASE model (no PDD/turbo), phase-2 guider = PDD model; "
                f"euler both phases, CFG 1.0, SigmaShift 12/3.")
        return (_sigmas_tensor(sigmas), p2, info)


class MiniMaxH3AVLatentUpscaleBy:
    """Spatial upscale for the MiniMax-H3 nested AV latent (video half only).

    Core LatentUpscale(By) cannot handle the nested video+audio pair. The video
    latent is resized per frame; the audio latent passes through untouched. New
    latent dims snap to even (the model packs 2x2 spatial patches), i.e. pixel
    dims stay on the 32px grid.
    """

    upscale_methods = ["bislerp", "bicubic", "bilinear", "nearest-exact", "area"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "upscale_method": (cls.upscale_methods,),
            "scale_by": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 4.0, "step": 0.05,
                                   "tooltip": "1.5 on a 896x512 pass-1 render lands exactly on "
                                              "the model-native 1344x768."}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "model/latent/minimax"
    DESCRIPTION = ("Upscales the video half of a MiniMax-H3 AV latent (audio untouched) for "
                   "two-pass latent-upscale workflows: full PDD pass at low res -> this node -> "
                   "partial-denoise PDD pass (MiniMaxH3PDDAccScheduler with denoise < 1.0).")

    def upscale(self, samples, upscale_method, scale_by):
        nt = samples["samples"]
        if not getattr(nt, "is_nested", False) or len(nt.tensors) != 2 \
                or nt.tensors[0].ndim != 5:
            raise ValueError(
                "MiniMaxH3AVLatentUpscaleBy: expected a MiniMax-H3 AV latent (nested "
                "video+audio pair, e.g. from a MiniMax H3 conditioning node or sampler).")
        video, audio = nt.tensors
        b, c, t, h, w = video.shape
        nh = max(2, round(h * scale_by / 2) * 2)
        nw = max(2, round(w * scale_by / 2) * 2)
        vid = video.movedim(2, 1).reshape(b * t, c, h, w)
        vid = comfy.utils.common_upscale(vid, nw, nh, upscale_method, "disabled")
        vid = vid.reshape(b, t, c, nh, nw).movedim(1, 2)
        s = samples.copy()
        s.pop("noise_mask", None)   # sized for the old resolution
        s["samples"] = comfy.nested_tensor.NestedTensor((vid, audio))
        return (s,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PDDAccApply": MiniMaxH3PDDAccApply,
    "MiniMaxH3PDDAccScheduler": MiniMaxH3PDDAccScheduler,
    "MiniMaxH3PDDAccWarmupScheduler": MiniMaxH3PDDAccWarmupScheduler,
    "MiniMaxH3AVLatentUpscaleBy": MiniMaxH3AVLatentUpscaleBy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PDDAccApply": "MiniMax H3 PDD Acc LoRA (Apply)",
    "MiniMaxH3PDDAccScheduler": "MiniMax H3 PDD Acc Scheduler",
    "MiniMaxH3PDDAccWarmupScheduler": "MiniMax H3 PDD Acc Warmup Scheduler (2-phase)",
    "MiniMaxH3AVLatentUpscaleBy": "MiniMax H3 AV Latent Upscale By",
}

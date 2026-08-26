# Verbatim reference implementation from alibaba-pai/MiniMax-H3-Acc-LoRAs
# (https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs, Apache-2.0).
# Kept unmodified as the numerical test oracle for this pack.
"""PDD LoRA adapter for MiniMax-H3 on Diffusers 0.40.

Parallel Decoding Distillation (PDD, arXiv 2607.26004) is not a PEFT LoRA: the
backbone gets low-rank updates, and the two final heads (`proj_out`,
`audio_proj_out`) are repeated once per interval of a length-`N` grid. Each
generation step fuses a block of those heads into one Euler velocity, so
`NFE = N / L` transformer evaluations cover the whole trajectory.

This module is inference-only. It mutates a loaded
`MiniMaxH3Transformer3DModel` in place and arms the fused heads on every
forward, so the stock Diffusers modular pipeline does not need a step callback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PDD_CONFIG = {
    "pdd_num_steps": 32,
    "pdd_block_size": 4,
    "lora_rank": 64,
    "lora_alpha": 64.0,
    "lora_targets": "to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear",
}


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    """Ascending grid `0 = t_0 < ... < t_N = 1` of one MiniMax-H3 schedule."""
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(shift, sigma)


def pdd_sampling_plan(step_sizes: torch.Tensor, start: int, block_size: int) -> torch.Tensor:
    """Mean velocity of one block, which an Euler step over the block boundaries consumes."""
    plan = torch.zeros(1, step_sizes.shape[0], dtype=step_sizes.dtype, device=step_sizes.device)
    span = step_sizes[start : start + block_size].sum()
    plan[0, start : start + block_size] = step_sizes[start : start + block_size] / span
    return plan


class MiniMaxH3ParallelHead(nn.Module):
    """`N` per-interval output heads in place of one final linear layer."""

    def __init__(self, source: nn.Linear, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = nn.Parameter(source.weight.detach()[None].repeat(num_steps, 1, 1).clone())
        self.bias = (
            None if source.bias is None else nn.Parameter(source.bias.detach()[None].repeat(num_steps, 1).clone())
        )
        self.plan = torch.zeros(1, num_steps)
        self.plan[0, 0] = 1.0

    def set_plan(self, plan: torch.Tensor) -> None:
        if plan.ndim != 2 or plan.shape[1] != self.num_steps:
            raise ValueError(f"A PDD plan must be `(num_directions, {self.num_steps})`, got {list(plan.shape)}.")
        self.plan = plan

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        plan = self.plan.to(device=self.weight.device, dtype=self.weight.dtype)
        weight = torch.einsum("pn,noi->poi", plan, self.weight).flatten(0, 1)
        bias = None if self.bias is None else torch.einsum("pn,no->po", plan, self.bias).flatten()
        return F.linear(hidden_states, weight, bias)


class LoRALinear(nn.Module):
    """Frozen `nn.Linear` plus `y += (alpha / rank) * B A x`."""

    # Diffusers' AdaLN casts activations with `get_parameter_dtype(linear)`. That
    # helper returns the first floating parameter, so skip the float32 adapters
    # and expose the frozen backbone dtype.
    _keep_in_fp32_modules = ["lora_down", "lora_up"]

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_down = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_up = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_down, a=5**0.5)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.base(hidden_states)
        update = F.linear(
            F.linear(hidden_states, self.lora_down.to(device=hidden_states.device, dtype=hidden_states.dtype)),
            self.lora_up.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        return out + self.scaling * update.to(out.dtype)


def attach_parallel_decoder(transformer: nn.Module, num_steps: int) -> None:
    transformer.proj_out = MiniMaxH3ParallelHead(transformer.proj_out, num_steps)
    transformer.audio_proj_out = MiniMaxH3ParallelHead(transformer.audio_proj_out, num_steps)


def add_lora(module: nn.Module, target_names: Sequence[str], rank: int, alpha: float) -> int:
    targets = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear) and any(name.endswith(suffix) for suffix in target_names)
    ]
    for name, child in targets:
        parent_name, _, attribute = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attribute, LoRALinear(child, rank, alpha))
    return len(targets)


def set_parallel_plan(transformer: nn.Module, video_plan: torch.Tensor, audio_plan: torch.Tensor) -> None:
    transformer.proj_out.set_plan(video_plan)
    transformer.audio_proj_out.set_plan(audio_plan)


def _load_pdd_config(config_path: Path | None) -> dict:
    config = dict(DEFAULT_PDD_CONFIG)
    if config_path is None or not config_path.is_file():
        return config
    with config_path.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    for key in config:
        if key in saved:
            config[key] = saved[key]
    if not isinstance(config["lora_targets"], str):
        config["lora_targets"] = ",".join(config["lora_targets"])
    return config


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(path, device="cpu")
    else:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state-dict mapping in {path}, got {type(checkpoint).__name__}.")
    return checkpoint


def resolve_pdd_checkpoint(path: str) -> tuple[Path, Path | None]:
    """Return `(weights_path, config_path)` from a local file/folder or a Hub repo id."""
    local = Path(path).expanduser()
    if local.is_file():
        return local.resolve(), local.resolve().parent / "pdd_config.json"
    if local.is_dir():
        for name in ("pdd.pt", "pdd.safetensors"):
            candidate = local / name
            if candidate.is_file():
                return candidate.resolve(), (local / "pdd_config.json").resolve()
        raise FileNotFoundError(f"No pdd.pt or pdd.safetensors in {local}")
    if "/" not in path:
        raise FileNotFoundError(
            f"PDD checkpoint not found: {path}. Pass a local pdd.pt / folder, or a Hugging Face repo id."
        )

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        weights = Path(hf_hub_download(path, "pdd.pt"))
    except (EntryNotFoundError, RepositoryNotFoundError):
        weights = Path(hf_hub_download(path, "pdd.safetensors"))
    try:
        config = Path(hf_hub_download(path, "pdd_config.json"))
    except EntryNotFoundError:
        config = None
    return weights, config


class _PDDStepArm:
    """Arm the fused heads before each transformer forward, then advance one block."""

    def __init__(
        self,
        transformer: nn.Module,
        video_steps: torch.Tensor,
        audio_steps: torch.Tensor,
        block_size: int,
        nfe: int,
    ):
        self.transformer = transformer
        self.video_steps = video_steps
        self.audio_steps = audio_steps
        self.block_size = block_size
        self.nfe = nfe
        self.index = 0
        self.arm(0)

    def arm(self, step_index: int) -> None:
        start = step_index * self.block_size
        set_parallel_plan(
            self.transformer,
            pdd_sampling_plan(self.video_steps, start, self.block_size).float(),
            pdd_sampling_plan(self.audio_steps, start, self.block_size).float(),
        )

    def __call__(self, _module, _args, output):
        self.index += 1
        if self.index < self.nfe:
            self.arm(self.index)
        else:
            self.index = 0
            self.arm(0)
        return output


def apply_pdd_lora(transformer: nn.Module, checkpoint: str, video_shift: float, audio_shift: float) -> int:
    """Inject PDD LoRA + parallel heads, load weights, and arm them on every forward.

    Args:
        transformer: MiniMax-H3 `transformer` or `transformer_ref`.
        checkpoint: Local `pdd.pt` / folder / Hugging Face repo id.
        video_shift, audio_shift: Scheduler shifts (12.0 / 3.0 as released).

    Returns:
        `nfe`, the number of transformer evaluations. Diffusers' MiniMax-H3
        scheduler counts the terminal sigma, so call the pipeline with
        `num_inference_steps=nfe + 1`.
    """
    weights_path, config_path = resolve_pdd_checkpoint(checkpoint)
    config = _load_pdd_config(config_path)
    num_steps = int(config["pdd_num_steps"])
    block_size = int(config["pdd_block_size"])
    if block_size < 1 or num_steps % block_size != 0:
        raise ValueError(f"pdd_num_steps={num_steps} must be divisible by pdd_block_size={block_size}.")
    nfe = num_steps // block_size

    add_lora(
        transformer,
        config["lora_targets"].split(","),
        int(config["lora_rank"]),
        float(config["lora_alpha"]),
    )
    attach_parallel_decoder(transformer, num_steps)

    state_dict = _load_state_dict(weights_path)
    incompatible = transformer.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        preview = ", ".join(incompatible.unexpected_keys[:3])
        raise RuntimeError(
            f"{weights_path} holds keys the parallel decoder does not have, e.g. {preview}."
        )
    print(
        f"Loaded PDD LoRA: {weights_path} ({len(state_dict)} tensors, "
        f"grid={num_steps}, block={block_size}, nfe={nfe})",
        flush=True,
    )

    video_steps = pdd_time_grid(float(video_shift), num_steps).diff()
    audio_steps = pdd_time_grid(float(audio_shift), num_steps).diff()
    controller = _PDDStepArm(transformer, video_steps, audio_steps, block_size, nfe)
    transformer.register_forward_hook(controller)
    transformer._pdd_step_arm = controller
    transformer.requires_grad_(False)
    transformer.eval()
    return nfe

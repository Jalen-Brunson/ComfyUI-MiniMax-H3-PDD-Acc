#!/usr/bin/env python3
"""Convert an alibaba-pai MiniMax-H3-Acc-LoRAs file to the ComfyUI redistribution format.

Standalone (torch + safetensors only, no ComfyUI needed):

    python3 convert_pdd_acc.py MiniMax-H3-FL2VA-Acc-8Step.safetensors \
        minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors

Output file = the trunk LoRA renamed to ComfyUI H3 keys
(diffusion_model.*.lora_A/B.weight + .alpha; qkv fused block-diagonal, SwiGLU
half-swap on fc1, refiner_blocks -> blocks, adaln 1:1) + the untouched PDD head
bank (proj_out / audio_proj_out stacks) + provenance metadata. Both the output
and the original load in ComfyUI through the MiniMaxH3PDDAccApply node; the
converted file additionally lets the trunk half be inspected with standard
ComfyUI LoRA tooling.
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdd_acc_core import CONVERTED_FORMAT, HEAD_KEYS, convert_pdd_lora  # noqa: E402


def read_metadata(path):
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header.get("__metadata__", {}) or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    meta = read_metadata(args.src)
    sd = load_file(args.src)

    heads = {}
    for k in HEAD_KEYS:
        if k not in sd:
            raise SystemExit(f"{args.src} has no '{k}' — not a PDD Acc file?")
        heads[k] = sd.pop(k)

    alpha = float(meta.get("lora_alpha", 64.0))
    lora_sd, leftovers = convert_pdd_lora(sd, alpha)
    if leftovers:
        raise SystemExit(f"{len(leftovers)} unrecognized keys, e.g. {sorted(leftovers)[:4]}")
    modules = sum(1 for k in lora_sd if k.endswith(".lora_A.weight"))

    with open(args.src, "rb") as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()

    out_meta = {
        "format": CONVERTED_FORMAT,
        "pdd_num_steps": meta.get("pdd_num_steps", "32"),
        "pdd_block_size": meta.get("pdd_block_size", "4"),
        "lora_rank": meta.get("lora_rank", "64"),
        "lora_alpha": meta.get("lora_alpha", "64.0"),
        "source_repo": "alibaba-pai/MiniMax-H3-Acc-LoRAs",
        "source_file": os.path.basename(args.src),
        "source_sha256": src_sha,
        "conversion": ("to_q/k/v -> attn.qkv_proj (concat lora_A, block-diagonal lora_B, "
                       "alpha x3); ff.net.0.proj -> mlp.fc1 (SwiGLU [value;gate] -> "
                       "[gate;value] lora_B row half-swap); ff.net.2 -> mlp.fc2; to_out.0 -> "
                       "attn.out_proj; adaln_proj.linear copied 1:1; "
                       "token_refiner.refiner_blocks -> token_refiner.blocks; head bank "
                       "(proj_out/audio_proj_out) unchanged"),
        "converter": "https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc",
        "license": "apache-2.0",
    }

    out = dict(lora_sd)
    out.update(heads)
    save_file(out, args.dst, metadata=out_meta)
    size = os.path.getsize(args.dst) / 1e9
    print(f"wrote {args.dst}: {len(out)} tensors ({modules} lora modules + head bank), "
          f"{size:.2f} GB")


if __name__ == "__main__":
    main()

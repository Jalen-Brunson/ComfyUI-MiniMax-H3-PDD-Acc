#!/usr/bin/env python3
"""Bake a trunk partition fingerprint from a MiniMax-H3 checkpoint.

Extracts final_layer.video_out.weight — fp32-unquantized in every published
build and bit-identical across the int8_convrot / fp8_scaled / pruned /
adaln-rebased variants of one trunk (measured 2026-08-27) — and stores it fp16
(~1 MB, adds ~2e-4 relative noise vs the 0.0503 fl2va<->ref2va gap). The Apply
node compares the loaded model's tensor against every shipped fingerprint by
relative Frobenius distance to refuse an FL2VA file on a ref2va UNET (and vice
versa), which otherwise applies cleanly and renders silently wrong because the
two trunks share identical key sets.

Usage:
    python3 bake_partition_fingerprint.py <checkpoint.safetensors> <trunk>
    # e.g. ... minimax_h3_fl2va_int8_convrot.safetensors fl2va

Writes partition_fingerprints/video_out_<trunk>.safetensors next to this file.
"""

import argparse
import hashlib
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

KEY = "final_layer.video_out.weight"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("trunk", help="partition name, e.g. fl2va or ref2va")
    args = ap.parse_args()

    with safe_open(args.checkpoint, framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith(KEY)]
        if len(keys) != 1:
            raise SystemExit(f"{args.checkpoint}: expected exactly one '*{KEY}', got {keys}")
        w = f.get_tensor(keys[0]).to(torch.float32)

    sha = hashlib.sha256(w.numpy().tobytes()).hexdigest()[:16]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "partition_fingerprints")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"video_out_{args.trunk}.safetensors")
    save_file({"video_out_weight": w.to(torch.float16)}, dst, metadata={
        "partition": args.trunk,
        "tensor": keys[0],
        "source_file": os.path.basename(args.checkpoint),
        "fp32_sha256_16": sha,
        "note": ("fp16 copy of the trunk's fp32 final_layer.video_out.weight; "
                 "compared by relative Frobenius distance, tolerance 0.015"),
    })
    err = float((w - w.half().float()).norm() / w.norm())
    print(f"wrote {dst}: {list(w.shape)} fp16, fp32 sha {sha}, "
          f"fp16 storage error {err:.2e}")


if __name__ == "__main__":
    main()

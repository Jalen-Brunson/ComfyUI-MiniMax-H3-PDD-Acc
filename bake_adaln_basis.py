#!/usr/bin/env python3
"""Bake an adaln curve basis for pruned (curve-form) MiniMax-H3 checkpoints.

Pruned H3 checkpoints replace the dense per-block adaln (`[96768, 2688]` on
silu(time_embedder(t))) with a shared `adaln_t_table [1025, 8]` (rows = the
8-dim curve coordinates of silu(t_emb(t)) at t = j/1024) and `[96768, 8]`
projections. To apply a DENSE adaln LoRA to such a model, the dense delta must
be expressed in table coordinates via the affine map

    silu(t_emb(t))  ≈  c + V @ table(t),      V [2688, 8], c [2688]

solved by least squares over the 1025 grid rows. This script computes (c, V)
from a matching FULL checkpoint's time_embedder (fp32 tensors, tiny) and the
canonical table, and stores them with the table + fit residual:

    python3 bake_adaln_basis.py minimax_h3_fl2va_int8_convrot.safetensors \
        adaln_basis/table_fl2va.npy adaln_basis/basis_fl2va.safetensors --trunk fl2va

The two shipped basis files cover the entire ecosystem: exactly two adaln
tables exist in the wild (one per trunk, byte-identical across every pruned
release — Comfy-Org, koongrizzly w4a8, xmarre fused all share them).
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdd_acc_core import table_sha  # noqa: E402

TE_KEYS = ("time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
           "time_embedder.proj_out.weight", "time_embedder.proj_out.bias")


def silu_temb_grid(win, bin_, wout, bout, num_rows):
    """u(t_j) = silu(time_embedder(t_j)), t_j = j/(num_rows-1) — verbatim core math.

    comfy/ldm/minimax/model.py TimeEmbedder.forward: half = freq_dim//2,
    freqs = exp(-log(10000)*arange(half)/half), emb = cat[cos(t*f), sin(t*f)]
    (cos BEFORE sin), temb = proj_out(silu(proj_in(emb))); AdalnProj then
    applies one more silu on the dense path (apply_silu=True).
    """
    freq_dim = win.shape[1]
    half = freq_dim // 2
    t = torch.arange(num_rows, dtype=torch.float64) / (num_rows - 1)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    hidden = torch.nn.functional.silu(emb @ win.double().T + bin_.double())
    temb = hidden @ wout.double().T + bout.double()
    return torch.nn.functional.silu(temb)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("full_checkpoint", help="full (dense) checkpoint with fp32 time_embedder keys")
    ap.add_argument("table", help="adaln_t_table source: .npy, or a .safetensors holding adaln_t_table")
    ap.add_argument("out", help="output basis .safetensors")
    ap.add_argument("--trunk", required=True, help="trunk label (fl2va / ref2va)")
    args = ap.parse_args()

    tensors = {}
    with safe_open(args.full_checkpoint, framework="pt") as f:
        for k in TE_KEYS:
            t = f.get_tensor(k)
            if t.dtype != torch.float32:
                raise SystemExit(f"{k} is {t.dtype}, expected fp32 (quantized time_embedder "
                                 f"needs proper dequantization — use a full fp32/bf16 checkpoint)")
            tensors[k] = t.to(torch.float32).clone()

    if args.table.endswith(".npy"):
        table = torch.from_numpy(np.load(args.table)).to(torch.float64)
    else:
        table = load_file(args.table)["adaln_t_table"].to(torch.float64)
    rows, k = table.shape

    u = silu_temb_grid(tensors[TE_KEYS[0]], tensors[TE_KEYS[1]],
                       tensors[TE_KEYS[2]], tensors[TE_KEYS[3]], rows)   # [rows, 2688] f64
    X = torch.cat([torch.ones(rows, 1, dtype=torch.float64), table], dim=1)  # [rows, k+1]
    sol = torch.linalg.lstsq(X, u).solution                                  # [k+1, 2688]
    resid = torch.linalg.norm(X @ sol - u) / torch.linalg.norm(u)
    c = sol[0].contiguous()          # [2688]
    V = sol[1:].T.contiguous()       # [2688, k]

    save_file(
        {"c": c.to(torch.float32), "V": V.to(torch.float32),
         "adaln_t_table": table.to(torch.float32)},
        args.out,
        metadata={"trunk": args.trunk, "residual": f"{float(resid):.3e}",
                  "table_sha": table_sha(table), "grid": f"t_j=j/{rows-1}, j=0..{rows-1}",
                  "source_checkpoint": args.full_checkpoint.split("/")[-1],
                  "desc": "silu(t_emb(t)) ~= c + V @ adaln_t_table(t), lstsq over all rows (f64)"})
    print(f"{args.trunk}: residual {float(resid):.3e}, table_sha {table_sha(table)}, "
          f"c {list(c.shape)}, V {list(V.shape)} -> {args.out}")


if __name__ == "__main__":
    main()

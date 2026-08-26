# ComfyUI-MiniMax-H3-PDD-Acc

Native ComfyUI support for the **official MiniMax-H3 8-step PDD acceleration LoRAs**
([alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)) —
full audio+video generation in **8 (or 4) sampler steps**, no CFG.

These files are *not* ordinary LoRAs: alongside a rank-64 trunk LoRA they carry a
**Parallel Decoding Distillation head bank** — 32 per-interval copies of the final-layer
video/audio projections that get fused into one mean-block-velocity head per sampler step
([PDD, Shaul et al. 2026](https://arxiv.org/abs/2607.26004)). A plain LoRA loader can't read
them, and dropping the head bank silently loses the distill. This pack loads the whole thing.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc
```

Put the PDD file(s) in `ComfyUI/models/pdd_acc/` (the folder is created on first launch).
Either release works — the loader auto-detects the format:

| Source | Files |
|---|---|
| Original (alibaba-pai) | [MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs): `MiniMax-H3-FL2VA-Acc-8Step.safetensors`, `MiniMax-H3-Ref2VA-Acc-8Step.safetensors` |
| Pre-converted ComfyUI keys | [aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI](https://huggingface.co/aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI): `minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors`, `minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors` |

Pair **FL2VA** with an fl2va UNET and **Ref2VA** with a ref2va UNET (bf16 originals or int8
convrot builds both work — LoRA application goes through ComfyUI's quant-aware patch path).

## Nodes

### MiniMax H3 PDD Acc LoRA (Apply) — `MiniMaxH3PDDAccApply`
`MODEL → MODEL + SIGMAS + info`. One node does everything: applies the trunk LoRA
(converting diffusers keys to ComfyUI naming in memory when given an original-format file)
and installs the PDD head bank on `final_layer`, armed per step **by sigma** — so looping /
chunked samplers, resumes and split schedules can't desync it.

- **nfe** — model evaluations. `8` = trained block size (default). `4` regroups two blocks
  per step (officially sanctioned — the release demos both). `6` uses the non-uniform default
  partition `8,8,4,4,4,4` (the two merged size-8 blocks sit at high sigma where the block
  boundaries span almost no sigma, and the late heavyweight blocks stay at trained size —
  every knot stays on the trained fine grid). `16`/`32` use shorter blocks.
- **partition** (optional) — custom block sizes in fine steps, comma-separated, summing to 32
  (e.g. `8,4,4,4,4,4,4` for 7 steps). Overrides `nfe`; the sigmas output follows. Sizes 4 and 8
  on multiple-of-4 starts stay inside the officially demonstrated envelope.
- **lora_strength / head_strength** — trained at 1.0 / 1.0.
- **on_off_grid** — `error` (default): refuse evaluation at sigmas that are not trained block
  boundaries, with a message telling you what to fix. `clamp`: nearest block, degraded output.

### MiniMax H3 PDD Acc Scheduler — `MiniMaxH3PDDAccScheduler`
Standalone SIGMAS emitter for partial-denoise / split-sigma workflows. At `denoise 1.0` it
equals the Apply node's sigmas output.

## Required recipe

| Setting | Value | Why |
|---|---|---|
| Sampler | **euler** (KSamplerSelect) | each step consumes one mean block velocity; multi-stage samplers (er_sde, dpmpp, res_*) evaluate off-grid |
| Sigmas | the Apply node's **sigmas output** → SamplerCustomAdvanced | trained boundaries `12t/(1+11t)`, `t = linspace(1,0,nfe+1)` |
| Guidance | **CFG 1.0** (BasicGuider) | guidance is distilled in; single forward per step |
| SigmaShift | **12.0 / 3.0** exactly | the training grid; the node fails closed otherwise |

**Remove** other distill LoRAs (lightx2v turbo etc.) — distills don't stack. Character LoRAs
stack normally. **Do not stack** step-caching packs (blockcache / EasyCache — the final-layer
patch fails closed, and an 8-step distill has nothing to cache anyway).

## Pruned checkpoints

Pruned H3 UNETs (Comfy-Org `*_pruned_*`, the GGUF/w4a8/nvfp4 re-quants of them) replace the
dense adaln with a shared 8-dim curve table — a dense adaln LoRA can't patch them, which is
why plain loaders spam ~50 `ERROR lora ... adaln_proj` lines and silently drop that part of
the distill. This pack handles it: on a pruned model the 50 adaln LoRA modules are
**rebased onto the model's curve basis** (weight diff `B(AV)` + the mandatory DC bias diff
`B(Ac)`, from the affine fit `silu(t_emb(t)) ≈ c + V·table(t)` solved in float64 against a
matching full checkpoint — fit residual ~1.4e-5, effectively exact). The two shipped bases in
`adaln_basis/` cover every pruned release in the wild (exactly two adaln tables exist — one
per trunk, byte-identical across Comfy-Org / w4a8 / fused repacks); the node matches the
model's table automatically and warns on trunk mismatches. If a future release ships a new
table, bake a basis with `bake_adaln_basis.py` (see its docstring).

## Example workflow

[`example_workflows/pdd_acc_t2v_basic.json`](example_workflows/pdd_acc_t2v_basic.json) —
prompt-to-video+audio in 8 steps (Ref2VA trunk, zero references; wire images into
`ref_image_0…` for identity-locked r2v). Drag into ComfyUI.

## Converter (optional)

`convert_pdd_acc.py` produces the pre-converted redistribution format from an original file —
standalone, no ComfyUI required:

```bash
python3 convert_pdd_acc.py MiniMax-H3-FL2VA-Acc-8Step.safetensors \
    minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors
```

Output is bit-identical to what the loader computes in memory (tested), with the trunk LoRA
in standard `diffusion_model.*.lora_A/B.weight` + `.alpha` keys and full provenance metadata.

## How it works (short version)

- **LoRA conversion** (verified against both codebases' sources): `to_q/to_k/to_v` fuse into
  ComfyUI's `attn.qkv_proj` (concatenated `lora_A`, block-diagonal `lora_B`, alpha ×3);
  `ff.net.0.proj → mlp.fc1` with the SwiGLU `[value;gate] → [gate;value]` half-swap;
  `to_out.0 → attn.out_proj`, `ff.net.2 → mlp.fc2`, `adaln_proj.linear` 1:1 (layouts are
  bit-identical); `token_refiner.refiner_blocks → token_refiner.blocks`.
- **Head bank**: per-block plans (fine step sizes normalized per modality on the shift-12
  video / shift-3 audio grids) fuse the 32 heads into `nfe` fused fp32 heads at load —
  identical math to the reference `minimax_h3_pdd.py` einsum. A `DIFFUSION_MODEL` wrapper
  stashes the current sigma; an object patch on `final_layer.forward` selects the block and
  runs the fused projections (everything else in the layer is untouched).
- **Audio needs no extra conversion** on current ComfyUI core: the model's carried-audio
  mapping integrates a *mean* block velocity exactly over finite Euler steps
  (`s·Δσ_v/(c_i·c_j) = Δσ_a` is an algebraic identity since `c` is linear in σ — unit-tested).

## Tests

```bash
python3 tests/test_pdd_acc.py          # torch-only, no ComfyUI needed
PDD_ACC_SLOW=1 python3 tests/test_pdd_acc.py   # + full-tensor checks on the real files
```

13 tests: grid/plan/fusion vs the verbatim reference implementation shipped in the official
repo, boundary sigmas vs diffusers `set_timesteps`, qkv block-diag + SwiGLU swap numerics,
the carried-audio exactness identity, dual-format round-trip, and structural checks against
the real safetensors headers.

## Credits & license

- Acceleration LoRAs: [alibaba-pai](https://huggingface.co/alibaba-pai) (Apache-2.0);
  `tests/reference_minimax_h3_pdd.py` is their reference loader, kept verbatim as test oracle.
- Method: [Parallel Decoding Distillation](https://arxiv.org/abs/2607.26004), Shaul et al.
- Base model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

This pack: Apache-2.0.

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

**ComfyUI version:** v0.33.0 or newer (the MiniMax-H3 carried-audio mechanics,
comfyanonymous/ComfyUI#15243 — the node fails closed with an update message on older cores).
Both pre- and post-#15375 cores work; the final-layer patch delegates to your core's own
forward rather than replicating its internals.

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
  every knot stays on the trained fine grid).
- **partition** (optional) — custom block sizes in fine steps, comma-separated, summing to 32
  (e.g. `8,4,4,4,4,4,4` for 7 steps). Overrides `nfe`; the sigmas output follows.
- **Only block sizes 4 and 8 are legal** — the training envelope. PDD heads are conditioned on
  trunk features from block *starts* on the L_min=4 grid with blocks of 4 or 8 fine steps;
  evaluating the trunk anywhere else feeds the heads features they never trained on and renders
  as heavy noise (community-reported at 32 steps on FL2VA, reproduced locally on plain SDPA —
  it is not an attention-backend issue). The node therefore rejects off-envelope step counts
  and partitions instead of letting them degrade. More steps than 8 is not "closer to the
  teacher" here: the per-interval heads are only ever decoded from envelope block starts.
- **lora_strength / head_strength** — trained at 1.0 / 1.0.
- **on_off_grid** — `error` (default): refuse evaluation at sigmas that are not trained block
  boundaries, with a message telling you what to fix. `clamp`: nearest block, degraded output.
- **enabled** (optional, default true) — `false` = full bypass: the input model and
  `bypass_sigmas` pass through untouched (nothing is loaded or patched). Wire a boolean node
  here to A/B the distill or drive a subgraph toggle.
- **bypass_sigmas** (optional SIGMAS) — returned as the sigmas output when `enabled=false`
  (wire the schedule for the un-distilled model, e.g. a BasicScheduler). The node errors if
  you disable it without wiring this — the PDD block boundaries would be a wrong schedule for
  an unpatched model. Remember the rest of the un-distilled recipe (CFG, sampler, steps)
  differs too.

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

## Trunk pairing guard (partition fingerprint)

The FL2VA and Ref2VA trunks ship **identical tensor key sets**, so pairing an FL2VA distill
with a ref2va UNET (or vice versa) applies cleanly and renders **silently wrong**. The Apply
node now identifies the loaded model's trunk from its `final_layer.video_out.weight` — that
tensor is fp32-unquantized in every published build, bit-identical across the
int8_convrot/pruned/rebased variants of one trunk, and the two trunks sit 0.0503 apart in
relative Frobenius distance (fingerprints shipped fp16 in `partition_fingerprints/`,
tolerance 0.015 ≫ cast/storage noise ~2e-3). A confident mismatch **errors**; set the
optional `partition_check` input to `warn` for deliberate cross-trunk experiments. A
finetune or full-merge that matches neither fingerprint just logs "inconclusive" and
proceeds — the guard never blocks checkpoints it has no fingerprint for. New trunks:
`python3 bake_partition_fingerprint.py <checkpoint> <name>`.

Guard design after [fblissjr/ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations),
which shipped a partition fingerprint first.

## Pruned checkpoints

Pruned H3 UNETs (Comfy-Org `*_pruned_*`, the GGUF/w4a8/nvfp4 re-quants of them) replace the
dense adaln with a shared 8-dim curve table — a dense adaln LoRA can't patch them, which is
why plain loaders spam ~50 `ERROR lora ... adaln_proj` lines and silently drop that part of
the distill. This pack handles it: on a pruned model the 50 adaln LoRA modules are
**rebased onto the model's curve basis** (weight diff `B(AV)` + the mandatory DC bias diff
`B(Ac)`, from the affine fit `silu(t_emb(t)) ≈ c + V·table(t)` solved in float64 against a
matching full checkpoint — fit residual ~1.4e-5, effectively exact). The node matches the
model's adaln table against the two shipped bases in `adaln_basis/` (one per trunk)
automatically and warns on trunk mismatches.

A repacked or requantized pruned build may carry a table that is **not byte-identical** to
the Comfy-Org ones but still describes the same trunk's curve. The node handles that too:
when no exact match is found it **auto-refits** each shipped basis onto the model's table
(rows of every table sample the same fixed timestep grid, so this is a float64 least-squares
fit) and accepts the best fit when the residual is same-trunk small (~1e-5; a genuinely
different finetune lands around 1e-1 and is refused). If your pruned checkpoint is refused,
it is not a repack of a known trunk — bake a basis with `bake_adaln_basis.py` (see its
docstring) or open an issue naming the exact checkpoint file/source so a basis can be
shipped.

**Hybrid trunks (fl2va+ref2va block merges):** a hybrid carries ONE adaln table — its BASE
trunk's — so e.g. an fl2va-based `b15-49` hybrid matches the fl2va basis and pairing it with
the Ref2VA PDD file logs a trunk-mismatch warning. If the hybrid pairing is what you intend,
the warning is informational: PDD fully applies (the node fails closed if any patch key
misses, and sampling would error — not silently skip PDD — if the heads were not armed;
check the `info` output for the applied module count). But hybrids are **off-label for
PDD**: the trunk LoRA and head bank were trained on the pure trunks, so quality on a merge
is untested. If output looks weak or wrong, A/B against the matching plain trunk before
blaming settings.

## Example workflows

- [`example_workflows/pdd_acc_t2v_basic.json`](example_workflows/pdd_acc_t2v_basic.json) —
  prompt-to-video+audio in 8 steps (Ref2VA trunk, zero references; wire images into
  `ref_image_0…` for identity-locked r2v). Drag into ComfyUI.
- [`example_workflows/pdd_acc_t2v_warmup_split.json`](example_workflows/pdd_acc_t2v_warmup_split.json) —
  two-phase warmup for better reference likeness: the Warmup Scheduler's sigmas are split at
  its `phase2_start_step` with core `SplitSigmas`; pass 1 samples the un-distilled BASE model
  over the warmup segment, pass 2 chains its `output` latent (via `DisableNoise`) into the
  PDD-patched model for the trained tail.
- [`example_workflows/pdd_acc_t2v_latent_upscale.json`](example_workflows/pdd_acc_t2v_latent_upscale.json) —
  two-pass latent upscale (hi-res fix): full 8-step PDD render at 896x512, then
  `MiniMax H3 AV Latent Upscale By` x1.5 (lands exactly on the model-native 1344x768) into a
  partial-denoise PDD pass — the `PDD Acc Scheduler` with `denoise 0.25` re-runs only the
  LAST 2 trained blocks (resume sigma 0.8), so the refine stays on the trained grid
  (0.125 = 1 block/subtle, 0.375 = 3 blocks/strong). Audio is decoded from pass 1 and is
  untouched by the refine. The upscale node exists because core `LatentUpscale` cannot
  handle H3's nested video+audio latent; it resizes the video half per frame (audio passes
  through) and snaps to the model's 2x2 patch grid.

- [`example_workflows/pdd_video_upscale_long.json`](example_workflows/pdd_video_upscale_long.json) —
  **upscale an EXISTING video of any length** (video-to-video hi-res fix): load by path,
  VAE-encode, neural latent upscale to 1344x768, then a PDD partial-denoise refine
  (`denoise 0.25` = last 2 trained blocks) sampled in 73-frame windows with 22-frame
  overlap and anchor frames — so a 70s clip refines in ~25 cheap windows instead of one
  quadratic-cost 500k-token pass. The source audio is muxed straight through untouched.
  The neural upscaler and the windowed refine sampler (`MinimaxH3LatentUpscaler3D`,
  `MMH3SplitUpscale`, `MMH3TemporalSplitParams`) are by **LBH-123-AI** — install their
  [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
  node pack and download the
  [upscaler model](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) into
  `models/latent_upscale_models/`. Also needs
  [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for the video load.
  Using the mamad-converted PDD file instead: set nfe 4 on BOTH the Apply node and the Scheduler.

## Converter (optional)

`convert_pdd_acc.py` produces the pre-converted redistribution format from an original file —
standalone, no ComfyUI required:

```bash
python3 convert_pdd_acc.py MiniMax-H3-FL2VA-Acc-8Step.safetensors \
    minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors
```

Output is bit-identical to what the loader computes in memory (tested), with the trunk LoRA
in standard `diffusion_model.*.lora_A/B.weight` + `.alpha` keys and full provenance metadata.

## Baking the trunk (optional — for cards where the model doesn't fully fit)

ComfyUI merges LoRA patches into weights **once at load** — but only for modules that fit
in VRAM. Offloaded modules get a per-forward patch instead: the LoRA (plus a dequantize) is
re-applied on **every step**. On a card at the VRAM edge that fixed per-step cost is large
at low resolutions (issue #4 measured ~2× s/it at 864×480 on a 32GB RTX 5090) and vanishes
into attention time at high ones. **If your model fully loads, baking buys you nothing** —
the runtime path already costs zero per step there.

Measured 2026-08-30 (H200, ref2va int8_convrot, 832×480×124f, 8 steps, SDPA; offload forced
with `--reserve-vram 120` so the whole 32.4GB trunk streams — the log's `lowvram patches`
count is the mechanism made visible):

| regime | runtime patches | baked trunk |
|---|---|---|
| fully offloaded | 2.44 s/it (`lowvram patches: 258`) | **2.06 s/it** (`lowvram patches: 0`) |
| fully loaded | 1.82 s/it | 1.75 s/it (same within noise) |

The bake removes the per-forward patch term entirely; what remains in the offloaded row is
pure weight streaming, which both arms pay equally. The patch term is hardware-dependent —
~0.4 s/step on an H200, ~2.5 s/step in the issue-#4 5090 report — so the win grows as the
card gets smaller, which is exactly who this is for.

`bake_pdd_trunk.py` merges the trunk LoRA (and the adaln update — curve-rebased first on
pruned bases) into the quantized checkpoint offline, using the same `comfy-kitchen` kernels
ComfyUI dequantizes with. Only the head bank stays runtime — it swaps `final_layer` per
fine-interval and cannot be baked. The write is streaming (peak RAM is one module, not one
checkpoint) and every tensor keeps its exact dtype, shape and byte length:

```bash
# audit first (no write): requant error per sampled module
python3 bake_pdd_trunk.py --check \
    --base minimax_h3_ref2va_int8_convrot.safetensors \
    --pdd  MiniMax-H3-Ref2VA-Acc-8Step.safetensors

python3 bake_pdd_trunk.py \
    --base minimax_h3_ref2va_int8_convrot.safetensors \
    --pdd  MiniMax-H3-Ref2VA-Acc-8Step.safetensors \
    --out  minimax_h3_ref2va_pddbaked_int8_convrot.safetensors
```

Load the baked file with a normal UNETLoader and set the Apply node's **`lora_strength`
to `0.0`** (baked-trunk mode: trunk patching skipped, heads/sigmas/guards unchanged; the
info output says so). Caveats: the strength is frozen into the file (re-bake to change
it); on an **unbaked** trunk, strength 0.0 renders the un-distilled model with PDD heads —
nothing can detect that, so check your file's `pdd_acc_baked` metadata if unsure. GGUF and
non-convrot formats are refused, not guessed. Requantizing the merged weight costs the
same class of error the runtime merge pays (it also requantizes); the `--check` audit
prints the measured number for your files.

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

#!/usr/bin/env python3
"""Generate example_workflows/pdd_acc_t2v_basic.json (browser workflow format)."""

import json
import os

NODES = []
LINKS = []
_link_id = 0


def link(src, srcslot, dst, dstslot, ltype):
    global _link_id
    _link_id += 1
    LINKS.append([_link_id, src, srcslot, dst, dstslot, ltype])
    return _link_id


def node(nid, ntype, pos, size, widgets=None, inputs=None, outputs=None, title=None, mode=0):
    n = {
        "id": nid, "type": ntype, "pos": list(pos), "size": list(size),
        "flags": {}, "order": nid, "mode": mode,
        "inputs": inputs or [], "outputs": outputs or [],
        "properties": {"Node name for S&R": ntype},
    }
    if widgets is not None:
        n["widgets_values"] = widgets
    if title:
        n["title"] = title
    NODES.append(n)
    return n


PROMPT = """summary:
A golden retriever runs along a sunlit beach at golden hour, kicking up sand, then splashes into shallow surf chasing a thrown red ball.

detailed_description:
Photorealistic handheld tracking shot, warm golden-hour light, gentle waves. The dog bounds joyfully along the wet sand, ears flapping, catches up to a red rubber ball at the water's edge and splashes into the shallow surf.

overall_soundscape:
Waves washing ashore, rapid paw impacts on wet sand, energetic dog panting, one playful bark, water splashes.

non_diegetic_music:
Light acoustic guitar, upbeat and warm."""

# loaders
unet = node(1, "UNETLoader", (-60, 60), (380, 82),
            widgets=["minimax_h3_ref2va_bf16.safetensors", "default"],
            outputs=[{"name": "MODEL", "type": "MODEL", "links": [], "slot_index": 0}])
clip = node(2, "CLIPLoader", (-60, 200), (380, 106),
            widgets=["qwen3vl_32b_minimax_h3_bf16.safetensors", "minimax", "default"],
            outputs=[{"name": "CLIP", "type": "CLIP", "links": [], "slot_index": 0}])
vvae = node(3, "VAELoader", (-60, 360), (380, 58),
            widgets=["minimax_h3_video_vae_fp16.safetensors"],
            outputs=[{"name": "VAE", "type": "VAE", "links": [], "slot_index": 0}])
avae = node(4, "VAELoader", (-60, 470), (380, 58),
            widgets=["minimax_h3_audio_vae_fp32.safetensors"],
            outputs=[{"name": "VAE", "type": "VAE", "links": [], "slot_index": 0}])

shift = node(5, "MiniMaxH3SigmaShift", (380, 60), (270, 82), widgets=[12, 3],
             inputs=[{"name": "model", "type": "MODEL", "link": None}],
             outputs=[{"name": "MODEL", "type": "MODEL", "links": [], "slot_index": 0}])
pdd = node(6, "MiniMaxH3PDDAccApply", (700, 60), (330, 190),
           widgets=["MiniMax-H3-Ref2VA-Acc-8Step.safetensors", "8", 1.0, 1.0, "error"],
           inputs=[{"name": "model", "type": "MODEL", "link": None}],
           outputs=[{"name": "model", "type": "MODEL", "links": [], "slot_index": 0},
                    {"name": "sigmas", "type": "SIGMAS", "links": [], "slot_index": 1},
                    {"name": "info", "type": "STRING", "links": [], "slot_index": 2}])

r2v = node(7, "MiniMaxH3ReferenceToVideo", (380, 330), (430, 430),
           widgets=[PROMPT, 832, 480, 124, "match"],
           inputs=[{"name": "clip", "type": "CLIP", "link": None},
                   {"name": "vae", "type": "VAE", "link": None},
                   {"name": "audio_vae", "type": "VAE", "link": None},
                   {"name": "ref_images.ref_image_0", "type": "IMAGE", "link": None},
                   {"name": "ref_videos.ref_video_0", "type": "IMAGE", "link": None},
                   {"name": "ref_video_audios.ref_video_audio_0", "type": "AUDIO", "link": None},
                   {"name": "ref_audios.ref_audio_0", "type": "AUDIO", "link": None}],
           outputs=[{"name": "positive", "type": "CONDITIONING", "links": [], "slot_index": 0},
                    {"name": "LATENT", "type": "LATENT", "links": [], "slot_index": 1}])

guider = node(8, "BasicGuider", (1090, 200), (222, 66),
              inputs=[{"name": "model", "type": "MODEL", "link": None},
                      {"name": "conditioning", "type": "CONDITIONING", "link": None}],
              outputs=[{"name": "GUIDER", "type": "GUIDER", "links": [], "slot_index": 0}])
ksel = node(9, "KSamplerSelect", (1090, 320), (222, 58), widgets=["euler"],
            outputs=[{"name": "SAMPLER", "type": "SAMPLER", "links": [], "slot_index": 0}])
noise = node(10, "RandomNoise", (1090, 60), (222, 82), widgets=[7, "randomize"],
             outputs=[{"name": "NOISE", "type": "NOISE", "links": [], "slot_index": 0}])
sca = node(11, "SamplerCustomAdvanced", (1370, 60), (270, 130),
           inputs=[{"name": "noise", "type": "NOISE", "link": None},
                   {"name": "guider", "type": "GUIDER", "link": None},
                   {"name": "sampler", "type": "SAMPLER", "link": None},
                   {"name": "sigmas", "type": "SIGMAS", "link": None},
                   {"name": "latent_image", "type": "LATENT", "link": None}],
           outputs=[{"name": "output", "type": "LATENT", "links": [], "slot_index": 0},
                    {"name": "denoised_output", "type": "LATENT", "links": None, "slot_index": 1}])

vdec = node(12, "VAEDecode", (1700, 60), (200, 66),
            inputs=[{"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None}],
            outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0}])
adec = node(13, "VAEDecodeAudio", (1700, 180), (200, 66),
            inputs=[{"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None}],
            outputs=[{"name": "AUDIO", "type": "AUDIO", "links": [], "slot_index": 0}])
cvid = node(14, "CreateVideo", (1960, 60), (210, 100), widgets=[24, 8],
            inputs=[{"name": "images", "type": "IMAGE", "link": None},
                    {"name": "audio", "type": "AUDIO", "link": None}],
            outputs=[{"name": "VIDEO", "type": "VIDEO", "links": [], "slot_index": 0}])
svid = node(15, "SaveVideo", (2220, 60), (280, 380), widgets=["video/pdd_acc", "mp4", "auto"],
            inputs=[{"name": "video", "type": "VIDEO", "link": None}])
info = node(16, "PreviewAny", (700, 320), (330, 200), widgets=[""],
            title="PDD info / recipe",
            inputs=[{"name": "source", "type": "*", "link": None}])

node(17, "Note", (-60, 600), (380, 240), title="Models", widgets=[
    "UNET + text encoder: Comfy-Org/MiniMax-H3 (or any repack — int8 convrot builds work too). "
    "Pair the Ref2VA PDD file with a ref2va UNET, FL2VA with fl2va.\n\n"
    "PDD Acc file goes in models/pdd_acc/ — either the original "
    "alibaba-pai/MiniMax-H3-Acc-LoRAs release or the pre-converted ComfyUI copy "
    "(aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI). Both load.\n\n"
    "Optional: wire reference images into ref_image_0... for identity-locked r2v."])
node(18, "Note", (1370, 260), (270, 250), title="Recipe (do not change)", widgets=[
    "Sampler MUST be euler and sigmas MUST come from the Apply node (trained PDD block "
    "boundaries). CFG 1.0 (BasicGuider). SigmaShift 12/3.\n\n"
    "nfe 8 = trained; 4 also official. No turbo/distill LoRAs on top, no cache nodes "
    "(T8/EasyCache). Character LoRAs are fine.\n\n"
    "Wrong sampler/scheduler stops with an explanatory error (on_off_grid=error)."])

# wiring
def wire(srcnode, srcslot, dstnode, dstname, ltype):
    lid = link(srcnode["id"], srcslot, dstnode["id"],
               [i["name"] for i in dstnode["inputs"]].index(dstname), ltype)
    dstnode["inputs"][[i["name"] for i in dstnode["inputs"]].index(dstname)]["link"] = lid
    out = srcnode["outputs"][srcslot]
    if out["links"] is None:
        out["links"] = []
    out["links"].append(lid)

wire(unet, 0, shift, "model", "MODEL")
wire(shift, 0, pdd, "model", "MODEL")
wire(pdd, 0, guider, "model", "MODEL")
wire(pdd, 1, sca, "sigmas", "SIGMAS")
wire(pdd, 2, info, "source", "*")
wire(clip, 0, r2v, "clip", "CLIP")
wire(vvae, 0, r2v, "vae", "VAE")
wire(avae, 0, r2v, "audio_vae", "VAE")
wire(r2v, 0, guider, "conditioning", "CONDITIONING")
wire(r2v, 1, sca, "latent_image", "LATENT")
wire(noise, 0, sca, "noise", "NOISE")
wire(guider, 0, sca, "guider", "GUIDER")
wire(ksel, 0, sca, "sampler", "SAMPLER")
wire(sca, 0, vdec, "samples", "LATENT")
wire(sca, 0, adec, "samples", "LATENT")
wire(vvae, 0, vdec, "vae", "VAE")
wire(avae, 0, adec, "vae", "VAE")
wire(vdec, 0, cvid, "images", "IMAGE")
wire(adec, 0, cvid, "audio", "AUDIO")
wire(cvid, 0, svid, "video", "VIDEO")

wf = {
    "last_node_id": max(n["id"] for n in NODES),
    "last_link_id": _link_id,
    "nodes": NODES,
    "links": LINKS,
    "groups": [],
    "config": {},
    "extra": {},
    "version": 0.4,
}

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "example_workflows")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "pdd_acc_t2v_basic.json")
with open(out_path, "w") as f:
    json.dump(wf, f, indent=1)
print("wrote", out_path)

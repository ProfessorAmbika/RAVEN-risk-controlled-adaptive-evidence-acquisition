"""Acquisition rollouts and single-view runs for RAVEN on Qwen2.5-VL (V*, HR-Bench, MME-RealWorld).

For every sample it runs the full acquisition trajectory once:
  step 0: global view only
  step t: global view + the t highest-ranked crops (ranked once, by the model's own
          last-token attention over the global view)
and logs, at every step, the answer, confidence features and visual tokens used.
analyze_gonogo.py then fits the sufficiency score and calibrates the stopping rule.

Usage (on the GPU machine):
  python run_gonogo.py --bench vstar    --out logs/vstar_qwen3b.jsonl
  python run_gonogo.py --bench textvqa  --n 800 --out logs/textvqa_qwen3b.jsonl
  python run_gonogo.py --bench hrbench4k --model Qwen/Qwen2.5-VL-7B-Instruct --out logs/hrbench4k_qwen7b.jsonl
The output is appended to and finished question ids are skipped, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import time

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

GLOBAL_PIXELS = 448 * 448          # about 256 visual tokens
CROP_PIXELS = 512 * 28 * 28        # at most 512 visual tokens per crop
GRIDS = (2, 3)                     # candidate cells: 2x2 and 3x3 grids (13 cells)
ATT_LAYER_FRAC = 0.5               # middle LLM layer for the attention map
Image.MAX_IMAGE_PIXELS = None      # V* images are very large


# ---------------------------------------------------------------- data
def load_vstar():
    from huggingface_hub import snapshot_download
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    with open(os.path.join(root, "test_questions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            label = r["label"]
            if isinstance(label, int):
                label = "ABCD"[label]
            q = r["text"].strip()
            if "letter" not in q.lower():
                q += "\nAnswer with the option's letter from the given choices directly."
            yield {
                "qid": f"vstar_{r['question_id']}",
                "image": lambda p=os.path.join(root, r["image"]): Image.open(p).convert("RGB"),
                "question": q,
                "answers": [label.strip().upper()],
                "kind": "mc",
                "category": r.get("category", ""),
            }


def load_textvqa(n, seed=0):
    from datasets import load_dataset
    # streaming: downloads only validation shards, not the large train/test splits
    ds = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000).take(n)
    for r in ds:
        yield {
            "qid": f"textvqa_{r['question_id']}",
            "image": lambda im=r["image"]: im.convert("RGB"),
            "question": r["question"].strip() + "\nAnswer the question using a single word or phrase.",
            "answers": list(r["answers"]),
            "kind": "open",
            "category": "textvqa",
        }


def load_hrbench(version):
    """HR-Bench 4K / 8K (Wang et al., AAAI 2025): 800 multiple-choice questions each, images in base64."""
    import base64, glob, io
    import pandas as pd
    from huggingface_hub import snapshot_download
    # official release (Wang et al., AAAI 2025)
    root = snapshot_download("DreamMr/HR-Bench", repo_type="dataset", allow_patterns=[f"*{version}*.parquet"])
    files = sorted(glob.glob(os.path.join(root, "**", f"*{version}*.parquet"), recursive=True))
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files])
    else:
        df = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(root, "**", f"*{version}*.tsv"), recursive=True)])
    for _, r in df.iterrows():
        img = r["image"]
        raw = img["bytes"] if isinstance(img, dict) else base64.b64decode(img)
        opts = "\n".join(f"{k}. {r[k]}" for k in "ABCD" if k in r and isinstance(r[k], str))
        yield {
            "qid": f"hrbench{version}_{r['index']}",
            "image": lambda b=raw: Image.open(io.BytesIO(b)).convert("RGB"),
            "question": f"{r['question'].strip()}\n{opts}\nAnswer with the option's letter from the given choices directly.",
            "answers": [str(r["answer"]).strip().upper()],
            "kind": "mc",
            "category": r.get("category", ""),
        }


# ---------------------------------------------------------------- answers
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(ans: str, kind: str) -> str:
    ans = ans.strip()
    if kind == "mc":
        m = re.search(r"\b([A-E])\b", ans.upper()) or re.match(r"\(?([A-E])", ans.upper())
        return m.group(1) if m else ans.upper()[:1]
    ans = ans.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(_ARTICLES.sub(" ", ans).split())


def score(pred: str, answers, kind: str) -> float:
    if kind == "mc":
        return float(pred == answers[0])
    gts = [normalize(a, "open") for a in answers]
    return min(1.0, sum(g == pred for g in gts) / 3.0)  # standard VQA accuracy


# ---------------------------------------------------------------- views
def cells():
    out = []
    for g in GRIDS:
        for i in range(g):
            for j in range(g):
                out.append((j / g, i / g, (j + 1) / g, (i + 1) / g))  # x0, y0, x1, y1 (normalised)
    return out


def crop(img, box):
    W, H = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(x0 * W), int(y0 * H), int(math.ceil(x1 * W)), int(math.ceil(y1 * H))))


def build_inputs(processor, views, question):
    content = [{"type": "image", "image": views[0], "max_pixels": GLOBAL_PIXELS}]
    for v in views[1:]:
        content.append({"type": "image", "image": v, "max_pixels": CROP_PIXELS})
    if len(views) > 1:
        lead = ("The first image is the full picture. The following images are zoomed-in crops "
                "of regions of the same picture.\n")
    else:
        lead = ""
    content.append({"type": "text", "text": lead + question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")


def visual_tokens(inputs) -> int:
    return int((inputs["image_grid_thw"].prod(dim=1) // 4).sum())


# ---------------------------------------------------------------- model calls
@torch.no_grad()
def answer(model, processor, inputs, max_new_tokens):
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         output_logits=True, return_dict_in_generate=True)
    gen = out.sequences[0, inputs["input_ids"].shape[1]:]
    logps, margins = [], []
    for k, logits in enumerate(out.logits):
        p = torch.softmax(logits[0].float(), -1)
        top2 = torch.topk(p, 2).values
        margins.append(float(top2[0] - top2[1]))
        logps.append(float(torch.log(p[gen[k]] + 1e-12)))
    text = processor.decode(gen, skip_special_tokens=True)
    n = max(1, len([t for t in gen.tolist() if t not in (processor.tokenizer.eos_token_id,)]))
    return text, {
        "margin0": margins[0],
        "mean_logp": sum(logps[:n]) / n,
        "min_logp": min(logps[:n]),
    }


@torch.no_grad()
def attention_over_cells(model, inputs):
    """Last-token attention at a middle layer, summed over each candidate cell."""
    inputs = inputs.to(model.device)
    out = model(**inputs, output_attentions=True)
    L = int(len(out.attentions) * ATT_LAYER_FRAC)
    att = out.attentions[L][0, :, -1, :].float().mean(0)          # (seq,)
    ids = inputs["input_ids"][0]
    img_pos = (ids == model.config.image_token_id).nonzero().squeeze(-1)
    _, h, w = inputs["image_grid_thw"][0].tolist()
    gh, gw = h // 2, w // 2                                        # 2x2 patch merge
    a = att[img_pos[: gh * gw]].reshape(gh, gw)
    a = a / a.sum()
    masses = []
    for (x0, y0, x1, y1) in cells():
        r0, r1 = int(y0 * gh), max(int(y0 * gh) + 1, int(round(y1 * gh)))
        c0, c1 = int(x0 * gw), max(int(x0 * gw) + 1, int(round(x1 * gw)))
        masses.append(float(a[r0:r1, c0:c1].sum()))
    return masses


@torch.no_grad()
def hidden_vec(model, inputs):
    """Last-token hidden state (middle and last LLM layer) for a learned sufficiency probe, as base64 float16."""
    import base64
    import numpy as np
    inputs = inputs.to(model.device)
    hs = model(**inputs, output_hidden_states=True).hidden_states
    v = torch.cat([hs[len(hs) // 2][0, -1], hs[-1][0, -1]]).float().cpu().numpy().astype(np.float16)
    return base64.b64encode(v.tobytes()).decode()


def rank_cells(masses, T):
    """Greedy order by attention density, skipping cells mostly covered by ones already chosen."""
    cs = cells()
    area = [(x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in cs]
    order = sorted(range(len(cs)), key=lambda k: masses[k] / math.sqrt(area[k]), reverse=True)
    chosen = []
    for k in order:
        x0, y0, x1, y1 = cs[k]
        covered = 0.0
        for c in chosen:
            u0, v0, u1, v1 = cs[c]
            covered += max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
        if covered / area[k] < 0.5:
            chosen.append(k)
        if len(chosen) == T:
            break
    return chosen


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["vstar", "textvqa", "hrbench4k", "hrbench8k", "mmerw"], required=True)
    ap.add_argument("--n", type=int, default=800, help="samples (textvqa); questions per domain (mmerw); vstar/hrbench use all")
    ap.add_argument("--mmerw_repo", default="yifanzhang114/MME-RealWorld-lmms-eval")
    ap.add_argument("--T", type=int, default=4, help="maximum number of crops (full budget)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--global_tokens", type=int, default=256, help="visual-token cap for the global view")
    ap.add_argument("--save_hidden", action="store_true", help="store the step-0 hidden state for probe.py")
    ap.add_argument("--crop_tokens", type=int, default=512, help="visual-token cap per crop")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global GLOBAL_PIXELS, CROP_PIXELS
    GLOBAL_PIXELS, CROP_PIXELS = args.global_tokens * 28 * 28, args.crop_tokens * 28 * 28

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["qid"] for l in f if l.strip()}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager" if args.T > 0 else "sdpa", device_map="cuda")  # eager only for the attention map
    model.eval()
    # eager is needed only for the LLM attention map; the vision tower's eager attention builds a
    # full fp32 (heads x seq x seq) matrix over all crops and runs out of memory on large V* images
    try:  # transformers < 4.48 has a separate SDPA class for the vision tower
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionSdpaAttention
        for blk in model.visual.blocks:
            blk.attn.__class__ = Qwen2_5_VLVisionSdpaAttention
    except ImportError:  # later releases choose the attention function from the config at run time
        model.visual.config._attn_implementation = "sdpa"
    processor = AutoProcessor.from_pretrained(args.model)

    if args.bench == "vstar":
        data = load_vstar()
    elif args.bench == "textvqa":
        data = load_textvqa(args.n)
    elif args.bench == "mmerw":
        from mmerw import load_mmerw
        data = load_mmerw(args.mmerw_repo, per_domain=args.n)
    else:
        data = load_hrbench(args.bench[len("hrbench"):])
    max_new = 12 if args.bench == "textvqa" else 4
    cs = cells()
    with open(args.out, "a") as fout:
        for ex in data:
            if ex["qid"] in done:
                continue
            t0 = time.time()
            img = ex["image"]()
            inp0 = build_inputs(processor, [img], ex["question"])
            # T = 0 is the single-view baseline: no ranking needed (and the attention pass would not fit at high res)
            masses = attention_over_cells(model, inp0) if args.T > 0 else [0.0] * len(cs)
            order = rank_cells(masses, args.T)
            steps = []
            for t in range(args.T + 1):
                views = [img] + [crop(img, cs[k]) for k in order[:t]]
                inp = inp0 if t == 0 else build_inputs(processor, views, ex["question"])
                raw, feats = answer(model, processor, inp, max_new)
                pred = normalize(raw, ex["kind"])
                residual = 1.0 - min(1.0, sum(masses[k] for k in order[:t]))
                steps.append({"t": t, "raw": raw, "pred": pred,
                              "correct": score(pred, ex["answers"], ex["kind"]),
                              "vis_tokens": visual_tokens(inp), "residual_att": residual, **feats})
            rec = {"qid": ex["qid"], "bench": args.bench, "category": ex["category"], "question": ex["question"],
                   "answers": ex["answers"], "order": order, "cell_mass": masses,
                   "steps": steps, "seconds": round(time.time() - t0, 2)}
            if args.save_hidden:
                rec["hidden"] = hidden_vec(model, inp0)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"{ex['qid']}: " + " ".join(s["pred"] for s in steps) + f"  ({rec['seconds']}s)", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

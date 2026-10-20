"""ViCrop baseline (Zhang et al., ICLR 2025, "MLLMs know where to look"), relative-attention variant,
reimplemented for Qwen2.5-VL. ViCrop's released code targets LLaVA-1.5 and InstructBLIP, so the paper must
call this "ViCrop (rel-att), our reimplementation".

Per question:
  1. two forward passes on the global view: one with the question, one with a generic prompt;
  2. rel-att map = attention of the last input token to the image tokens (mean over heads) for the question,
     divided by the same map for the generic prompt, averaged over --att_layers (the middle layer alone
     gives a map that ignores the question; see --dump);
  3. crop the window with the highest attention density (windows of 1/4, 3/8 and 1/2 of each side);
  4. answer once from [global view, crop], always (ViCrop never skips the crop).
Logged cost: visual tokens of the answer pass, and total tokens including the two attention passes.

  python run_vicrop.py --bench mmerw --n 1000 --global_tokens 1024 --crop_tokens 1024 \
      --model Qwen/Qwen2.5-VL-7B-Instruct --out logs/mmerw_7b_vicrop.jsonl
"""
import argparse, json, os, time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_gonogo as rg

GENERIC = "Write a general description of the image."
SIZES = (0.25, 0.375, 0.5)


@torch.no_grad()
def att_map(model, inputs, all_layers=False):
    inputs = inputs.to(model.device)
    out = model(**inputs, output_attentions=True)
    ids = inputs["input_ids"][0]
    pos = (ids == model.config.image_token_id).nonzero().squeeze(-1)
    _, h, w = inputs["image_grid_thw"][0].tolist()
    gh, gw = h // 2, w // 2
    if all_layers:  # (layers, gh, gw), each layer normalised; for --dump diagnostics
        a = torch.stack([x[0, :, -1, :].float().mean(0)[pos[: gh * gw]].reshape(gh, gw) for x in out.attentions])
        del out
        return a / a.sum((1, 2), keepdim=True)
    L = int(len(out.attentions) * rg.ATT_LAYER_FRAC)
    a = out.attentions[L][0, :, -1, :].float().mean(0)[pos[: gh * gw]].reshape(gh, gw)
    del out
    return a / a.sum()


def best_window(rel):
    """Window (normalised x0, y0, x1, y1) with the highest relative-attention density."""
    gh, gw = rel.shape
    ii = torch.nn.functional.pad(rel.cumsum(0).cumsum(1), (1, 0, 1, 0))  # integral image
    best, box = -1.0, (0.0, 0.0, 1.0, 1.0)
    for s in SIZES:
        h, w = max(1, round(s * gh)), max(1, round(s * gw))
        sums = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]       # (gh-h+1, gw-w+1)
        score = sums / (h * w) ** 0.5
        k = int(score.argmax())
        r, c = divmod(k, score.shape[1])
        if float(score.max()) > best:
            best, box = float(score.max()), (c / gw, r / gh, (c + w) / gw, (r + h) / gh)
    return box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["vstar", "hrbench4k", "hrbench8k", "mmerw"], required=True)
    ap.add_argument("--n", type=int, default=1000, help="questions per domain (mmerw)")
    ap.add_argument("--mmerw_repo", default="yifanzhang114/MME-RealWorld-lmms-eval")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--global_tokens", type=int, default=1024)
    ap.add_argument("--crop_tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--att_layers", default="19,20,21,22",
                    help="LLM layers whose relative attention is averaged. Chosen from --dump diagnostics on 30 questions "
                         "(no labels): for Qwen2.5-VL-7B the middle layer (14) gives the same bottom-half map for every "
                         "question, layers 19-22 vary with the question")
    ap.add_argument("--dump", type=int, default=0, help="diagnostics: save all-layer maps of the first N questions to --out (.npz) and stop")
    args = ap.parse_args()
    rg.GLOBAL_PIXELS, rg.CROP_PIXELS = args.global_tokens * 28 * 28, args.crop_tokens * 28 * 28
    layers = [int(x) for x in args.att_layers.split(",")]

    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["qid"] for l in open(args.out) if l.strip()}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda")
    model.eval()
    try:  # transformers < 4.48 has a separate SDPA class for the vision tower
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionSdpaAttention
        for blk in model.visual.blocks:
            blk.attn.__class__ = Qwen2_5_VLVisionSdpaAttention
    except ImportError:  # later releases choose the attention function from the config at run time
        model.visual.config._attn_implementation = "sdpa"
    processor = AutoProcessor.from_pretrained(args.model)

    if args.bench == "mmerw":
        from mmerw import load_mmerw
        data = load_mmerw(args.mmerw_repo, per_domain=args.n)
    elif args.bench == "vstar":
        data = rg.load_vstar()
    else:
        data = rg.load_hrbench(args.bench[len("hrbench"):])

    if args.dump:
        import numpy as np
        maps = {}
        for i, ex in enumerate(data):
            if i == args.dump:
                break
            img = ex["image"]()
            q = att_map(model, rg.build_inputs(processor, [img], ex["question"]), True).cpu().numpy()
            g = att_map(model, rg.build_inputs(processor, [img], GENERIC), True).cpu().numpy()
            maps[f"{ex['qid']}_q"], maps[f"{ex['qid']}_g"] = q.astype(np.float16), g.astype(np.float16)
            print(ex["qid"], q.shape, flush=True)
            torch.cuda.empty_cache()
        np.savez_compressed(args.out, **maps)
        print("wrote", args.out)
        return

    with open(args.out, "a") as fout:
        for ex in data:
            if ex["qid"] in done:
                continue
            t0 = time.time()
            img = ex["image"]()
            q_in = rg.build_inputs(processor, [img], ex["question"])
            g_in = rg.build_inputs(processor, [img], GENERIC)
            Q, G = att_map(model, q_in, True), att_map(model, g_in, True)
            rel = torch.stack([(Q[l] / (G[l] + 1e-6)) / (Q[l] / (G[l] + 1e-6)).sum() for l in layers]).mean(0)
            box = best_window(rel)
            inp = rg.build_inputs(processor, [img, rg.crop(img, box)], ex["question"])
            raw, feats = rg.answer(model, processor, inp, 4)
            pred = rg.normalize(raw, ex["kind"])
            vt = rg.visual_tokens(inp)
            rec = {"qid": ex["qid"], "bench": args.bench, "category": ex["category"], "answers": ex["answers"],
                   "box": box, "raw": raw, "pred": pred, "correct": rg.score(pred, ex["answers"], ex["kind"]),
                   "vis_tokens": vt, "vis_tokens_total": vt + 2 * rg.visual_tokens(q_in), **feats,
                   "seconds": round(time.time() - t0, 2)}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"{ex['qid']}: {pred}  box {tuple(round(b, 2) for b in box)}  ({rec['seconds']}s)", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

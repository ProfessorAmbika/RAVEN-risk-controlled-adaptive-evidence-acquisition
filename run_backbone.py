"""Single-view runs for other backbones (and timed runs for any backbone), in the log format of run_gonogo.py,
so make_ladder.py and probe.py work unchanged.

One run = one view level for every question: answer, confidence features, visual tokens, the last-prompt-token
hidden state (middle + last LLM layer, taken from the same generate() call, so the probe adds no forward pass),
and CUDA-synchronised timings split into preprocessing and model time.

Backbones (--family):
  internvl  OpenGVLab/InternVL3-8B-hf        view level = max number of 448-px tiles (256 tokens each, plus a
                                             thumbnail tile when the image is split): --tiles 4 (~1.3k tokens),
                                             --tiles 16 (~4.3k tokens)
  llavaov   llava-hf/llava-onevision-qwen2-7b-ov-hf   view level = longest image side before AnyRes tiling:
                                             --long_side 384 (one tile + base), --long_side 1536 (up to 4x4 tiles)
  qwen      Qwen/Qwen2.5-VL-7B-Instruct      view level = --tokens cap (same views as the main experiments);
                                             use this for the latency measurement

  python run_backbone.py --family internvl --tiles 4  --bench mmerw --n 1000 --out logs/mmerw_internvl8b_t4.jsonl
  python run_backbone.py --family internvl --tiles 16 --bench mmerw --n 1000 --out logs/mmerw_internvl8b_t16.jsonl
  python make_ladder.py logs/mmerw_internvl8b_t4.jsonl logs/mmerw_internvl8b_t16.jsonl --out logs/mmerw_internvl8b_ladder.jsonl
  python probe.py --train logs/mmerw_internvl8b_ladder.jsonl --by_domain
Timing (Qwen, 100 questions per domain, both views in one process so the GPU state is the same):
  python run_backbone.py --family qwen --tokens 1024 --bench mmerw --n 100 --out logs/time_qwen_1024.jsonl
  python run_backbone.py --family qwen --tokens 4096 --bench mmerw --n 100 --out logs/time_qwen_4096.jsonl
Resumable: finished question ids in --out are skipped.
"""
import argparse, base64, json, os, time

import numpy as np
import torch

import run_gonogo as rg

MODELS = {"internvl": "OpenGVLab/InternVL3-8B-hf", "llavaov": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
          "qwen": "Qwen/Qwen2.5-VL-7B-Instruct"}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(family, name):
    import transformers
    from transformers import AutoProcessor
    # transformers 5 renamed torch_dtype to dtype
    dt = {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": torch.bfloat16}
    if family == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, attn_implementation="sdpa", device_map="cuda", **dt)
    else:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            name, attn_implementation="sdpa", device_map="cuda", **dt)
    model.eval()
    return model, AutoProcessor.from_pretrained(name)


def image_token_id(model):
    for k in ("image_token_id", "image_token_index"):
        v = getattr(model.config, k, None)
        if v is not None:
            return v
    raise AttributeError("no image token id in model.config")


def build(family, processor, img, question, a):
    if family == "qwen":
        rg.GLOBAL_PIXELS = a.tokens * 28 * 28
        return rg.build_inputs(processor, [img], question)
    if family == "llavaov":
        s = a.long_side / max(img.size)
        if s < 1:
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    kw = {}
    if family == "internvl":
        kw = {"crop_to_patches": a.tiles > 1, "max_patches": a.tiles}
    return processor(images=[img], text=text, return_tensors="pt", **kw)


@torch.no_grad()
def answer(model, processor, inputs, max_new):
    inputs = inputs.to(model.device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, output_logits=True,
                         output_hidden_states=True, return_dict_in_generate=True)
    gen = out.sequences[0, inputs["input_ids"].shape[1]:]
    logps, margins = [], []
    for k, logits in enumerate(out.logits):
        p = torch.softmax(logits[0].float(), -1)
        top2 = torch.topk(p, 2).values
        margins.append(float(top2[0] - top2[1]))
        logps.append(float(torch.log(p[gen[k]] + 1e-12)))
    hs = out.hidden_states[0]                                   # prompt pass: tuple over layers of (1, seq, d)
    hid = torch.cat([hs[len(hs) // 2][0, -1], hs[-1][0, -1]]).float().cpu().numpy().astype(np.float16)
    tok = getattr(processor, "tokenizer", processor)
    text = tok.decode(gen, skip_special_tokens=True)
    n = max(1, len([t for t in gen.tolist() if t != tok.eos_token_id]))
    return text, {"margin0": margins[0], "mean_logp": sum(logps[:n]) / n, "min_logp": min(logps[:n])}, \
        base64.b64encode(hid.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(MODELS), required=True)
    ap.add_argument("--model", help="override the default checkpoint of --family")
    ap.add_argument("--bench", choices=["vstar", "hrbench4k", "hrbench8k", "mmerw"], required=True)
    ap.add_argument("--n", type=int, default=1000, help="questions per domain (mmerw)")
    ap.add_argument("--mmerw_repo", default="yifanzhang114/MME-RealWorld-lmms-eval")
    ap.add_argument("--tiles", type=int, default=4, help="internvl: max tiles")
    ap.add_argument("--long_side", type=int, default=384, help="llavaov: longest image side")
    ap.add_argument("--tokens", type=int, default=1024, help="qwen: visual-token cap")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warm-up passes on the first question")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):  # a run killed mid-write leaves a truncated last line: drop it and resume
        good = []
        for l in open(a.out):
            try:
                done.add(json.loads(l)["qid"]); good.append(l if l.endswith("\n") else l + "\n")
            except (json.JSONDecodeError, KeyError):
                print("dropping a truncated log line", flush=True)
        open(a.out, "w").writelines(good)
    model, processor = load_model(a.family, a.model or MODELS[a.family])
    img_id = None if a.family == "qwen" else image_token_id(model)
    if a.bench == "mmerw":
        from mmerw import load_mmerw
        data = load_mmerw(a.mmerw_repo, per_domain=a.n)
    elif a.bench == "vstar":
        data = rg.load_vstar()
    else:
        data = rg.load_hrbench(a.bench[len("hrbench"):])
    warm = a.warmup
    with open(a.out, "a") as fout:
        for ex in data:
            if ex["qid"] in done:
                continue
            if warm > 0:  # warm the GPU on the first question; it is then run again and logged
                for _ in range(warm):
                    answer(model, processor, build(a.family, processor, ex["image"](), ex["question"], a), 4)
                warm = 0
            sync(); t0 = time.perf_counter()
            img = ex["image"]()
            inp = build(a.family, processor, img, ex["question"], a)
            sync(); t1 = time.perf_counter()
            raw, feats, hid = answer(model, processor, inp, 4)
            sync(); t2 = time.perf_counter()
            pred = rg.normalize(raw, ex["kind"])
            vt = rg.visual_tokens(inp) if a.family == "qwen" else int((inp["input_ids"] == img_id).sum())
            step = {"t": 0, "raw": raw, "pred": pred, "correct": rg.score(pred, ex["answers"], ex["kind"]),
                    "vis_tokens": vt, "residual_att": 1.0, **feats}
            rec = {"qid": ex["qid"], "bench": a.bench, "category": ex["category"], "answers": ex["answers"],
                   "steps": [step], "hidden": hid, "seconds": round(t2 - t0, 3),
                   "sec_pre": round(t1 - t0, 3), "sec_model": round(t2 - t1, 3)}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"{ex['qid']}: {pred}  {vt} tok  pre {rec['sec_pre']}s  model {rec['sec_model']}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

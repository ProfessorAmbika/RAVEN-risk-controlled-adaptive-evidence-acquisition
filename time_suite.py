"""Measured efficiency suite for Qwen2.5-VL-7B (repeated timing, crop timing, batching).

All parts load the model once (SDPA attention, bf16) and use the timed questions of run_backbone.py
(mmerw.py, --n questions per domain). CUDA-synchronised perf_counter timings; peak GPU memory per pass.

  repeat  --reps full passes over the questions, one after the other, each timing per question: image decoding,
          then the 1,024-token view (preprocessing, model) and the 4,096-token view (preprocessing, model) on the
          same decoded image. Gives run-to-run variability and the per-question RAVEN latency components.
  crop    --crop_n questions per domain: the crop rung as deployed. Rung 0 must return attention maps, so the
          text model switches to eager attention for it (the vision tower stays SDPA):
            rung0_sdpa    1,024 view answer with SDPA (what the resolution ladder pays)
            rung0_eager   the same answer with eager attention + attention maps (what the crop ladder pays)
            generic       the generic-prompt attention pass (paid only on escalation)
            crop_answer   box selection + answer from [view, crop] (paid only on escalation)
          with peak memory of each pass.
  batch   --batch_n questions per domain, each view at each --batch size: time per batch split into image decoding,
          preprocessing and model time, peak memory, out-of-memory recorded; predictions are kept so batched
          answers can be compared with batch size 1 (left padding).

  python time_suite.py --parts repeat crop batch --out logs/timesuite_qwen
Writes <out>_repeat.jsonl, <out>_crop.jsonl, <out>_batch.jsonl; resumable (finished entries are skipped).
"""
import argparse, json, os, time

import torch
from qwen_vl_utils import process_vision_info

import run_gonogo as rg
from run_backbone import load_model, answer as answer_hidden, sync
from run_vicrop import GENERIC, best_window

pc = time.perf_counter
LAYERS = (19, 20, 21, 22)  # run_vicrop.py --att_layers


def gib():
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)


def view(processor, img, question, tokens):
    rg.GLOBAL_PIXELS = tokens * 28 * 28
    return rg.build_inputs(processor, [img], question)


def done_keys(path, key):
    if not os.path.exists(path):
        return set()
    good, keys = [], set()
    for l in open(path):
        try:
            r = json.loads(l); keys.add(key(r)); good.append(l if l.endswith("\n") else l + "\n")
        except (json.JSONDecodeError, KeyError):
            print("dropping a truncated log line", flush=True)
    open(path, "w").writelines(good)
    return keys


def text_attention(model, impl):
    """Switch the LLM's attention implementation at run time; the vision tower keeps SDPA."""
    cfgs = {}
    for m in model.modules():
        n = type(m).__name__
        if n.endswith("Attention") and "Vision" not in n and hasattr(m, "config"):
            cfgs[id(m.config)] = m.config
    tc = getattr(model.config, "text_config", None)
    if tc is not None:
        cfgs[id(tc)] = tc
    for c in cfgs.values():
        c._attn_implementation = impl


def rel_map(att_q, att_g, inputs, model):
    ids = inputs["input_ids"][0]
    pos = (ids == model.config.image_token_id).nonzero().squeeze(-1)
    _, h, w = inputs["image_grid_thw"][0].tolist()
    gh, gw = h // 2, w // 2
    maps = []
    for l in LAYERS:
        q = att_q[l][0, :, -1, :].float().mean(0)[pos[: gh * gw]].reshape(gh, gw)
        g = att_g[l][0, :, -1, :].float().mean(0)[pos[: gh * gw]].reshape(gh, gw)
        r = (q / q.sum()) / (g / g.sum() + 1e-6)
        maps.append(r / r.sum())
    return torch.stack(maps).mean(0)


def part_repeat(model, processor, data, a):
    path = a.out + "_repeat.jsonl"
    done = done_keys(path, lambda r: (r["rep"], r["qid"]))
    with open(path, "a") as f:
        for rep in range(a.reps):
            for ex in data:
                if (rep, ex["qid"]) in done:
                    continue
                sync(); t0 = pc(); img = ex["image"](); sync(); t1 = pc()
                rec = {"rep": rep, "qid": ex["qid"], "category": ex["category"], "answers": ex["answers"],
                       "sec_decode": round(t1 - t0, 4)}
                for T in (1024, 4096):
                    sync(); s0 = pc(); inp = view(processor, img, ex["question"], T); sync(); s1 = pc()
                    torch.cuda.reset_peak_memory_stats()
                    raw, _, _ = answer_hidden(model, processor, inp, 4); sync(); s2 = pc()
                    rec |= {f"pre{T}": round(s1 - s0, 4), f"model{T}": round(s2 - s1, 4), f"mem{T}": gib(),
                            f"tok{T}": rg.visual_tokens(inp), f"pred{T}": rg.normalize(raw, "mc")}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"repeat {rep} {ex['qid']}: decode {rec['sec_decode']:.2f}  1024 {rec['model1024']:.2f}s  "
                      f"4096 {rec['model4096']:.2f}s", flush=True)
                torch.cuda.empty_cache()


def part_crop(model, processor, data, a):
    path = a.out + "_crop.jsonl"
    done = done_keys(path, lambda r: r["qid"])
    rg.CROP_PIXELS = 1024 * 28 * 28
    with open(path, "a") as f:
        for ex in data:
            if ex["qid"] in done:
                continue
            img = ex["image"]()
            rec = {"qid": ex["qid"], "category": ex["category"]}
            # rung 0 as the resolution ladder runs it
            sync(); s0 = pc(); inp = view(processor, img, ex["question"], 1024)
            torch.cuda.reset_peak_memory_stats()
            answer_hidden(model, processor, inp, 4); sync()
            rec |= {"rung0_sdpa": round(pc() - s0, 4), "mem_rung0_sdpa": gib()}
            # rung 0 as the crop ladder runs it: same answer, plus attention maps of the prompt pass
            text_attention(model, "eager")
            try:
                sync(); s0 = pc(); inp = view(processor, img, ex["question"], 1024).to(model.device)
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=4, do_sample=False, output_logits=True,
                                         output_hidden_states=True, output_attentions=True,
                                         return_dict_in_generate=True)
                att_q = [x.detach() for x in out.attentions[0]] if out.attentions else None
                if att_q is None or att_q[0] is None:
                    raise RuntimeError("no attention maps returned with eager attention; report this output")
                att_q = {l: att_q[l] for l in LAYERS}
                del out
                sync(); rec |= {"rung0_eager": round(pc() - s0, 4), "mem_rung0_eager": gib()}
                # escalation: generic-prompt attention pass, box, answer from [view, crop]
                sync(); s0 = pc(); g_in = view(processor, img, GENERIC, 1024).to(model.device)
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    go = model(**g_in, output_attentions=True)
                att_g = {l: go.attentions[l].detach() for l in LAYERS}
                del go
                sync(); rec |= {"generic": round(pc() - s0, 4), "mem_generic": gib()}
            finally:
                text_attention(model, "sdpa")
            sync(); s0 = pc()
            box = best_window(rel_map(att_q, att_g, inp, model))
            del att_q, att_g
            c_in = rg.build_inputs(processor, [img, rg.crop(img, box)], ex["question"])
            torch.cuda.reset_peak_memory_stats()
            raw, _, _ = answer_hidden(model, processor, c_in, 4); sync()
            rec |= {"crop_answer": round(pc() - s0, 4), "mem_crop_answer": gib(), "box": box,
                    "tok_crop": rg.visual_tokens(c_in), "pred_crop": rg.normalize(raw, "mc")}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"crop {ex['qid']}: rung0 sdpa {rec['rung0_sdpa']:.2f}s eager {rec['rung0_eager']:.2f}s "
                  f"({rec['mem_rung0_sdpa']:.1f} -> {rec['mem_rung0_eager']:.1f} GiB)  generic {rec['generic']:.2f}s  "
                  f"crop answer {rec['crop_answer']:.2f}s", flush=True)
            torch.cuda.empty_cache()


def batch_inputs(processor, imgs, questions, tokens):
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im, "max_pixels": tokens * 28 * 28},
                                          {"type": "text", "text": q}]}] for im, q in zip(imgs, questions)]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    ims, _ = process_vision_info(msgs)
    return processor(text=texts, images=ims, padding=True, return_tensors="pt")


def part_batch(model, processor, data, a):
    path = a.out + "_batch.jsonl"
    done = done_keys(path, lambda r: (r["tokens"], r["batch"], r["first"]))
    tok = processor.tokenizer
    tok.padding_side = "left"
    with open(path, "a") as f:
        for T in (1024, 4096):
            for b in a.batch:
                oom = False
                for i in range(0, len(data) - b + 1, b):
                    if (T, b, data[i]["qid"]) in done:
                        continue
                    chunk = data[i:i + b]
                    sync(); s0 = pc(); imgs = [ex["image"]() for ex in chunk]; sync(); s1 = pc()
                    try:
                        inp = batch_inputs(processor, imgs, [ex["question"] for ex in chunk], T).to(model.device)
                        inp.pop("mm_token_type_ids", None)  # transformers 5 processors emit it; Qwen2.5-VL does not take it
                        if inp["pixel_values"].dtype == torch.float32:
                            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
                        sync(); s2 = pc()
                        torch.cuda.reset_peak_memory_stats()
                        with torch.no_grad():
                            out = model.generate(**inp, max_new_tokens=4, do_sample=False, output_logits=True,
                                                 output_hidden_states=(T == 1024), return_dict_in_generate=True)
                        sync(); s3 = pc()
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                        torch.cuda.empty_cache()
                        f.write(json.dumps({"tokens": T, "batch": b, "first": data[i]["qid"], "oom": True}) + "\n")
                        print(f"batch {T} x{b}: out of memory", flush=True)
                        break
                    gen = out.sequences[:, inp["input_ids"].shape[1]:]
                    preds = [rg.normalize(tok.decode(g, skip_special_tokens=True), "mc") for g in gen]
                    rec = {"tokens": T, "batch": b, "first": chunk[0]["qid"], "qids": [ex["qid"] for ex in chunk],
                           "categories": [ex["category"] for ex in chunk], "preds": preds,
                           "sec_decode": round(s1 - s0, 4), "sec_pre": round(s2 - s1, 4),
                           "sec_model": round(s3 - s2, 4), "mem": gib(), "oom": False}
                    del out, inp
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    print(f"batch {T} x{b} {chunk[0]['qid']}: {(s3 - s0) / b:.3f} s/question  mem {rec['mem']:.1f} GiB",
                          flush=True)
                    torch.cuda.empty_cache()
                if oom:
                    continue
    tok.padding_side = "right"


def first_k(data, k):
    seen, out = {}, []
    for ex in data:
        d = ex["category"].split("/")[0]
        seen[d] = seen.get(d, 0) + 1
        if seen[d] <= k:
            out.append(ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", choices=["repeat", "crop", "batch"], default=["repeat", "crop", "batch"])
    ap.add_argument("--n", type=int, default=100, help="questions per domain (the timed questions of run_backbone.py)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--crop_n", type=int, default=20, help="questions per domain for the crop part")
    ap.add_argument("--batch_n", type=int, default=40, help="questions per domain for the batch part")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mmerw_repo", default="yifanzhang114/MME-RealWorld-lmms-eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    model, processor = load_model("qwen", "Qwen/Qwen2.5-VL-7B-Instruct")
    from mmerw import load_mmerw
    data = list(load_mmerw(a.mmerw_repo, per_domain=a.n))
    ex = data[0]
    for _ in range(a.warmup):  # untimed warm-up
        answer_hidden(model, processor, view(processor, ex["image"](), ex["question"], 4096), 4)
    if "repeat" in a.parts:
        part_repeat(model, processor, data, a)
    if "crop" in a.parts:
        part_crop(model, processor, first_k(data, a.crop_n), a)
    if "batch" in a.parts:
        part_batch(model, processor, first_k(data, a.batch_n), a)
    print("time_suite finished", flush=True)


if __name__ == "__main__":
    main()

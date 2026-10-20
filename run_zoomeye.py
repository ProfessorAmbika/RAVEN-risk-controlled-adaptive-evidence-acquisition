"""Zoom Eye (Shen et al., EMNLP 2025; github.com/om-ai-lab/ZoomEye) as a visual-acquisition baseline, run with the
authors' own Qwen2.5-VL code on exactly our MME-RealWorld questions, with the same cost accounting as RAVEN.

What is ours and what is theirs:
  theirs  ZoomEye/zoom_model_qwenvl.py (prompts, confidences), ZoomEye/zoom_eye.py (tree search, stopping, final
          answer), ZoomEye/ic_examples/mme-realworld.json, and the Qwen2.5-VL search settings of their
          ZoomEye/eval/perform_zoom_eye.py (pop limit 3*depth, thresholds 0.95 / 0, decrease [0.05, 0.1, 0.2],
          visual cue threshold 0.6, 4-way split)
  ours    question selection (mmerw.py, the same 5,000 questions), answer scoring (run_gonogo.py), and the meters:
          visual tokens summed over every forward pass that sees an image (each Zoom Eye step is a fresh pass),
          number of such passes, and CUDA-synchronised seconds per question
  changed SDPA attention instead of flash-attention-2 (not installed here; same outputs up to numerics), and
          max_pixels = --tokens * 28 * 28 for every image Zoom Eye builds (theirs: 12.8 MP, up to ~16k tokens per
          image, which does not fit a 24 GB RTX 3090 with a 7B model); --tokens 4096 matches the fixed high view

Setup (once):  git clone https://github.com/om-ai-lab/ZoomEye.git
Run:
  python run_zoomeye.py --zoomeye path/to/ZoomEye --per_domain 100 --out logs/mmerw_zoomeye_q7b.jsonl
--per_domain k takes the first k of each domain's 1,000 questions (the same questions as the other logs), so it can
be raised later and the run resumes where it stopped.
"""
import argparse, functools, json, os, sys, tempfile, time, types

import torch

import run_gonogo as rg


def stub_llava_modules():
    """zoom_eye.py imports the LLaVA and InternVL wrappers, which need the LLaVA-NeXT package; Qwen does not."""
    zm = types.ModuleType("ZoomEye.zoom_model")
    class ZoomModel: pass
    class ZoomModelLocal(ZoomModel): pass
    class ZoomModelGlobalLocal(ZoomModel): pass
    zm.ZoomModel, zm.ZoomModelLocal, zm.ZoomModelGlobalLocal, zm.BOX_COLOR = ZoomModel, ZoomModelLocal, ZoomModelGlobalLocal, "red"
    zi = types.ModuleType("ZoomEye.zoom_model_internvl")
    zi.BOX_COLOR = "red"
    sys.modules["ZoomEye.zoom_model"], sys.modules["ZoomEye.zoom_model_internvl"] = zm, zi


def force_sdpa():
    from transformers import Qwen2_5_VLForConditionalGeneration as Q
    orig = Q.from_pretrained.__func__
    def fp(cls, *args, **kw):
        kw["attn_implementation"] = "sdpa"
        if "torch_dtype" in kw:  # transformers 5 renamed it
            kw["dtype"] = kw.pop("torch_dtype")
        return orig(cls, *args, **kw)
    Q.from_pretrained = classmethod(fp)


class Meter:
    """Counts image tokens and image-bearing forward passes of every model call (prefill only: decode steps carry
    no image tokens)."""
    def __init__(self, model):
        self.img_id = model.config.image_token_id
        self.reset()
        fwd = model.forward
        @functools.wraps(fwd)  # keep forward's signature: generate() validates its kwargs against it
        def counted(*args, **kw):
            kw.pop("mm_token_type_ids", None)  # transformers 5 processors emit it; Qwen2.5-VL does not take it
            ids = kw.get("input_ids", args[0] if args else None)
            if ids is not None:
                k = int((ids == self.img_id).sum())
                if k:
                    self.tokens += k; self.passes += 1
            return fwd(*args, **kw)
        model.forward = counted
        gen = model.generate
        @functools.wraps(gen)
        def generate(*args, **kw):
            kw.pop("mm_token_type_ids", None)
            return gen(*args, **kw)
        model.generate = generate

    def reset(self):
        self.tokens, self.passes = 0, 0


def split_question(q):
    """mmerw.py's prompt -> (question, options) in the form Zoom Eye's MME-RealWorld path expects."""
    head, rest = q.split("\nThe choices are listed below:\n", 1)
    opts = rest.split("\nSelect the best answer")[0].strip().split("\n")
    return head.strip(), [o.strip() for o in opts if o.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoomeye", required=True, help="path of the cloned ZoomEye repository")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--tokens", type=int, default=4096, help="max visual tokens per image Zoom Eye builds")
    ap.add_argument("--per_domain", type=int, default=200)
    ap.add_argument("--mmerw_repo", default="yifanzhang114/MME-RealWorld-lmms-eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(os.path.expanduser(a.zoomeye)))
    stub_llava_modules()
    force_sdpa()
    from ZoomEye.zoom_model_qwenvl import ZoomModelQwenVL
    from ZoomEye.zoom_eye import get_zoom_eye_response
    from mmerw import load_mmerw

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):  # resume; drop a truncated last line
        good = []
        for l in open(a.out):
            try:
                done.add(json.loads(l)["qid"]); good.append(l if l.endswith("\n") else l + "\n")
            except (json.JSONDecodeError, KeyError):
                print("dropping a truncated log line", flush=True)
        open(a.out, "w").writelines(good)

    zm = ZoomModelQwenVL(model_path=a.model, device="cuda:0", torch_dtype=torch.bfloat16, patch_scale=1.2,
                         max_pixels=a.tokens * 28 * 28)
    ip = zm.processor.image_processor  # transformers 5 reads size["longest_edge"], older versions max_pixels
    ip.max_pixels, ip.min_pixels = a.tokens * 28 * 28, 3136
    if isinstance(getattr(ip, "size", None), dict) or hasattr(getattr(ip, "size", None), "longest_edge"):
        ip.size = {"shortest_edge": 3136, "longest_edge": a.tokens * 28 * 28}
    meter = Meter(zm.model)
    search_kwargs = dict(pop_limit=lambda max_depth: max_depth * 3, threshold_descrease=[0.05, 0.1, 0.2],
                         answering_confidence_threshold_lower=0, answering_confidence_threshold_upper=0.95,
                         visual_cue_threshold=0.6)
    ic = json.load(open(os.path.join(os.path.expanduser(a.zoomeye), "ZoomEye/ic_examples/mme-realworld.json")))
    tmp = os.path.join(tempfile.gettempdir(), "zoomeye_input.png")
    seen, warm = {}, True
    with open(a.out, "a") as fout:
        for ex in load_mmerw(a.mmerw_repo, per_domain=1000):
            dom = ex["category"].split("/")[0]
            seen[dom] = seen.get(dom, 0) + 1
            if seen[dom] > a.per_domain or ex["qid"] in done:
                continue
            question, options = split_question(ex["question"])
            ann = {"input_image": tmp, "question": question, "options": options, "answer_type": "Multiple Choice"}
            ex["image"]().convert("RGB").save(tmp)  # Zoom Eye reads the image from a path
            if warm:  # one untimed pass so CUDA start-up is not billed to the first question
                get_zoom_eye_response(zoom_model=zm, annotation=dict(ann), ic_examples=ic,
                                      decomposed_question_template="What is the appearance of the {}?",
                                      split_num=4, **search_kwargs)
                warm = False
            meter.reset()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = get_zoom_eye_response(zoom_model=zm, annotation=ann, ic_examples=ic,
                                        decomposed_question_template="What is the appearance of the {}?",
                                        split_num=4, **search_kwargs)
            torch.cuda.synchronize(); sec = time.perf_counter() - t0
            raw = out if isinstance(out, str) else out.get("text", str(out))
            pred = rg.normalize(raw, ex["kind"])
            rec = {"qid": ex["qid"], "bench": "mmerw", "category": ex["category"], "answers": ex["answers"],
                   "raw": raw, "pred": pred, "correct": rg.score(pred, ex["answers"], ex["kind"]),
                   "vis_tokens_total": meter.tokens, "passes": meter.passes, "seconds": round(sec, 3),
                   "targets": ann.get("targets"), "num_pop": ann.get("num_pop"),
                   "searched_bbox": [list(map(float, b)) for b in ann.get("searched_bbox", [])]}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"{ex['qid']}: {pred}  {meter.tokens} tok in {meter.passes} passes  {sec:.2f}s", flush=True)
            if meter.passes and meter.tokens / meter.passes > 1.1 * a.tokens + 300:
                print("WARNING: more visual tokens per pass than --tokens allows; the pixel cap is not applied", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

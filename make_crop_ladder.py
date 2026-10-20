"""Build the crop ladder: rung 0 = the 1,024-token view (any two-rung log of run_gonogo.py, its step 0 and hidden
state are kept), rung 1 = the ViCrop answer of run_vicrop.py. Cost of rung 1 = ViCrop's total visual tokens
(two attention passes + the answer pass): the rung-0 pass doubles as ViCrop's question-attention pass, so the
cumulative cost after escalation equals ViCrop's own total.
Usage: python make_crop_ladder.py logs/mmerw_7b_l1024.jsonl logs/mmerw_7b_vicrop2.jsonl --out logs/mmerw_7b_lvicrop_reuse.jsonl"""
import argparse, ast, json
ap = argparse.ArgumentParser(); ap.add_argument("rung0"); ap.add_argument("vicrop"); ap.add_argument("--out", required=True)
a = ap.parse_args()
V = {r["qid"]: r for r in map(json.loads, open(a.vicrop))}
with open(a.out, "w") as f:
    for l in open(a.rung0):
        r = json.loads(l)
        if r["qid"] not in V:
            continue
        steps = r["steps"] if isinstance(r["steps"], list) else ast.literal_eval(r["steps"])
        v = V[r["qid"]]
        s1 = {"t": 1, "raw": v["raw"], "pred": v["pred"], "correct": v["correct"], "vis_tokens": v["vis_tokens_total"],
              "residual_att": 0.0, "margin0": v["margin0"], "mean_logp": v["mean_logp"], "min_logp": v["min_logp"]}
        r["steps"] = [steps[0], s1]
        f.write(json.dumps(r) + "\n")
print("wrote", a.out)

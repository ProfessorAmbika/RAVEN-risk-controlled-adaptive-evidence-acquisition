"""Build a ladder trajectory: all steps of the first run (a single view, or a view plus crops),
then the single-view answer of each later run (higher global token caps).
Step t = answer at the t-th resolution; cost = cumulative visual tokens (each level is a new pass).
Usage: python make_ladder.py logs/vstar_7b_global256.jsonl logs/vstar_7b_global1024.jsonl logs/vstar_7b_global4096.jsonl --out logs/vstar_7b_ladder.jsonl"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("logs", nargs="+"); ap.add_argument("--out", required=True)
a = ap.parse_args()
runs = [{r["qid"]: r for r in map(json.loads, open(p))} for p in a.logs]
with open(a.out, "w") as f:
    for qid in runs[0]:
        if not all(qid in r for r in runs):
            continue
        steps, cum = [], 0
        # all steps of the first run (e.g. a 1,024-token view + one crop), then step 0 of each later run
        seq = list(runs[0][qid]["steps"]) + [r[qid]["steps"][0] for r in runs[1:]]
        for t, s in enumerate(seq):
            s = dict(s)
            cum += s["vis_tokens"]  # every step is a fresh forward pass over the views it shows
            s.update(t=t, vis_tokens=cum, residual_att=1 - t / (len(seq) - 1))
            steps.append(s)
        rec = dict(runs[0][qid]); rec["steps"] = steps
        f.write(json.dumps(rec) + "\n")
print("wrote", a.out)

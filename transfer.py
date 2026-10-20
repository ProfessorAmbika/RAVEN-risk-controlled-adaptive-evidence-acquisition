"""Fit the sufficiency score on one set of logs, then calibrate and test the stopping rule on another.
Example: score trained on HR-Bench, guarantee calibrated and tested on V*.
Usage: python transfer.py --train logs/hrbench4k_qwen7b.jsonl logs/hrbench8k_qwen7b.jsonl --eval logs/vstar_qwen7b.jsonl"""
import argparse, numpy as np
from analyze_gonogo import load, tensors, fit_logreg, evaluate
from ltt_calibrate import calibrate

ap = argparse.ArgumentParser()
ap.add_argument("--train", nargs="+", required=True)
ap.add_argument("--eval", nargs="+", required=True)
ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.10])
args = ap.parse_args()

Xa, fa, ca, _ = tensors(load(args.train))
Xb, fb, cb, kb = tensors(load(args.eval))
n, T1, d = Xb.shape
print(f"train {len(Xa)}  eval {n}   eval acc global {cb[:,0].mean():.3f} full {cb[:,-1].mean():.3f} oracle {cb.max(1).mean():.3f}")
rng = np.random.default_rng(0)
for lname in ("fidelity", "acc-gap"):
    La = fa if lname == "fidelity" else np.clip(ca[:, -1:] - ca, 0, None)
    Lb = fb if lname == "fidelity" else np.clip(cb[:, -1:] - cb, 0, None)
    S = fit_logreg(Xa.reshape(-1, d), (La == 0).reshape(-1))(Xb.reshape(-1, d)).reshape(n, T1)
    for eps in args.eps:
        res, viol = [], 0
        for _ in range(100):
            p = rng.permutation(n); cal, te = p[:n // 2], p[n // 2:]
            c = calibrate(S[cal], Lb[cal], kb[cal], eps=eps, delta=0.1)
            r = evaluate(S[te], Lb[te], cb[te], kb[te], c.lam); res.append(r); viol += r["risk"] > eps
        m = {k: np.mean([r[k] for r in res]) for k in res[0]}
        print(f"{lname:8s} eps={eps:.2f}: risk {m['risk']:.3f} viol {viol}/100  acc {m['acc']:.3f}  "
              f"crops saved {m['crops_saved']:.1%}  tokens saved {m['tokens_saved']:.1%}  harmful-zoom {m['harmful_zoom']:.3f}")

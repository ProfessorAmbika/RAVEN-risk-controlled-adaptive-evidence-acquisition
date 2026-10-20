"""Per-benchmark RAVEN: score fit and calibration on each benchmark's own data.
Compares two losses: fidelity 1[pred_t != pred_T] (label-free) and accuracy gap max(0, acc_T - acc_t).
Usage: python per_bench2.py logs/vstar_qwen3b.jsonl logs/textvqa_qwen3b.jsonl"""
import sys, numpy as np
from analyze_gonogo import load, tensors, fit_logreg, evaluate
from ltt_calibrate import calibrate
recs = load(sys.argv[1:])
X, fid, correct, cost = tensors(recs)
gap = np.clip(correct[:, -1:] - correct, 0, None)
bench = np.array([r["bench"] for r in recs])
rng = np.random.default_rng(0)
for b in sorted(set(bench)):
    ib = np.where(bench == b)[0]
    n, T1, d = len(ib), X.shape[1], X.shape[2]
    # how much early stopping is possible at all: first step whose answer equals the final one and never changes after
    same = (fid[ib] == 0)
    first_stable = np.array([0 if row.all() else T1 - np.argmin(row[::-1]) for row in same])
    print(f"\n{b}: n={n}  acc global {correct[ib,0].mean():.3f} full {correct[ib,-1].mean():.3f} oracle {correct[ib].max(1).mean():.3f}  "
          f"max crops saveable (fidelity oracle) {1 - first_stable.mean()/(T1-1):.1%}")
    for lname, L in (("fidelity", fid), ("acc-gap", gap)):
        res, viol = [], 0
        for _ in range(100):
            p = rng.permutation(ib)
            tr, cal, te = np.split(p, [int(0.4*n), int(0.7*n)])
            g = fit_logreg(X[tr].reshape(-1, d), (L[tr] == 0).reshape(-1))
            S = g(X[ib].reshape(-1, d)).reshape(n, T1); pos = {k: j for j, k in enumerate(ib)}
            Sc, St = S[[pos[k] for k in cal]], S[[pos[k] for k in te]]
            c = calibrate(Sc, L[cal], cost[cal], eps=0.05, delta=0.1)
            r = evaluate(St, L[te], correct[te], cost[te], c.lam); res.append(r); viol += r["risk"] > 0.05
        m = {k: np.mean([r[k] for r in res]) for k in res[0]}
        print(f"  {lname:8s} eps=0.05: risk {m['risk']:.3f} viol {viol}/100  acc {m['acc']:.3f}  "
              f"crops saved {m['crops_saved']:.1%}  tokens saved {m['tokens_saved']:.1%}  harmful-zoom {m['harmful_zoom']:.3f}")

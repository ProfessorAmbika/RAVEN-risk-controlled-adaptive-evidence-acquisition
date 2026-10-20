"""Log utilities shared by the analysis scripts (load, tensors, fit_logreg, evaluate) and the pilot
analysis of multi-step crop trajectories written by run_gonogo.py.

1. Splits samples by id: 40% train the sufficiency score g (logistic regression),
   the rest is split repeatedly into calibration / test.
2. Calibrates the stopping threshold with Learn-then-Test (ltt_calibrate.py) on the
   label-free fidelity loss 1[answer_t != answer_T].
3. Reports validity, compute saved, and accuracy against global-only and full-budget runs,
   for the learned score and for a confidence-only baseline.

Usage:  python analyze_gonogo.py logs/vstar_qwen3b.jsonl logs/textvqa_qwen3b.jsonl
Pilot decision rule: GO if, at eps = 5%, the certified rule
saves >= 30% of crop acquisitions; RE-THINK if it saves < 10%.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ltt_calibrate import calibrate, stop_step  # noqa: E402

FEATURES = ["margin0", "mean_logp", "min_logp", "stable", "streak", "residual_att", "t_frac", "changed_from_global"]


def load(paths):
    recs = []
    for p in paths:
        with open(p) as f:
            recs += [json.loads(l) for l in f if l.strip()]
    return recs


def tensors(recs):
    """Per-sample arrays over steps t = 0..T: features, fidelity loss, correctness, cost."""
    T = len(recs[0]["steps"]) - 1
    n = len(recs)
    X = np.zeros((n, T + 1, len(FEATURES)))
    loss = np.zeros((n, T + 1))
    correct = np.zeros((n, T + 1))
    cost_tok = np.zeros((n, T + 1))
    for i, r in enumerate(recs):
        st = r["steps"]
        ref = st[-1]["pred"]
        streak = 0
        for t, s in enumerate(st):
            stable = float(t > 0 and s["pred"] == st[t - 1]["pred"])
            streak = streak + 1 if stable else 0
            X[i, t] = [s["margin0"], s["mean_logp"], s["min_logp"], stable, streak,
                       s["residual_att"], t / T, float(s["pred"] != st[0]["pred"])]
            loss[i, t] = float(s["pred"] != ref)
            correct[i, t] = s["correct"]
            cost_tok[i, t] = s["vis_tokens"]
    loss[:, T] = 0.0
    return X, loss, correct, cost_tok


def fit_logreg(X, y, l2=1e-2, iters=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return lambda Xn: 1 / (1 + np.exp(-(((Xn - mu) / sd) @ w + b)))


def evaluate(score, loss, correct, cost_tok, lam):
    rows = np.arange(len(score))
    t = stop_step(score, lam)
    T = score.shape[1] - 1
    return {
        "risk": loss[rows, t].mean(),
        "acc": correct[rows, t].mean(),
        "crops_saved": 1 - t.mean() / T,
        "tokens_saved": 1 - cost_tok[rows, t].mean() / cost_tok[:, T].mean(),
        "harmful_zoom": ((correct[:, 0] == 1) & (correct[rows, t] < 1)).mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = load(args.logs)
    X, loss, correct, cost_tok = tensors(recs)
    n, T1, _ = X.shape
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    tr, rest = perm[: int(0.4 * n)], perm[int(0.4 * n):]

    g = fit_logreg(X[tr].reshape(-1, X.shape[2]), (loss[tr] == 0).reshape(-1))
    scores = {
        "RAVEN (learned score)": g(X.reshape(-1, X.shape[2])).reshape(n, T1),
        "confidence only (margin0)": X[:, :, 0],
    }

    print(f"samples: {n} (train {len(tr)}, cal+test {len(rest)}), full budget T = {T1 - 1} crops")
    print(f"accuracy  global-only: {correct[:, 0].mean():.3f}   full budget: {correct[:, -1].mean():.3f}   "
          f"oracle best step: {correct.max(1).mean():.3f}")
    print(f"answers changed by full budget: {loss[:, 0].mean():.3f} of samples\n")

    verdict = None
    for name, S in scores.items():
        print(f"== {name}")
        for eps in args.eps:
            res, viol = [], 0
            for _ in range(args.splits):
                p = rng.permutation(rest)
                cal, te = p[: len(p) // 2], p[len(p) // 2:]
                c = calibrate(S[cal], loss[cal], cost_tok[cal], eps=eps, delta=args.delta)
                r = evaluate(S[te], loss[te], correct[te], cost_tok[te], c.lam)
                viol += r["risk"] > eps
                res.append(r)
            m = {k: np.mean([r[k] for r in res]) for k in res[0]}
            print(f"  eps={eps:.2f}: violations {viol}/{args.splits} (allowed ~{args.delta:.0%})  "
                  f"risk {m['risk']:.3f}  acc {m['acc']:.3f}  crops saved {m['crops_saved']:.1%}  "
                  f"tokens saved {m['tokens_saved']:.1%}  harmful-zoom {m['harmful_zoom']:.3f}")
            if name.startswith("RAVEN") and abs(eps - 0.05) < 1e-9:
                verdict = m["crops_saved"]
        print()

    # uncalibrated fixed threshold, the way current methods stop
    rows = np.arange(n)
    for thr in (0.5, 0.8, 0.9):
        t = stop_step(X[:, :, 0], thr)
        print(f"fixed margin threshold {thr}: realised fidelity risk {loss[rows, t].mean():.3f}, "
              f"crops saved {1 - t.mean() / (T1 - 1):.1%}")

    # per-benchmark view (same learned score, calibrated on the pooled cal split)
    benches = np.array([r.get("bench", "?") for r in recs])
    if len(set(benches)) > 1:
        print("\nPer benchmark at eps=0.05 (pooled calibration):")
        S = scores["RAVEN (learned score)"]
        for b in sorted(set(benches)):
            res = []
            for _ in range(args.splits):
                p = rng.permutation(rest)
                cal, te = p[: len(p) // 2], p[len(p) // 2:]
                c = calibrate(S[cal], loss[cal], cost_tok[cal], eps=0.05, delta=args.delta)
                te = te[benches[te] == b]
                if len(te):
                    res.append(evaluate(S[te], loss[te], correct[te], cost_tok[te], c.lam))
            m = {k: np.mean([r[k] for r in res]) for k in res[0]}
            idx = benches == b
            print(f"  {b}: n={idx.sum()}  acc global {correct[idx, 0].mean():.3f} full {correct[idx, -1].mean():.3f} "
                  f"RAVEN {m['acc']:.3f}  risk {m['risk']:.3f}  crops saved {m['crops_saved']:.1%}  "
                  f"tokens saved {m['tokens_saved']:.1%}")

    if verdict is not None:
        decision = "GO" if verdict >= 0.30 else ("RE-THINK" if verdict < 0.10 else "BORDERLINE")
        print(f"\nDecision at eps=5%: crops saved {verdict:.1%} -> {decision}")


if __name__ == "__main__":
    main()

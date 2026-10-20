"""Does a value-of-information head that CHOOSES the next action beat fixed-action ladders at matched token budgets?

Offline, on logs that answer the same questions three ways (Qwen2.5-VL-7B, MME-RealWorld):
  stop   : the 1,024-token view (rung 0)
  view   : escalate to the 4,096-token view   (cost = rung 0 + 4,096 view)
  crop   : escalate to the ViCrop crop        (cost = ViCrop total; rung 0's pass doubles as its attention pass)
Per domain and per random split (40% train / 60% test), logistic heads on the rung-0 state (hidden-state PCA +
confidence features) predict P(correct) for each action. A policy picks, per question,
    argmax_a  p_hat(a) - mu * cost(a)
and sweeping mu traces an accuracy-vs-tokens frontier. Compared policies:
  voi-choose : a in {stop, view, crop}                  (the VoI head choosing between actions)
  voi-view   : a in {stop, view}                        (same head, one action only)
  voi-crop   : a in {stop, crop}
  oracle     : best achievable with perfect knowledge of correctness, same action set as voi-choose
Reported: accuracy at matched mean-token budgets (interpolated on each frontier), averaged over splits.

  python voi_test.py --l1024 logs/mmerw_7b_l1024.jsonl --vicrop logs/mmerw_7b_vicrop2.jsonl --out results/voi_test.json
"""
import argparse, base64, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gonogo import load, tensors, fit_logreg  # noqa: E402

MUS = np.r_[0.0, np.geomspace(1e-6, 1e-2, 120), 1.0]


def frontier(P, C, A):
    """Mean (tokens, accuracy-estimate index) of the policy argmax_a P - mu*C over actions A, for each mu."""
    out = []
    for mu in MUS:
        u = P[:, A] - mu * C[:, A]
        out.append(np.array(A)[u.argmax(1)])
    return out  # list of per-question action indices


def curve(choices, COR, C):
    rows = np.arange(COR.shape[0])
    pts = sorted({(float(C[rows, a].mean()), float(COR[rows, a].mean())) for a in choices})
    t, acc = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    best = np.maximum.accumulate(acc)  # upper envelope: a policy can always spend less and do as well by randomising
    return t, best


def acc_at(t, acc, budget):
    if budget <= t[0]:
        return acc[0]
    if budget >= t[-1]:
        return acc[-1]
    return float(np.interp(budget, t, acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1024", required=True)
    ap.add_argument("--vicrop", required=True)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    V = {json.loads(l)["qid"]: json.loads(l) for l in open(a.vicrop) if l.strip()}
    allr = [r for r in load([a.l1024]) if r["qid"] in V]
    res = {}
    for dom in sorted({r["category"].split("/")[0] for r in allr}):
        recs = [r for r in allr if r["category"].split("/")[0] == dom]
        n = len(recs)
        Hx = np.stack([np.frombuffer(base64.b64decode(r["hidden"]), np.float16).astype(np.float32) for r in recs])
        X, _, cor, cost = tensors(recs)
        vc = np.array([V[r["qid"]]["correct"] for r in recs])
        vt = np.array([V[r["qid"]]["vis_tokens_total"] for r in recs], float)
        COR = np.stack([cor[:, 0], cor[:, 1], vc], 1)            # stop, view, crop
        C = np.stack([cost[:, 0], cost[:, 1], vt], 1)
        budgets = {"4096 view": float(C[:, 1].mean() - C[:, 0].mean()), "ViCrop": float(C[:, 2].mean())}
        budgets.update({f"{int(f * 100)}% of 4096 view": f * budgets["4096 view"] for f in (0.6, 0.8)})
        rng = np.random.default_rng(0)
        acc = {k: {b: [] for b in budgets} for k in ("voi-choose", "voi-view", "voi-crop", "oracle")}
        for _ in range(a.splits):
            p = rng.permutation(n)
            tr, te = p[: int(0.4 * n)], p[int(0.4 * n):]
            mu, sd = Hx[tr].mean(0), Hx[tr].std(0) + 1e-6
            _, _, Vt = np.linalg.svd((Hx[tr] - mu) / sd, full_matrices=False)
            F = np.hstack([((Hx - mu) / sd) @ Vt[: a.pca].T, X[:, 0]])
            P = np.stack([fit_logreg(F[tr], COR[tr, k], l2=1e-1)(F) for k in range(3)], 1)
            for name, A, PP in (("voi-choose", [0, 1, 2], P), ("voi-view", [0, 1], P), ("voi-crop", [0, 2], P),
                                ("oracle", [0, 1, 2], COR + 1e-3 * (C.max() - C) / C.max())):
                t, ac = curve(frontier(PP[te], C[te], A), COR[te], C[te])
                for b, v in budgets.items():
                    acc[name][b].append(acc_at(t, ac, v))
        res[dom] = {k: {b: float(np.mean(v)) for b, v in d.items()} for k, d in acc.items()}
        res[dom]["_splits"] = {k: {b: [float(x) for x in v] for b, v in d.items()} for k, d in acc.items()}
        res[dom]["_budgets"] = budgets
        res[dom]["_fixed"] = {"1024 view": float(COR[:, 0].mean()), "4096 view": float(COR[:, 1].mean()),
                              "ViCrop": float(COR[:, 2].mean())}
        print(f"\n[{dom}]  fixed: 1024 {COR[:,0].mean():.3f}  4096 {COR[:,1].mean():.3f} ({budgets['4096 view']:.0f} tok)"
              f"  ViCrop {COR[:,2].mean():.3f} ({budgets['ViCrop']:.0f} tok)")
        for k in acc:
            print(f"  {k:11s} " + "  ".join(f"@{b}: {res[dom][k][b]:.3f}" for b in budgets))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Baseline comparison for the two-rung ladder at matched conditions (same model, same questions, same splits,
same cost accounting, measured RTX 3090 latency).

Per domain, 100 random 40/30/30 train/calibration/test splits (as probe.py). On each split:
  fixed rung 0 / fixed rung 1      always the cheap / always the expensive view
  oracle                           stop iff the rung-0 answer already has zero loss (upper bound, not deployable)
  RAVEN (LTT)                      fixed-sequence Learn-then-Test, Hoeffding-Bentkus, delta = 0.1 (the method)
  plug-in threshold                same score and grid, stop scanning when the calibration risk exceeds eps
                                   (no p-value: what an uncalibrated "tune the threshold on held-out data" gives)
  CRC                              conformal risk control (Angelopoulos et al., ICLR 2024): (n R_hat + 1)/(n + 1) <= eps,
                                   controls the expected risk, not the probability of exceeding eps
  confidence router, matched       confidence-feature score, threshold set on the calibration split to RAVEN's mean
                                   cost on that split (no guarantee)
  random routing, matched          escalates a random share of questions giving RAVEN's mean cost on the test split
Reported on the test split: accuracy, mean visual tokens, mean seconds (measured per-view means, see --timing),
test risk, share of splits whose test risk exceeds eps.
A second part checks validity exactly (as audit_exact.py): router fit on 40%, the other 60% is the population,
calibration sets of n_cal drawn with replacement, exact population risk, violation rate over B draws.

  python baselines.py --ladder logs/mmerw_7b_l1024.jsonl --timing logs/time_qwen_1024.jsonl logs/time_qwen_4096.jsonl \
      --out results/baselines_qwen7b_l1024.json
"""
import argparse, base64, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gonogo import load, tensors, fit_logreg  # noqa: E402
from ltt_calibrate import hb_pvalue  # noqa: E402

GRID = np.linspace(1.0, 0.0, 201)


def risks_on_grid(s, l):
    """Empirical risk of stopping when s >= lam, for every lam in GRID (escalated questions have loss 0)."""
    return ((s[None, :] >= GRID[:, None]) * l[None, :]).mean(1)


def lam_ltt(s, l, eps, delta, pv):
    k = np.rint(risks_on_grid(s, l) * len(s)).astype(int)
    fail = np.nonzero(pv[k] > delta)[0]
    j = fail[0] if len(fail) else len(GRID)
    return np.inf if j == 0 else GRID[j - 1]


def lam_scan(s, l, ok):
    """Fixed-sequence scan with an arbitrary acceptance rule ok(risk_hat) (plug-in, CRC)."""
    fail = np.nonzero(~ok(risks_on_grid(s, l)))[0]
    j = fail[0] if len(fail) else len(GRID)
    return np.inf if j == 0 else GRID[j - 1]


def lam_matched(s, cost0, cost1, target):
    """Highest-coverage threshold whose mean cost on this split is >= target (closest match from above)."""
    c = np.array([np.where(s >= lam, cost0, cost1).mean() for lam in GRID])
    j = np.nonzero(c >= target)[0]
    return GRID[j[-1]] if len(j) else GRID[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--timing", nargs=2, help="timed single-view logs (rung 0, rung 1) from run_backbone.py")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--n_cal", type=int, default=300)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--domains", nargs="+", help="only these domains (default: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dom = lambda r: r["category"].split("/")[0]
    timing = {}
    if a.timing:  # per domain: rung-0 total seconds, rung-1 total seconds, rung-1 model seconds
        T0, T1 = (load([p]) for p in a.timing)
        for d in {dom(r) for r in T0}:
            timing[d] = (np.mean([r["seconds"] for r in T0 if dom(r) == d]),
                         np.mean([r["seconds"] for r in T1 if dom(r) == d]),
                         np.mean([r["sec_model"] for r in T1 if dom(r) == d]))
    allr = load([a.ladder])
    res = {}
    for d in sorted({dom(r) for r in allr}):
        if a.domains and d not in a.domains:
            continue
        recs = [r for r in allr if dom(r) == d]
        n = len(recs)
        Hx = np.stack([np.frombuffer(base64.b64decode(r["hidden"]), np.float16).astype(np.float32) for r in recs])
        X, fid, cor, cost = tensors(recs)
        c0, c1 = cost[:, 0], cost[:, 1]                                 # c1 = cumulative (both passes)
        t0, t1, m1 = timing.get(d, (np.nan, np.nan, np.nan))
        sec = lambda esc: t0 + esc * m1                                 # escalation re-uses the decoded image
        res[d] = {"_fixed": {"rung0": dict(acc=cor[:, 0].mean(), tokens=c0.mean(), seconds=t0),
                             "rung1": dict(acc=cor[:, 1].mean(), tokens=(c1 - c0).mean(), seconds=t1)}}
        rng = np.random.default_rng(0)
        for lname in ("fidelity", "acc-gap"):
            L = fid[:, 0] if lname == "fidelity" else np.clip(cor[:, 1] - cor[:, 0], 0, None)
            rows = {}
            for _ in range(a.reps):
                p = rng.permutation(n); tr, ca, te = p[:int(.4 * n)], p[int(.4 * n):int(.7 * n)], p[int(.7 * n):]
                mu, sd = Hx[tr].mean(0), Hx[tr].std(0) + 1e-6
                _, _, Vt = np.linalg.svd((Hx[tr] - mu) / sd, full_matrices=False)
                Z = ((Hx - mu) / sd) @ Vt[:a.pca].T
                y = (L[tr] == 0).astype(float)
                sp = fit_logreg(np.hstack([Z[tr], X[tr, 0]]), y, l2=1e-1)(np.hstack([Z, X[:, 0]]))
                sc = fit_logreg(X[tr, 0], y)(X[:, 0])
                nc = len(ca)
                pv = {e: np.array([hb_pvalue(k / nc, nc, e) for k in range(nc + 1)]) for e in a.eps}

                def record(name, stop):
                    esc = 1 - stop[te].mean()
                    rows.setdefault(name, []).append(dict(
                        acc=np.where(stop, cor[:, 0], cor[:, 1])[te].mean(),
                        tokens=np.where(stop, c0, c1)[te].mean(), seconds=sec(esc),
                        risk=(L * stop)[te].mean(), stopped=stop[te].mean()))
                record("oracle", L == 0)
                for e in a.eps:
                    lam = lam_ltt(sp[ca], L[ca], e, a.delta, pv[e])
                    record(f"RAVEN (LTT) eps={e}", sp >= lam)
                    record(f"plug-in eps={e}", sp >= lam_scan(sp[ca], L[ca], lambda r: r <= e))
                    record(f"CRC eps={e}", sp >= lam_scan(sp[ca], L[ca], lambda r: (nc * r + 1) / (nc + 1) <= e))
                    target = np.where(sp >= lam, c0, c1)[ca].mean()
                    record(f"confidence, matched to RAVEN eps={e}", sc >= lam_matched(sc[ca], c0[ca], c1[ca], target))
                    q = 1 - (np.where(sp >= lam, c0, c1)[te].mean() - c0[te].mean()) / (c1[te] - c0[te]).mean()
                    record(f"random, matched to RAVEN eps={e}", rng.random(n) < np.clip(q, 0, 1))
            out = {}
            for name, rs in rows.items():
                m = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
                e = float(name.split("eps=")[1]) if "eps=" in name else None
                if e is not None:
                    m["test_viol"] = float(np.mean([r["risk"] > e for r in rs]))
                out[name] = m
            # exact validity check of the three calibration rules (probe router, one router fit)
            p = np.random.default_rng(1000).permutation(n); tr, pop = p[:int(.4 * n)], p[int(.4 * n):]
            mu, sd = Hx[tr].mean(0), Hx[tr].std(0) + 1e-6
            _, _, Vt = np.linalg.svd((Hx[tr] - mu) / sd, full_matrices=False)
            Z = ((Hx - mu) / sd) @ Vt[:a.pca].T
            sp = fit_logreg(np.hstack([Z[tr], X[tr, 0]]), (L[tr] == 0).astype(float), l2=1e-1)(np.hstack([Z, X[:, 0]]))
            for e in a.eps:
                pv = np.array([hb_pvalue(k / a.n_cal, a.n_cal, e) for k in range(a.n_cal + 1)])
                rules = {"RAVEN (LTT)": lambda s, l: lam_ltt(s, l, e, a.delta, pv),
                         "plug-in": lambda s, l: lam_scan(s, l, lambda r: r <= e),
                         "CRC": lambda s, l: lam_scan(s, l, lambda r: (a.n_cal * r + 1) / (a.n_cal + 1) <= e)}
                v = {k: 0 for k in rules}
                for _ in range(a.B):
                    c = rng.integers(0, len(pop), a.n_cal)
                    for k, f in rules.items():
                        v[k] += (L[pop] * (sp[pop] >= f(sp[pop][c], L[pop][c]))).mean() > e
                for k in rules:
                    out[f"{k} eps={e}"]["exact_viol"] = v[k] / a.B
            res[d][lname] = out
            print(f"\n[{d}] {lname}   fixed rung0 acc {cor[:,0].mean():.3f} tok {c0.mean():.0f} {t0:.2f}s   "
                  f"rung1 acc {cor[:,1].mean():.3f} tok {(c1-c0).mean():.0f} {t1:.2f}s")
            for name, m in out.items():
                extra = "".join(f"  {k} {m[k]:.3f}" for k in ("test_viol", "exact_viol") if k in m)
                print(f"  {name:38s} acc {m['acc']:.3f}  tok {m['tokens']:5.0f}  {m['seconds']:.2f}s  "
                      f"risk {m['risk']:.3f}  stop {m['stopped']:.2f}{extra}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

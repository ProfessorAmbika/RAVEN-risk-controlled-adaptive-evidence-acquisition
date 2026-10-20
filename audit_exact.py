"""Exact check of the LTT guarantee, one number per configuration, with no test-set noise.

The guarantee is P( R(lambda_hat) > eps ) <= delta, where R is the *population* risk and the probability is over the
calibration draw. Counting "test risk > eps" on a finite test split mixes this with test-set noise. Here, per domain:
  1. a router-training split (40%) fits the score once (probe / confidence / random), as in probe.py;
  2. the remaining questions ARE the population: calibration sets of size n_cal are drawn i.i.d. from them
     (with replacement), lambda_hat is chosen with calibrate()'s rule, and R(lambda_hat) is the exact mean loss
     over the whole population;
  3. this is repeated B times per router seed; the violation rate gets a Clopper-Pearson 95% interval.
Under the theorem the rate is <= delta for every configuration and every router seed.
The fixed-sequence search below reproduces ltt_calibrate.calibrate exactly (checked against it in --selftest).

  python audit_exact.py --ladder l1024=logs/mmerw_7b_l1024.jsonl --ladder l256=logs/mmerw_7b_l256.jsonl \
      --out results/audit_exact.json
"""
import argparse, base64, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gonogo import load, tensors, fit_logreg  # noqa: E402
from ltt_calibrate import hb_pvalue, calibrate  # noqa: E402

GRID = np.linspace(1.0, 0.0, 201)


def beta_ppf(q, a, b, lo=0.0, hi=1.0):
    """Inverse regularised incomplete beta by bisection (no scipy)."""
    def cdf(x):  # P(Binom(a+b-1, x) >= a) = I_x(a, b) for integer a, b
        n = a + b - 1
        if x <= 0.0 or x >= 1.0:
            return float(x >= 1.0)
        lx, l1x = math.log(x), math.log1p(-x)
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lx + (n - j) * l1x)
                   for j in range(a, n + 1))
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if cdf(m) < q else (lo, m)
    return (lo + hi) / 2


def clopper_pearson(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta_ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def lam_hat(s, l0, pv, delta):
    """Fixed-sequence LTT over GRID for a two-rung ladder with binary loss l0 at rung 0 and 0 at rung 1."""
    k = (s[None, :] >= GRID[:, None]) @ l0                     # losses at each threshold
    fail = np.nonzero(pv[k.astype(int)] > delta)[0]
    j = fail[0] if len(fail) else len(GRID)
    return np.inf if j == 0 else GRID[j - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", action="append", required=True, help="name=path")
    ap.add_argument("--n_cal", type=int, default=300)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    pv = {e: np.array([hb_pvalue(k / a.n_cal, a.n_cal, e) for k in range(a.n_cal + 1)]) for e in a.eps}
    rows = []
    for spec in a.ladder:
        name, path = spec.split("=", 1)
        allr = load([path])
        for dom in sorted({r["category"].split("/")[0] for r in allr}):
            recs = [r for r in allr if r["category"].split("/")[0] == dom]
            Hx = np.stack([np.frombuffer(base64.b64decode(r["hidden"]), np.float16).astype(np.float32) for r in recs])
            X, fid, cor, _ = tensors(recs)
            losses = {"fidelity": fid[:, 0], "acc-gap": np.clip(cor[:, 1] - cor[:, 0], 0, None)}
            n = len(recs)
            for seed in range(a.seeds):
                rng = np.random.default_rng(1000 + seed)
                perm = rng.permutation(n)
                tr, pop = perm[: int(0.4 * n)], perm[int(0.4 * n):]
                mu, sd = Hx[tr].mean(0), Hx[tr].std(0) + 1e-6
                _, _, Vt = np.linalg.svd((Hx[tr] - mu) / sd, full_matrices=False)
                Z = ((Hx - mu) / sd) @ Vt[: a.pca].T
                for lname, L in losses.items():
                    y = (L[tr] == 0).astype(float)
                    scores = {"probe": fit_logreg(np.hstack([Z[tr], X[tr, 0]]), y, l2=1e-1)(np.hstack([Z, X[:, 0]])),
                              "confidence": fit_logreg(X[tr, 0], y)(X[:, 0]),
                              "random": rng.random(n)}
                    for rname, s in scores.items():
                        sp, lp = s[pop], L[pop]
                        for e in a.eps:
                            risks, saved = [], []
                            for _ in range(a.B):
                                c = rng.integers(0, len(pop), a.n_cal)
                                lam = lam_hat(sp[c], lp[c], pv[e], a.delta)
                                if a.selftest and len(risks) < 20:
                                    S = np.stack([sp[c], np.ones(a.n_cal)], 1)
                                    ref = calibrate(S, np.stack([lp[c], np.zeros(a.n_cal)], 1), np.zeros((a.n_cal, 2)),
                                                    eps=e, delta=a.delta).lam
                                    assert ref == lam, (ref, lam)
                                stop = sp >= lam
                                risks.append(float((lp * stop).mean()))
                                saved.append(float(stop.mean()))
                            risks = np.array(risks)
                            v = int((risks > e).sum())
                            lo, hi = clopper_pearson(v, a.B)
                            rows.append(dict(ladder=name, domain=dom, seed=seed, loss=lname, router=rname, eps=e,
                                             viol=v, B=a.B, rate=v / a.B, ci=[lo, hi], mean_risk=float(risks.mean()),
                                             max_risk=float(risks.max()), stopped_early=float(np.mean(saved))))
                            print(f"{name:6s} {dom[:12]:12s} s{seed} {lname:8s} {rname:10s} eps={e:.2f}  "
                                  f"viol {v:4d}/{a.B} = {v / a.B:.3f} [{lo:.3f},{hi:.3f}]  mean R {risks.mean():.3f}  "
                                  f"max R {risks.max():.3f}  stop0 {np.mean(saved):.2f}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=0)


if __name__ == "__main__":
    main()

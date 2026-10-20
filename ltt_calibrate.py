"""Risk-controlled stopping thresholds for adaptive visual evidence acquisition.

The calibration step of RAVEN.
Implements Learn-then-Test (Angelopoulos et al., Ann. Appl. Stat. 2025) with
Hoeffding-Bentkus p-values and fixed-sequence testing over a 1-D threshold path.

Inputs, per calibration sample i and acquisition step t = 0..T:
  score[i, t]  sufficiency score after step t (higher = more confident evidence suffices)
  loss[i, t]   loss if we stop after step t, in [0, 1]
               (fidelity loss: 1[y_hat_t != y_hat_ref]; needs no labels)
  cost[i, t]   cumulative compute spent after step t (tokens, FLOPs or ms)
Step T is the full budget; the policy always stops there.

Guarantee (exchangeable calibration/test data):
  P( E[loss at stop] <= eps ) >= 1 - delta  for the returned threshold.

Only numpy is required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(Binom(n, p) <= k), computed in log space."""
    if k >= n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return -math.inf
    lp, lq = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq
        for j in range(k + 1)
    ]
    m = max(terms)
    return m + math.log(sum(math.exp(x - m) for x in terms))


def hb_pvalue(risk_hat: float, n: int, eps: float) -> float:
    """Hoeffding-Bentkus p-value for H0: true risk > eps (losses bounded in [0, 1])."""
    if risk_hat >= eps:
        return 1.0
    # Hoeffding part
    a, b = risk_hat, eps
    kl = a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b)) if a > 0 else -math.log(1 - b)
    p_h = math.exp(-n * kl)
    # Bentkus part (the tolerance stops a float count k from rounding up to k+1)
    p_b = math.e * math.exp(_log_binom_cdf(math.ceil(n * risk_hat - 1e-9), n, eps))
    return min(p_h, p_b, 1.0)


def stop_step(score: np.ndarray, lam: float) -> np.ndarray:
    """First step t with score >= lam, else the last step."""
    n, T1 = score.shape
    hit = score >= lam
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), T1 - 1)
    return first


@dataclass
class Calibration:
    lam: float
    emp_risk: float
    mean_cost: float
    tested: int


def calibrate(score, loss, cost, eps=0.05, delta=0.1, grid=None, groups=None) -> Calibration:
    """Fixed-sequence LTT: scan thresholds from most to least conservative, stop at first non-rejection.

    groups: optional group label per sample (e.g. the same question asked several times). Samples in a group
    are not independent, so the p-value treats each group's mean loss as one bounded observation
    (n = number of groups), which keeps the guarantee valid when groups, not samples, are exchangeable."""
    score, loss, cost = map(np.asarray, (score, loss, cost))
    n = score.shape[0]
    if groups is not None:
        _, gidx = np.unique(np.asarray(groups), return_inverse=True)
        gsize = np.bincount(gidx)
    if grid is None:
        grid = np.linspace(1.0, 0.0, 201)  # descending: high threshold = conservative
    rows = np.arange(n)
    best = None
    for j, lam in enumerate(grid):
        t = stop_step(score, lam)
        r = float(loss[rows, t].mean())
        if groups is not None:
            p = hb_pvalue(float((np.bincount(gidx, loss[rows, t]) / gsize).mean()), len(gsize), eps)
        else:
            p = hb_pvalue(r, n, eps)
        if p > delta:
            break
        best = Calibration(float(lam), r, float(cost[rows, t].mean()), j + 1)
    if best is None:  # nothing certified: always use the full budget
        T = score.shape[1] - 1
        best = Calibration(math.inf, float(loss[:, T].mean()), float(cost[:, T].mean()), len(grid))
    return best


def evaluate(score, loss, cost, lam):
    rows = np.arange(score.shape[0])
    t = stop_step(score, lam)
    return float(loss[rows, t].mean()), float(cost[rows, t].mean())


def _synthetic(n, T=6, seed=0):
    """Toy data: each sample has a latent step k* after which the answer matches the reference."""
    rng = np.random.default_rng(seed)
    k_star = rng.choice(T + 1, size=n, p=[0.45, 0.2, 0.12, 0.09, 0.06, 0.05, 0.03])
    t = np.arange(T + 1)[None, :]
    loss = (t < k_star[:, None]).astype(float)
    # noisy, imperfectly calibrated sufficiency score
    logits = 2.5 * (t - k_star[:, None] + 0.5) + rng.normal(0, 1.5, size=(n, T + 1))
    score = 1 / (1 + np.exp(-logits))
    loss[:, T] = 0.0  # full budget equals the reference by definition
    cost = np.cumsum(np.r_[1.0, np.full(T, 0.6)])[None, :].repeat(n, 0)
    return score, loss, cost


if __name__ == "__main__":
    eps, delta = 0.05, 0.1
    violations, savings = 0, []
    trials = 200
    for s in range(trials):
        cal = _synthetic(1000, seed=2 * s)
        test = _synthetic(5000, seed=2 * s + 1)
        c = calibrate(*cal, eps=eps, delta=delta)
        r, cst = evaluate(*test, c.lam)
        violations += r > eps
        savings.append(1 - cst / test[2][0, -1])
    print(f"eps={eps} delta={delta}: test-risk violations {violations}/{trials} "
          f"(must be <= ~{delta:.0%}), mean compute saved {np.mean(savings):.1%}")

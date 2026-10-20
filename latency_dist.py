"""Per-question latency distribution of RAVEN on the timed questions, against the fixed views.

The timed questions (run_backbone.py logs) are held out: the router is fit on 40% of the remaining questions of the
domain and the threshold is calibrated with LTT on n_cal others, then applied to the timed questions. Per timed
question: RAVEN time = t(rung 0, total) + [escalated] * t(rung 1, model time), since escalation re-uses the decoded
image. Repeated over --reps router/calibration draws; the routing (PCA projection + logistic) and the calibration
are timed on CPU as well.

  python latency_dist.py --ladder logs/mmerw_7b_l1024.jsonl --timing logs/time_qwen_1024.jsonl logs/time_qwen_4096.jsonl
"""
import argparse, base64, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gonogo import load, tensors, fit_logreg  # noqa: E402
from ltt_calibrate import calibrate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--timing", nargs=2, required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n_cal", type=int, default=300)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T0 = {r["qid"]: r for r in load([a.timing[0]])}
    T1 = {r["qid"]: r for r in load([a.timing[1]])}
    allr = load([a.ladder])
    dom = lambda r: r["category"].split("/")[0]
    q = lambda v: {f"p{p}": float(np.percentile(v, p)) for p in (10, 50, 90, 99)} | {"mean": float(np.mean(v))}
    res = {}
    for d in sorted({dom(r) for r in allr}):
        recs = [r for r in allr if dom(r) == d]
        timed = np.array([r["qid"] in T0 and r["qid"] in T1 for r in recs])
        Hx = np.stack([np.frombuffer(base64.b64decode(r["hidden"]), np.float16).astype(np.float32) for r in recs])
        X, fid, cor, cost = tensors(recs)
        L = fid[:, 0]
        te = np.nonzero(timed)[0]
        t0 = np.array([T0[recs[i]["qid"]]["seconds"] for i in te])
        t1 = np.array([T1[recs[i]["qid"]]["seconds"] for i in te])
        m1 = np.array([T1[recs[i]["qid"]]["sec_model"] for i in te])
        res[d] = {"n_timed": int(len(te)), "fixed rung0": q(t0), "fixed rung1": q(t1)}
        rng = np.random.default_rng(0)
        rest = np.nonzero(~timed)[0]
        route_ms, cal_ms = [], []
        for e in a.eps:
            times, stops = [], []
            for _ in range(a.reps):
                p = rng.permutation(rest); tr, ca = p[:int(.4 * len(rest))], p[int(.4 * len(rest)):][:a.n_cal]
                mu, sd = Hx[tr].mean(0), Hx[tr].std(0) + 1e-6
                _, _, Vt = np.linalg.svd((Hx[tr] - mu) / sd, full_matrices=False)
                P = Vt[:64]
                f = fit_logreg(np.hstack([((Hx[tr] - mu) / sd) @ P.T, X[tr, 0]]), (L[tr] == 0).astype(float), l2=1e-1)
                s = f(np.hstack([((Hx - mu) / sd) @ P.T, X[:, 0]]))
                c0 = time.perf_counter()
                cal = calibrate(np.stack([s[ca], np.ones(len(ca))], 1), np.stack([L[ca], np.zeros(len(ca))], 1),
                                cost[ca], eps=e, delta=0.1)
                cal_ms.append(1e3 * (time.perf_counter() - c0))
                c0 = time.perf_counter()
                for i in te[:20]:  # routing cost of one question, as deployed
                    f(np.hstack([((Hx[i:i + 1] - mu) / sd) @ P.T, X[i:i + 1, 0]]))
                route_ms.append(1e3 * (time.perf_counter() - c0) / 20)
                stop = s[te] >= cal.lam
                times.append(t0 + (~stop) * m1); stops.append(stop.mean())
            tt = np.concatenate(times)
            res[d][f"RAVEN eps={e}"] = q(tt) | {"stopped": float(np.mean(stops)),
                                                 "mean_sd_over_draws": float(np.std([x.mean() for x in times]))}
        res[d]["routing_ms_per_question"] = float(np.median(route_ms))
        res[d]["calibration_ms"] = float(np.median(cal_ms))
        print(f"\n[{d}] {len(te)} timed questions; routing {np.median(route_ms):.2f} ms/question, "
              f"calibration {np.median(cal_ms):.0f} ms once")
        for k in ["fixed rung0", "fixed rung1"] + [f"RAVEN eps={e}" for e in a.eps]:
            v = res[d][k]
            print(f"  {k:16s} mean {v['mean']:.2f}  p10 {v['p10']:.2f}  p50 {v['p50']:.2f}  p90 {v['p90']:.2f}  "
                  f"p99 {v['p99']:.2f}" + (f"  stopped {v['stopped']:.2f}  sd(mean) {v['mean_sd_over_draws']:.3f}"
                                           if "stopped" in v else ""), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

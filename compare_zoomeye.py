"""Zoom Eye against the fixed views, the ViCrop rung and RAVEN on exactly the questions Zoom Eye answered.

Per domain, on the Zoom Eye questions (run_zoomeye.py log):
  fixed 1,024 / fixed 4,096 view   rung 0 / rung 1 of the resolution ladder (tokens of that view alone)
  ViCrop                           rung 1 of the crop ladder (cumulative: 1,024 view + crop)
  RAVEN eps                        probe router + LTT (fidelity loss, delta = 0.1) as latency_dist.py: the Zoom Eye
                                   questions are held out, the router is fit on 40% of the other questions of the
                                   domain and calibrated on n_cal others; mean over --reps draws
  Zoom Eye                         accuracy, visual tokens summed over its passes, passes, measured seconds
Seconds of the fixed views and RAVEN: per-domain means of the controlled timing run (as baselines.py; RAVEN =
t(1,024) + escalated x t(4,096 model time)). Paired accuracy differences: bootstrap over questions, 95% interval.

  python compare_zoomeye.py --zoomeye logs/mmerw_zoomeye_q7b.jsonl --ladder logs/mmerw_7b_l1024.jsonl \
      --crop logs/mmerw_7b_lvicrop_reuse.jsonl --timing logs/time_qwen_1024.jsonl logs/time_qwen_4096.jsonl --out ze.json
"""
import argparse, base64, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gonogo import load, tensors, fit_logreg  # noqa: E402
from ltt_calibrate import calibrate  # noqa: E402


def boot_ci(d, rng, B=5000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoomeye", required=True)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--crop", required=True)
    ap.add_argument("--timing", nargs=2, required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n_cal", type=int, default=300)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dom = lambda r: r["category"].split("/")[0]
    ZE = {r["qid"]: r for r in map(json.loads, open(a.zoomeye))}
    crop = {r["qid"]: r for r in load([a.crop])}
    T0, T1 = (load([p]) for p in a.timing)
    allr = load([a.ladder])
    rng = np.random.default_rng(0)
    res, pooled = {}, {}
    for d in sorted({dom(r) for r in allr}):
        recs = [r for r in allr if dom(r) == d]
        held = np.array([r["qid"] in ZE for r in recs])
        if not held.any():
            continue
        te, rest = np.nonzero(held)[0], np.nonzero(~held)[0]
        t0 = np.mean([r["seconds"] for r in T0 if dom(r) == d])
        t1 = np.mean([r["seconds"] for r in T1 if dom(r) == d])
        m1 = np.mean([r["sec_model"] for r in T1 if dom(r) == d])
        Hx = np.stack([np.frombuffer(base64.b64decode(r["hidden"]), np.float16).astype(np.float32) for r in recs])
        X, fid, cor, cost = tensors(recs)
        L = fid[:, 0]
        z = [ZE[recs[i]["qid"]] for i in te]
        zc = np.array([r["correct"] for r in z])
        zs = np.array([r["seconds"] for r in z])
        zt = np.array([r["vis_tokens_total"] for r in z])
        zp = np.array([r["passes"] for r in z])
        cr = [crop[recs[i]["qid"]]["steps"][-1] for i in te]
        per_q = {"fixed 1,024": cor[te, 0], "fixed 4,096": cor[te, 1], "ViCrop": np.array([s["correct"] for s in cr]),
                 "Zoom Eye": zc}
        out = {"n": int(len(te)),
               "fixed 1,024": dict(acc=cor[te, 0].mean(), tokens=cost[te, 0].mean(), seconds=t0),
               "fixed 4,096": dict(acc=cor[te, 1].mean(), tokens=(cost[te, 1] - cost[te, 0]).mean(), seconds=t1),
               "ViCrop": dict(acc=per_q["ViCrop"].mean(), tokens=np.mean([s["vis_tokens"] for s in cr])),
               "Zoom Eye": dict(acc=zc.mean(), tokens=zt.mean(), tokens_p50=np.median(zt), passes=zp.mean(),
                                one_pass=(zp == 1).mean(), seconds=zs.mean(), seconds_p50=np.median(zs),
                                seconds_p90=np.percentile(zs, 90))}
        for e in a.eps:
            accs, toks, escs, corq = [], [], [], np.zeros(len(te))
            for _ in range(a.reps):
                p = rng.permutation(rest); tr, ca = p[:int(.4 * len(rest))], p[int(.4 * len(rest)):][:a.n_cal]
                mu, sd = Hx[tr].mean(0), Hx[tr].std(0) + 1e-6
                _, _, Vt = np.linalg.svd((Hx[tr] - mu) / sd, full_matrices=False)
                Z = ((Hx - mu) / sd) @ Vt[:64].T
                s = fit_logreg(np.hstack([Z[tr], X[tr, 0]]), (L[tr] == 0).astype(float), l2=1e-1)(np.hstack([Z, X[:, 0]]))
                lam = calibrate(np.stack([s[ca], np.ones(len(ca))], 1), np.stack([L[ca], np.zeros(len(ca))], 1),
                                cost[ca], eps=e, delta=0.1).lam
                stop = s[te] >= lam
                c = np.where(stop, cor[te, 0], cor[te, 1])
                accs.append(c.mean()); corq += c / a.reps
                toks.append(np.where(stop, cost[te, 0], cost[te, 1]).mean()); escs.append(1 - stop.mean())
            out[f"RAVEN eps={e}"] = dict(acc=np.mean(accs), tokens=np.mean(toks), seconds=t0 + np.mean(escs) * m1,
                                         escalated=np.mean(escs))
            per_q[f"RAVEN eps={e}"] = corq
        out["Zoom Eye minus"] = {k: boot_ci(zc - v, rng) for k, v in per_q.items() if k != "Zoom Eye"}
        for k, v in per_q.items():
            pooled.setdefault(k, []).append(v)
        res[d] = out
        print(f"\n[{d}] {len(te)} questions")
        for k, v in out.items():
            if k == "Zoom Eye minus":
                print("  Zoom Eye accuracy minus: " + ", ".join(f"{kk} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                                                              for kk, (m, lo, hi) in v.items()))
            elif isinstance(v, dict):
                print(f"  {k:14s} " + "  ".join(f"{kk} {vv:.3f}" if vv < 100 else f"{kk} {vv:.0f}"
                                                for kk, vv in v.items()))
    zc = np.concatenate(pooled["Zoom Eye"])
    res["all"] = {"Zoom Eye minus": {k: boot_ci(zc - np.concatenate(v), rng) for k, v in pooled.items()
                                     if k != "Zoom Eye"},
                  "acc": {k: float(np.concatenate(v).mean()) for k, v in pooled.items()}}
    print("\n[all] accuracy " + ", ".join(f"{k} {v:.3f}" for k, v in res["all"]["acc"].items()))
    print("  Zoom Eye minus: " + ", ".join(f"{k} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                                          for k, (m, lo, hi) in res["all"]["Zoom Eye minus"].items()))
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

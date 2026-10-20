"""Two-rung ladder (e.g. 1,024 -> 4,096 tokens): learned sufficiency probe on the step-0 hidden state vs
hand-crafted confidence features vs random routing, all with the same LTT calibration on the eval set.
The ladder logs must carry "hidden" (run_gonogo.py --save_hidden on the first rung, then make_ladder.py).
Usage: python probe.py --train logs/hrbench4k_7b_ladder2h.jsonl logs/hrbench8k_7b_ladder2h.jsonl --eval logs/vstar_7b_ladder2h.jsonl"""
import argparse, base64, json, numpy as np
from analyze_gonogo import load, tensors, fit_logreg, evaluate
from ltt_calibrate import calibrate


def grouped(args, recs=None, label=""):
    """Train / calibrate / test on one pool, split 40/30/30 by group; probe refitted on every split."""
    recs = load(args.train) if recs is None else recs
    gmap = json.load(open(args.groups)) if args.groups else {}
    g = np.array([gmap.get(r["qid"], r["qid"]) for r in recs])
    ug = np.unique(g)
    Hx = np.stack([np.frombuffer(base64.b64decode(r["hidden"]), np.float16).astype(np.float32) for r in recs])
    X, fid, cor, cost = tensors(recs)
    n = len(recs)
    print(f"{label}pool {n} questions in {len(ug)} groups   acc rung0 {cor[:,0].mean():.3f}  rung1 {cor[:,1].mean():.3f}  oracle {cor.max(1).mean():.3f}")
    rng = np.random.default_rng(0)
    for lname in ("fidelity", "acc-gap"):
        L = fid if lname == "fidelity" else np.clip(cor[:, -1:] - cor, 0, None)
        out = {k: {e: [] for e in args.eps} for k in ("probe (hidden + conf)", "confidence features", "random routing")}
        viol = {k: {e: 0 for e in args.eps} for k in out}
        for _ in range(args.reps):
            pg = rng.permutation(ug); a, b = int(.4 * len(ug)), int(.7 * len(ug))
            tr, ca, te = (np.where(np.isin(g, pg[s]))[0] for s in (slice(0, a), slice(a, b), slice(b, None)))
            mu, sd = Hx[tr].mean(0), Hx[tr].std(0) + 1e-6
            _, _, Vt = np.linalg.svd((Hx[tr] - mu) / sd, full_matrices=False)
            Z = ((Hx - mu) / sd) @ Vt[:args.pca].T
            y = (L[tr, 0] == 0).astype(float)
            s0 = {"probe (hidden + conf)": fit_logreg(np.hstack([Z[tr], X[tr, 0]]), y, l2=1e-1)(np.hstack([Z, X[:, 0]])),
                  "confidence features": fit_logreg(X[tr, 0], y)(X[:, 0]),
                  "random routing": rng.random(n)}
            for k, sc in s0.items():
                S = np.stack([sc, np.ones(n)], 1)
                for e in args.eps:
                    c = calibrate(S[ca], L[ca], cost[ca], eps=e, delta=0.1, groups=g[ca])
                    r = evaluate(S[te], L[te], cor[te], cost[te], c.lam)
                    out[k][e].append(r); viol[k][e] += r["risk"] > e
        for k in out:
            for e in args.eps:
                m = {q: np.mean([r[q] for r in out[k][e]]) for q in out[k][e][0]}
                print(f"{lname:8s} {k:22s} eps={e:.2f}: risk {m['risk']:.3f} viol {viol[k][e]:3d}/{args.reps}  acc {m['acc']:.3f}  "
                      f"stopped early {m['crops_saved']:.1%}  avg tokens {cost[:, 1].mean() * (1 - m['tokens_saved']):.0f}")


ap = argparse.ArgumentParser()
ap.add_argument("--train", nargs="+", required=True)
ap.add_argument("--eval", nargs="+", help="separate eval logs; omit to use grouped splits of --train")
ap.add_argument("--groups", help="JSON {qid: group}; with no --eval, splits keep each group in one part")
ap.add_argument("--reps", type=int, default=100)
ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10])
ap.add_argument("--pca", type=int, default=64)
ap.add_argument("--by_domain", action="store_true", help="calibrate and test each domain (category prefix) separately")
args = ap.parse_args()

if args.eval is None and args.by_domain:
    allr = load(args.train)
    for d in sorted({r["category"].split("/")[0] for r in allr}):
        grouped(args, [r for r in allr if r["category"].split("/")[0] == d], label=f"[{d}] ")
    raise SystemExit
if args.eval is None:
    grouped(args)
    raise SystemExit
ra, rb = load(args.train), load(args.eval)
H = lambda recs: np.stack([np.frombuffer(base64.b64decode(r["hidden"]), np.float16).astype(np.float32) for r in recs])
Ha, Hb = H(ra), H(rb)
Xa, fa, ca, _ = tensors(ra)
Xb, fb, cb, kb = tensors(rb)
n, T1, d = Xb.shape
assert T1 == 2, "probe.py expects a two-rung ladder"
mu, sd = Ha.mean(0), Ha.std(0) + 1e-6
_, _, Vt = np.linalg.svd((Ha - mu) / sd, full_matrices=False)
P = lambda Hx: ((Hx - mu) / sd) @ Vt[:args.pca].T
print(f"train {len(ra)}  eval {n}   eval acc rung0 {cb[:,0].mean():.3f}  rung1 {cb[:,1].mean():.3f}  oracle {cb.max(1).mean():.3f}  "
      f"tokens rung0 {kb[:,0].mean():.0f}  both {kb[:,1].mean():.0f}")
rng = np.random.default_rng(0)
for lname in ("fidelity", "acc-gap"):
    La = fa if lname == "fidelity" else np.clip(ca[:, -1:] - ca, 0, None)
    Lb = fb if lname == "fidelity" else np.clip(cb[:, -1:] - cb, 0, None)
    ya = (La[:, 0] == 0).astype(float)
    scores = {
        "probe (hidden + conf)": fit_logreg(np.hstack([P(Ha), Xa[:, 0]]), ya, l2=1e-1)(np.hstack([P(Hb), Xb[:, 0]])),
        "confidence features": fit_logreg(Xa[:, 0], ya)(Xb[:, 0]),
        "random routing": rng.random(n),
    }
    for name, s0 in scores.items():
        S = np.stack([s0, np.ones(n)], 1)
        for eps in args.eps:
            res, viol = [], 0
            for _ in range(100):
                p = rng.permutation(n); cal, te = p[:n // 2], p[n // 2:]
                c = calibrate(S[cal], Lb[cal], kb[cal], eps=eps, delta=0.1)
                r = evaluate(S[te], Lb[te], cb[te], kb[te], c.lam); res.append(r); viol += r["risk"] > eps
            m = {k: np.mean([r[k] for r in res]) for k in res[0]}
            tok = kb[:, 1].mean() * (1 - m["tokens_saved"])
            print(f"{lname:8s} {name:22s} eps={eps:.2f}: risk {m['risk']:.3f} viol {viol:3d}/100  acc {m['acc']:.3f}  "
                  f"stopped early {m['crops_saved']:.1%}  avg tokens {tok:.0f}")

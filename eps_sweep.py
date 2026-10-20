"""Per-split sweep over eps for the MME-RealWorld two-rung ladders (validity and frontier figures).
Same protocol as probe.py grouped(): per domain, 40/30/30 splits, 100 reps, LTT-HB, delta=0.1."""
import argparse, os, sys, json, base64, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gonogo import tensors, fit_logreg, evaluate
from ltt_calibrate import calibrate
from multiprocessing import Pool

EPS = [0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]

def run(args):
    ladder, dom, logs = args
    recs = [json.loads(l) for l in open(os.path.join(logs, f"mmerw_7b_{ladder}.jsonl")) if l.strip()]
    recs = [r for r in recs if r["category"].split("/")[0] == dom]
    for r in recs:
        if isinstance(r["steps"], str):
            import ast; r["steps"] = ast.literal_eval(r["steps"])
    g = np.array([r["qid"] for r in recs]); ug = np.unique(g)
    Hx = np.stack([np.frombuffer(base64.b64decode(r["hidden"]), np.float16).astype(np.float32) for r in recs])
    X, fid, cor, cost = tensors(recs); n = len(recs)
    rng = np.random.default_rng(0)
    out = {"ladder": ladder, "domain": dom, "n": n, "acc0": cor[:,0].mean(), "acc1": cor[:,1].mean(),
           "tok0": cost[:,0].mean(), "tok1": cost[:,1].mean(), "fixed_tokens": None, "res": {}}
    for lname in ("fidelity", "acc-gap"):
        L = fid if lname == "fidelity" else np.clip(cor[:, -1:] - cor, 0, None)
        for _ in range(100):
            pg = rng.permutation(ug); a, b = int(.4*len(ug)), int(.7*len(ug))
            tr, ca, te = (np.where(np.isin(g, pg[s]))[0] for s in (slice(0,a), slice(a,b), slice(b,None)))
            mu, sd = Hx[tr].mean(0), Hx[tr].std(0) + 1e-6
            A = (Hx[tr]-mu)/sd
            ev, U = np.linalg.eigh(A @ A.T)              # Gram trick: same top PCs as SVD, up to sign
            idx = np.argsort(ev)[::-1][:64]
            V = (A.T @ U[:, idx]) / np.sqrt(np.maximum(ev[idx], 1e-12))
            Z = ((Hx-mu)/sd) @ V
            y = (L[tr,0] == 0).astype(float)
            s0 = {"probe": fit_logreg(np.hstack([Z[tr], X[tr,0]]), y, l2=1e-1)(np.hstack([Z, X[:,0]])),
                  "confidence": fit_logreg(X[tr,0], y)(X[:,0]),
                  "random": rng.random(n)}
            for k, sc in s0.items():
                S = np.stack([sc, np.ones(n)], 1)
                for e in EPS:
                    c = calibrate(S[ca], L[ca], cost[ca], eps=e, delta=0.1, groups=g[ca])
                    r = evaluate(S[te], L[te], cor[te], cost[te], c.lam)
                    # fixed-4096 baseline on the same test split
                    r["fixed_acc"] = cor[te, 1].mean()
                    r["fixed_tok"] = cost[te, 1].mean() - cost[te, 0].mean()  # 4096 view alone
                    r["tok"] = cost[te, 1].mean() * (1 - r["tokens_saved"])
                    out["res"].setdefault(f"{lname}|{k}|{e}", []).append({q: float(v) for q, v in r.items()})
    print(ladder, dom, "done", flush=True)
    return out

if __name__ == "__main__":
    doms = ["Autonomous_Driving", "Diagram and Table", "Monitoring", "OCR with Complex Context", "Remote Sensing"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs", help="folder with mmerw_7b_l1024.jsonl and mmerw_7b_l256.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/sweep.json")
    a = ap.parse_args()
    jobs = [(l, d, a.logs) for l in ("l1024", "l256") for d in doms]
    with Pool(a.workers) as p:
        res = p.map(run, jobs)
    json.dump(res, open(a.out, "w"))

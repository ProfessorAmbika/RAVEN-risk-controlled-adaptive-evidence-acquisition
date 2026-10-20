"""MME-RealWorld (Zhang et al., ICLR 2025) loader for run_gonogo.py.

  python mmerw.py --inspect            # file list, total size, columns of one shard (downloads one shard only)
  python mmerw.py --plan --n 1000      # downloads all shards, prints the per-domain sample it will use

Column names differ between releases, so they are resolved at run time and printed; check them once with --inspect.
The sample is stratified: up to n questions per domain, spread evenly over that domain's sub-tasks, fixed seed.
"""
import argparse, base64, glob, io, os, re

import numpy as np

QUESTION = ["question", "Text", "text"]
OPTIONS = ["multi-choice options", "Answer choices", "options", "choices"]
ANSWER = ["answer", "Ground truth", "gt"]
IMAGE = ["bytes", "image", "Image"]
QID = ["question_id", "Question_id", "index", "id"]
DOMAIN = ["subtask", "Subtask", "domain", "category", "Category", "l2-category"]
SUBTASK = ["l2-category", "category", "Category", "task", "Task"]


def pick(cols, names, what):
    for n in names:
        if n in cols:
            return n
    raise KeyError(f"no {what} column among {names}; columns are {list(cols)}")


def shards(repo):
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=["*.parquet"])
    return sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))


def meta_columns(files):
    import pyarrow.parquet as pq
    cols = pq.read_schema(files[0]).names
    m = {k: pick(cols, v, k) for k, v in
         [("question", QUESTION), ("options", OPTIONS), ("answer", ANSWER), ("image", IMAGE), ("qid", QID)]}
    # domain = the coarsest label with a handful of values; subtask = a finer one, if present
    import pandas as pd
    lab = [c for c in dict.fromkeys(DOMAIN + SUBTASK) if c in cols]
    small = pd.concat([pq.read_table(f, columns=lab).to_pandas() for f in files], ignore_index=True)
    nuniq = {c: small[c].nunique() for c in small.columns}
    m["domain"] = next((c for c in DOMAIN if c in nuniq and 3 <= nuniq[c] <= 12), None)
    m["subtask"] = next((c for c in SUBTASK if c in nuniq and c != m["domain"]), None)
    return m, nuniq


def plan(files, m, per_domain, seed=0):
    """Return {(file, row)} chosen: up to per_domain per domain, balanced over sub-tasks."""
    import pandas as pd
    import pyarrow.parquet as pq
    keep = [c for c in (m["domain"], m["subtask"]) if c]
    parts = []
    for f in files:
        t = pq.read_table(f, columns=keep).to_pandas()
        t["_file"], t["_row"] = f, np.arange(len(t))
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    # "Perception/Remote Sensing" and "Reasoning/Remote Sensing" are one domain; the task prefix becomes part of the sub-task
    df["_dom"] = df[m["domain"]].astype(str).str.split("/").str[-1] if m["domain"] else "all"
    df["_sub"] = df[keep].astype(str).agg("|".join, axis=1) if keep else "all"
    rng = np.random.default_rng(seed)
    chosen = []
    for d, dd in df.groupby("_dom"):
        groups = [g for _, g in dd.groupby("_sub")]
        quota, left = per_domain, len(groups)
        for g in sorted(groups, key=len):  # small sub-tasks first, leftover quota flows to larger ones
            k = min(len(g), quota // left)
            chosen.append(g.iloc[rng.permutation(len(g))[:k]])
            quota, left = quota - k, left - 1
    return pd.concat(chosen).sort_values(["_file", "_row"]).reset_index(drop=True)


def to_image(x):
    from PIL import Image
    if isinstance(x, dict):
        x = x.get("bytes")
    if isinstance(x, str):
        x = base64.b64decode(x)
    return Image.open(io.BytesIO(x)).convert("RGB")


def fmt_options(opts):
    opts = list(opts) if not isinstance(opts, str) else [o for o in re.split(r"\n|(?=\([A-E]\))", opts) if o.strip()]
    out = []
    for i, o in enumerate(opts):
        o = str(o).strip()
        out.append(o if re.match(r"^\(?[A-E][\).:]", o) else f"({'ABCDE'[i]}) {o}")
    return "\n".join(out)


def take_rows(f, idx):
    """Rows idx (sorted) of one shard, read a row group at a time: a whole shard's base64 column exceeds the 2 GB
    offset limit of a pyarrow string array, so Table.take over the full shard fails with 'offset overflow'."""
    import pandas as pd
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(f)
    out, start = [], 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        local = idx[(idx >= start) & (idx < start + n)] - start
        if len(local):
            out.append(pf.read_row_group(g).to_pandas().iloc[local])
        start += n
    return pd.concat(out, ignore_index=True)


def load_mmerw(repo, per_domain=1000, seed=0):
    import pyarrow.parquet as pq
    files = shards(repo)
    m, _ = meta_columns(files)
    print(f"MME-RealWorld: {len(files)} shards, columns {m}", flush=True)
    sel = plan(files, m, per_domain, seed)
    print(f"MME-RealWorld: {len(sel)} questions selected", flush=True)
    for f, rows in sel.groupby("_file", sort=True):
        t = take_rows(f, rows["_row"].to_numpy())  # only the chosen rows reach pandas
        for i, (_, s) in enumerate(rows.iterrows()):
            r = t.iloc[i]
            dom, sub = s["_dom"], s["_sub"]
            yield {
                "qid": f"mmerw_{r[m['qid']]}",
                "image": lambda b=r[m["image"]]: to_image(b),
                "question": f"{str(r[m['question']]).strip()}\nThe choices are listed below:\n{fmt_options(r[m['options']])}\n"
                            "Select the best answer to the above multiple-choice question based on the image. "
                            "Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:",
                "answers": [str(r[m["answer"]]).strip().strip("()").upper()[:1]],
                "kind": "mc",
                "category": f"{dom}/{sub}",
            }
        del t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="yifanzhang114/MME-RealWorld-lmms-eval")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    if a.inspect:
        from huggingface_hub import HfApi, hf_hub_download
        info = HfApi().dataset_info(a.repo, files_metadata=True)
        fs = [(s.rfilename, s.size or 0) for s in info.siblings]
        for name, size in fs:
            print(f"{size / 2**30:7.2f} GB  {name}")
        print(f"total {sum(s for _, s in fs) / 2**30:.1f} GB in {len(fs)} files")
        first = next(n for n, _ in fs if n.endswith(".parquet"))
        import pyarrow.parquet as pq
        p = hf_hub_download(a.repo, first, repo_type="dataset")
        print("columns:", pq.read_schema(p).names)
        t = pq.read_table(p).to_pandas().head(2)
        for c in t.columns:
            v = t[c].iloc[0]
            print(f"  {c}: {type(v).__name__} {str(v)[:120]!r}")
    if a.plan:
        files = shards(a.repo)
        m, nuniq = meta_columns(files)
        print("columns:", m, "unique values:", nuniq)
        sel = plan(files, m, a.n)
        print(sel.groupby("_dom").size().to_string())
        print(sel.groupby(["_dom", "_sub"]).size().to_string())
        print("total", len(sel))

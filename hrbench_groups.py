"""Map every HR-Bench log id (hrbench4k_<index>, hrbench8k_<index>) to a group = its normalised question text.
4K images are crops of the 8K images and each question appears several times (rotated options), so all
train / calibration / test splits must keep a group together.
Usage: python hrbench_groups.py --out logs/hrbench_groups.json"""
import argparse, glob, json, os
import pandas as pd
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
groups = {}
for v in ("4k", "8k"):
    root = snapshot_download("DreamMr/HR-Bench", repo_type="dataset", allow_patterns=[f"*{v}*.parquet"])
    for f in glob.glob(os.path.join(root, "**", f"*{v}*.parquet"), recursive=True):
        df = pd.read_parquet(f, columns=["index", "question"])
        for i, q in zip(df["index"], df["question"]):
            groups[f"hrbench{v}_{i}"] = " ".join(str(q).lower().split())
json.dump(groups, open(a.out, "w"))
print(f"wrote {a.out}: {len(groups)} ids in {len(set(groups.values()))} groups")

"""Pairwise interaction features on a pruned tag: for the top-K features (list order = importance),
all pairs -> x*y and x/(y+1). Writes FEATURES_DIR/<out_tag>/anchor_*.parquet (user_id + pair cols).
Usage: python build_pair_feats.py --tag v2p77 --key p77 --top-k 20 --out-tag v2_pairs
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, polars as pl
from common import FEATURES_DIR, TASK_DIR

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2p77"); ap.add_argument("--key", default="p77")
    ap.add_argument("--top-k", type=int, default=20); ap.add_argument("--out-tag", default="v2_pairs")
    ap.add_argument("--list", default=str(TASK_DIR / "artifacts" / "feat_prune_lists.json"))
    a = ap.parse_args()
    top = json.loads(open(a.list).read())[a.key][:a.top_k]
    out = FEATURES_DIR / a.out_tag; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for f in sorted((FEATURES_DIR / a.tag).glob("anchor_*.parquet")):
        df = pl.read_parquet(f, columns=["user_id"] + top)
        X = df.select(top).to_numpy().astype(np.float32); X = np.nan_to_num(X, nan=0.0)
        cols, names = [], []
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                cols.append(X[:, i] * X[:, j]); names.append(f"px_{top[i]}__{top[j]}")
                cols.append(X[:, i] / (np.abs(X[:, j]) + 1.0)); names.append(f"pr_{top[i]}__{top[j]}")
        pl.DataFrame({"user_id": df["user_id"], **{n: c for n, c in zip(names, cols)}}).write_parquet(out / f.name)
    print(f"{a.out_tag}: {len(names)} pair features from top-{a.top_k} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()

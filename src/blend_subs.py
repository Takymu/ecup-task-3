"""Log-space blend of existing submission CSVs into a new submission.

Usage: python blend_subs.py --name sub_v8 --spec "sub_v2_blend:0.5,sub_v5_dartfull_seq_x110:0.5" [--mult 1.0]
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from common import SUBMISSIONS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--mult", type=float, default=1.0)
    args = ap.parse_args()

    specs = []
    for part in args.spec.split(","):
        stem, w = part.strip().rsplit(":", 1)
        specs.append((stem, float(w)))
    wsum = sum(w for _, w in specs)

    acc, users = None, None
    for stem, w in specs:
        df = pl.read_csv(SUBMISSIONS_DIR / f"{stem}.csv").sort("user_id")
        lp = np.log1p(np.clip(df["predict"].to_numpy(), 0, None))
        print(f"  {stem} w={w:.2f} mean_lp={lp.mean():.4f}")
        if acc is None:
            users, acc = df["user_id"], (w / wsum) * lp
        else:
            acc += (w / wsum) * lp

    pred = np.clip(np.expm1(acc), 0, None) * args.mult
    out = SUBMISSIONS_DIR / f"{args.name}.csv"
    pl.DataFrame({"user_id": users, "predict": pred}).write_csv(out)
    print(f"saved {out}  mean_lp={np.log1p(pred).mean():.4f}")


if __name__ == "__main__":
    main()

"""Blend precomputed {stem}_testpred.parquet files (raw preds) into a submission CSV.

Usage:
  python blend_testpreds.py --name sub_v4_quick --spec "lgbm_srv_dart:0.55,srv_seq:0.45" [--mult 1.0]
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from common import MODELS_DIR, SAMPLE_SUBMIT, SUBMISSIONS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--spec", required=True, help="stem:weight,stem:weight,...")
    ap.add_argument("--mult", type=float, default=1.0)
    args = ap.parse_args()

    specs = []
    for part in args.spec.split(","):
        stem, w = part.strip().rsplit(":", 1)
        specs.append((stem, float(w)))
    wsum = sum(w for _, w in specs)

    acc = None
    for stem, w in specs:
        df = pl.read_parquet(MODELS_DIR / f"{stem}_testpred.parquet").sort("user_id")
        lp = np.log1p(np.clip(df["pred"].to_numpy(), 0, None))
        print(f"  {stem} w={w:.2f}  mean_lp={lp.mean():.4f}  n={len(lp)}")
        if acc is None:
            users, acc = df["user_id"].to_numpy(), (w / wsum) * lp
        else:
            assert len(lp) == len(acc), "testpred row-count mismatch"
            acc += (w / wsum) * lp

    pred = np.clip(np.expm1(acc), 0, None) * args.mult
    sub = pl.DataFrame({"user_id": users, "predict": pred})
    sample = pl.read_csv(SAMPLE_SUBMIT)
    sub = sample.select("user_id").join(sub, on="user_id", how="left")
    assert sub["predict"].null_count() == 0, "users missing from testpreds"
    out = SUBMISSIONS_DIR / f"{args.name}.csv"
    sub.write_csv(out)
    print(f"saved {out}")
    print(f"pred stats: mean={pred.mean():.2f} p50={np.median(pred):.2f} p99={np.percentile(pred, 99):.2f} "
          f"max={pred.max():.2f} ~zeros={(pred < 0.5).mean():.3f}")


if __name__ == "__main__":
    main()

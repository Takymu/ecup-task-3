"""E18: per-user 'residual trait' — how the user historically deviates from the naive
autoregressive forecast (0.3 x gmv_30d). Computed from PAST anchor windows only (no leakage).

For anchor a and user u:
  trait_mean = mean over past anchors p <= a-30 of [log1p(target_p) - log1p(0.3*gmv30_p)]
  trait_std, trait_n similarly.

Reads v2 anchor files, writes artifacts/features/<tag>_trait/anchor_<a>.parquet.
Usage: python build_trait.py --tag v2
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import numpy as np
import polars as pl

from common import FEATURES_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    src = FEATURES_DIR / args.tag
    out_dir = FEATURES_DIR / f"{args.tag}_trait"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("anchor_*.parquet"))
    # residuals per anchor (only anchors with real targets)
    resid = {}
    anchors = []
    for f in files:
        d = pl.read_parquet(f, columns=["user_id", "anchor_date", "gmv_sum_30d", "target"])
        a = d["anchor_date"][0]
        anchors.append((a, f))
        if d["target"].null_count() == 0:
            r = (
                np.log1p(np.clip(d["target"].to_numpy(), 0, None))
                - np.log1p(0.3 * np.clip(d["gmv_sum_30d"].to_numpy(), 0, None))
            )
            resid[a] = d.select("user_id").with_columns(pl.Series("r", r))

    for a, f in anchors:
        past = [resid[p] for p in resid if p <= a - timedelta(days=30)]
        base = pl.read_parquet(f, columns=["user_id"])
        if past:
            allr = pl.concat(past)
            tr = allr.group_by("user_id").agg(
                pl.col("r").mean().alias("trait_mean"),
                pl.col("r").std().alias("trait_std"),
                pl.len().alias("trait_n"),
            )
            out = base.join(tr, on="user_id", how="left")
        else:
            out = base.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("trait_mean"),
                pl.lit(None, dtype=pl.Float64).alias("trait_std"),
                pl.lit(None, dtype=pl.UInt32).alias("trait_n"),
            )
        out.write_parquet(out_dir / f.name)
        print(f"  {a}: past_anchors={len(past)}", flush=True)


if __name__ == "__main__":
    main()

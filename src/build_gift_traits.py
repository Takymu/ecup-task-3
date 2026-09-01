"""Gift-season user traits from spring 2025 (constant per user, earlier than every anchor).

Windows (all in history for any anchor >= 2025-04-01, so leak-free as constants):
  pre-Valentine  2025-02-07..02-13, pre-Feb23  2025-02-09..02-22, pre-Mar8  2025-02-24..03-07.
Per window: gmv, active days, searches. Plus gender proxy = log1p(gmv_m8) - log1p(gmv_f23)
(shopping BEFORE Feb 23 = gifts for men -> buyer likely female, and vice versa).

Writes identical extras parquet per anchor: FEATURES_DIR/v2_gift/anchor_*.parquet
Usage: python build_gift_traits.py [--tag v2 --out-tag v2_gift]
"""
from __future__ import annotations

import argparse
import time
from datetime import date

import numpy as np
import polars as pl

from common import FEATURES_DIR, TRAIN_PARQUET

WINDOWS = {
    "val14": (date(2025, 2, 7), date(2025, 2, 13)),
    "f23": (date(2025, 2, 9), date(2025, 2, 22)),
    "m8": (date(2025, 2, 24), date(2025, 3, 7)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out-tag", default="v2_gift")
    args = ap.parse_args()

    t0 = time.time()
    df = pl.scan_parquet(TRAIN_PARQUET).select(
        ["event_date", "user_id", "gmv", "searches"])
    aggs = []
    for w, (lo, hi) in WINDOWS.items():
        m = (pl.col("event_date") >= lo) & (pl.col("event_date") <= hi)
        aggs += [
            pl.col("gmv").filter(m).sum().alias(f"gift_gmv_{w}"),
            pl.col("gmv").filter(m).count().alias(f"gift_days_{w}"),
            pl.col("searches").filter(m).sum().alias(f"gift_srch_{w}"),
        ]
    tr = df.group_by("user_id").agg(aggs).collect()
    tr = tr.with_columns([
        (pl.col("gift_gmv_m8").log1p() - pl.col("gift_gmv_f23").log1p()).alias("gender_proxy_gmv"),
        (pl.col("gift_srch_m8").log1p() - pl.col("gift_srch_f23").log1p()).alias("gender_proxy_srch"),
    ]).with_columns(pl.exclude("user_id").fill_null(0.0))
    print(f"traits for {len(tr):,} users in {time.time()-t0:.0f}s", flush=True)

    # align to the full 250k user set of the feature files
    out_dir = FEATURES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((FEATURES_DIR / args.tag).glob("anchor_*.parquet"))
    base_users = pl.read_parquet(files[0], columns=["user_id"]).sort("user_id")
    full = base_users.join(tr, on="user_id", how="left").with_columns(
        pl.exclude("user_id").fill_null(0.0))
    for f in files:
        full.write_parquet(out_dir / f.name)
    print(f"saved {len(files)} extras files to {out_dir} ({full.width-1} trait cols)", flush=True)


if __name__ == "__main__":
    main()

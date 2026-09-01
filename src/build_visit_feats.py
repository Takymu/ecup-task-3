"""Visit-only day features (EDA insight #2): 15% of rows are days with no search and no
catalog usage ("visit-only" — app open / other surfaces). Base features see them only as
generic active days; here they get their own recency/frequency aggregates per anchor.

Writes extras parquet per anchor: FEATURES_DIR/v2_visit/anchor_*.parquet (user_id + 8 cols).
Usage: python build_visit_feats.py [--tag v2 --out-tag v2_visit]
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import polars as pl

from common import FEATURES_DIR, TRAIN_PARQUET

WINDOWS = (7, 30, 90, 365)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out-tag", default="v2_visit")
    args = ap.parse_args()

    t0 = time.time()
    df = pl.scan_parquet(TRAIN_PARQUET).select(["event_date", "user_id", "search", "cat", "to_cart"]).with_columns(
        ((pl.col("search") == 0) & (pl.col("cat") == 0)).alias("visit_only")
    ).collect()
    files = sorted((FEATURES_DIR / args.tag).glob("anchor_*.parquet"))
    base_users = pl.read_parquet(files[0], columns=["user_id"]).sort("user_id")
    out_dir = FEATURES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        a = date.fromisoformat(f.stem.removeprefix("anchor_"))
        h = df.filter((pl.col("event_date") <= a) & (pl.col("event_date") > a - timedelta(days=365)))
        h = h.with_columns((pl.lit(a) - pl.col("event_date")).dt.total_days().alias("age"))
        aggs = []
        for w in WINDOWS:
            m = pl.col("age") < w
            aggs.append((pl.col("visit_only") & m).sum().alias(f"visit_only_{w}d"))
        aggs += [
            (pl.col("age") < 90).sum().alias("_act_90d"),
            pl.col("age").filter(pl.col("visit_only")).min().alias("days_since_last_visit_only"),
            (pl.col("visit_only") & (pl.col("to_cart") > 0) & (pl.col("age") < 90)).sum().alias("visit_only_cart_90d"),
        ]
        g = h.group_by("user_id").agg(aggs).with_columns(
            (pl.col("visit_only_90d") / pl.col("_act_90d").clip(lower_bound=1)).alias("visit_only_share_90d")
        ).drop("_act_90d")
        full = base_users.join(g, on="user_id", how="left")
        full = full.with_columns([pl.col(c).fill_null(0) for c in full.columns
                                  if c not in ("user_id", "days_since_last_visit_only")])
        full.write_parquet(out_dir / f.name)
        print(f"  {a}: {full.width-1} cols ({time.time()-t0:.0f}s)", flush=True)
    print(f"saved {len(files)} extras files to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

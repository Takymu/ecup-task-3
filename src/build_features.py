"""Build per-user features + targets at a set of anchor dates.

Usage:
  python build_features.py --tag v1 --anchors auto   # train anchors + val anchor
  python build_features.py --tag v1 --test           # test anchor (2026-02-13)

Output: artifacts/features/<tag>/anchor_YYYY-MM-DD.parquet
Each file: user_id, anchor_date, <features...>, target (null for test anchor).
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import polars as pl

from common import FEATURES_DIR, HORIZON_DAYS, TEST_ANCHOR, TRAIN_PARQUET, VAL_ANCHOR

# trailing windows in days, each ending at the anchor (inclusive)
WINDOWS = [7, 14, 30, 60, 90, 180, 365]
# windows for which we compute the "expensive" aggs (mean/max/std) beyond sum
RICH_WINDOWS = {30, 90}


def feature_exprs(anchor: date, value_cols: list[str]) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for w in WINDOWS:
        w_start = anchor - timedelta(days=w - 1)
        mask = pl.col("event_date") >= w_start
        # active days in window
        exprs.append(mask.sum().alias(f"days_active_{w}d"))
        for c in value_cols:
            exprs.append(pl.col(c).filter(mask).sum().alias(f"{c}_sum_{w}d"))
            if w in RICH_WINDOWS:
                exprs.append(pl.col(c).filter(mask).max().alias(f"{c}_max_{w}d"))
                exprs.append(pl.col(c).filter(mask).mean().alias(f"{c}_mean_{w}d"))
                exprs.append(pl.col(c).filter(mask).std().alias(f"{c}_std_{w}d"))
        # days with a purchase / with gmv in window
        exprs.append((mask & (pl.col("gmv") > 0)).sum().alias(f"gmv_days_{w}d"))
        exprs.append((mask & (pl.col("to_ord") > 0)).sum().alias(f"ord_days_{w}d"))

    anchor_lit = pl.lit(anchor)
    exprs += [
        (anchor_lit - pl.col("event_date").max()).dt.total_days().alias("days_since_last_event"),
        (anchor_lit - pl.col("event_date").min()).dt.total_days().alias("days_since_first_event"),
        (anchor_lit - pl.col("event_date").filter(pl.col("gmv") > 0).max())
        .dt.total_days().alias("days_since_last_gmv"),
        (anchor_lit - pl.col("event_date").filter(pl.col("to_ord") > 0).max())
        .dt.total_days().alias("days_since_last_ord"),
        (anchor_lit - pl.col("event_date").filter(pl.col("to_cart") > 0).max())
        .dt.total_days().alias("days_since_last_cart"),
        pl.len().alias("days_active_total"),
        (pl.col("gmv") > 0).sum().alias("gmv_days_total"),
        pl.col("gmv").sum().alias("gmv_sum_total"),
        pl.col("gmv").filter(pl.col("gmv") > 0).mean().alias("gmv_per_gmv_day"),
        pl.col("gmv").filter(pl.col("gmv") > 0).median().alias("gmv_median_pos"),
    ]
    return exprs


def derived_exprs() -> list[pl.Expr]:
    eps = 1e-6
    e: list[pl.Expr] = []
    for num, den in [(30, 90), (7, 30), (90, 365), (14, 60)]:
        e.append(
            (pl.col(f"gmv_sum_{num}d") / (pl.col(f"gmv_sum_{den}d") + eps)).alias(f"gmv_trend_{num}_{den}")
        )
        e.append(
            (pl.col(f"days_active_{num}d") / (pl.col(f"days_active_{den}d") + eps)).alias(
                f"act_trend_{num}_{den}"
            )
        )
    e.append((pl.col("gmv_sum_365d") / (pl.col("gmv_days_365d") + eps)).alias("aov_365d"))
    e.append((pl.col("gmv_sum_30d") / (pl.col("gmv_days_30d") + eps)).alias("aov_30d"))
    e.append(
        (pl.col("days_since_last_event") / (pl.col("days_since_first_event") + eps)).alias("recency_ratio")
    )
    return e


def gap_exprs(anchor: date) -> list[pl.Expr]:
    """Per-user purchase-gap stats over gmv>0 days (expects pre-sorted, pre-filtered frame)."""
    anchor_lit = pl.lit(anchor)
    return [
        pl.col("gap").median().alias("buy_gap_med"),
        pl.col("gap").mean().alias("buy_gap_mean"),
        pl.col("gap").std().alias("buy_gap_std"),
        pl.len().alias("n_buy_days_hist"),
        (anchor_lit - pl.col("event_date").max()).dt.total_days().alias("_since_last_buy"),
    ]


def weekly_slope_frame(hist: pl.DataFrame, anchor: date, weeks: int = 8) -> pl.DataFrame:
    """Slope of log1p(weekly gmv) and weekly activity over last `weeks` weeks."""
    w = hist.filter(pl.col("event_date") > anchor - timedelta(days=7 * weeks)).with_columns(
        ((pl.lit(anchor) - pl.col("event_date")).dt.total_days() // 7).alias("wk")
    )
    wk = w.group_by("user_id", "wk").agg(
        pl.col("gmv").sum().alias("g"),
        pl.len().alias("d"),
    )
    # x = -wk so positive slope = growth toward the anchor
    return wk.group_by("user_id").agg(
        pl.cov((-pl.col("wk")).cast(pl.Float64), pl.col("g").log1p()).alias("_cov_g"),
        (-pl.col("wk")).cast(pl.Float64).var().alias("_var_x"),
        pl.cov((-pl.col("wk")).cast(pl.Float64), pl.col("d").cast(pl.Float64)).alias("_cov_d"),
        pl.col("wk").n_unique().alias("n_weeks_active_8w"),
        (pl.col("g") > 0).sum().alias("n_weeks_gmv_8w"),
    ).with_columns(
        (pl.col("_cov_g") / (pl.col("_var_x") + 1e-9)).alias("gmv_slope_8w"),
        (pl.col("_cov_d") / (pl.col("_var_x") + 1e-9)).alias("act_slope_8w"),
    ).drop("_cov_g", "_cov_d", "_var_x")


def ly_window_frame(df: pl.DataFrame, anchor: date) -> pl.DataFrame:
    """Activity in the 364-day-shifted analog of the TARGET window [anchor+1, anchor+30]."""
    lo = anchor + timedelta(days=1 - 364)
    hi = anchor + timedelta(days=HORIZON_DAYS - 364)
    w = df.filter(pl.col("event_date").is_between(lo, hi))
    return w.group_by("user_id").agg(
        pl.col("gmv").sum().alias("ly_win_gmv"),
        (pl.col("gmv") > 0).sum().alias("ly_win_buy_days"),
        pl.len().alias("ly_win_act_days"),
        pl.col("searches").sum().alias("ly_win_searches"),
    )


def build_anchor(
    df: pl.DataFrame, all_users: pl.DataFrame, anchor: date, value_cols: list[str], with_target: bool
) -> pl.DataFrame:
    hist = df.filter(
        (pl.col("event_date") <= anchor)
        & (pl.col("event_date") >= anchor - timedelta(days=max(WINDOWS) - 1))
    )
    feats = hist.group_by("user_id").agg(feature_exprs(anchor, value_cols))
    out = all_users.join(feats, on="user_id", how="left")

    # v2: purchase-gap stats (full history <= anchor, not just 365d)
    buys = (
        df.filter((pl.col("gmv") > 0) & (pl.col("event_date") <= anchor))
        .select(["user_id", "event_date"])
        .sort(["user_id", "event_date"])
        .with_columns(pl.col("event_date").diff().over("user_id").dt.total_days().alias("gap"))
    )
    gaps = buys.group_by("user_id").agg(gap_exprs(anchor))
    gaps = gaps.with_columns(
        (pl.col("_since_last_buy") / (pl.col("buy_gap_med") + 1.0)).alias("overdue_ratio"),
        (pl.col("buy_gap_std") / (pl.col("buy_gap_med") + 1.0)).alias("buy_gap_cv"),
    ).drop("_since_last_buy")
    out = out.join(gaps, on="user_id", how="left")

    # v2: weekly slopes; v2: last-year target-window analog
    out = out.join(weekly_slope_frame(hist, anchor), on="user_id", how="left")
    ly_lo = anchor + timedelta(days=1 - 364)
    if ly_lo >= date(2025, 1, 1):
        out = out.join(ly_window_frame(df, anchor), on="user_id", how="left")
        out = out.with_columns(pl.col("ly_win_gmv").fill_null(0.0), pl.col("ly_win_buy_days").fill_null(0),
                               pl.col("ly_win_act_days").fill_null(0), pl.col("ly_win_searches").fill_null(0))
    else:
        out = out.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("ly_win_gmv"),
            pl.lit(None, dtype=pl.Int64).alias("ly_win_buy_days"),
            pl.lit(None, dtype=pl.Int64).alias("ly_win_act_days"),
            pl.lit(None, dtype=pl.Int64).alias("ly_win_searches"),
        )

    # v2: cart->order conversion recency
    out = out.with_columns(
        (pl.col("to_ord_sum_14d") / (pl.col("to_cart_sum_14d") + 1.0)).alias("cart2ord_14d"),
        (pl.col("to_ord_sum_60d") / (pl.col("to_cart_sum_60d") + 1.0)).alias("cart2ord_60d"),
        (pl.col("to_cart_sum_14d") - pl.col("to_ord_sum_14d")).alias("cart_pending_14d"),
    )

    # calendar context of the anchor
    out = out.with_columns(
        pl.lit(anchor).alias("anchor_date"),
        pl.lit(anchor.month).cast(pl.Int16).alias("anchor_month"),
        pl.lit(int(anchor.strftime("%j"))).cast(pl.Int16).alias("anchor_doy"),
    )
    out = out.with_columns(derived_exprs())

    if with_target:
        t_start = anchor + timedelta(days=1)
        t_end = anchor + timedelta(days=HORIZON_DAYS)
        tgt = (
            df.filter(pl.col("event_date").is_between(t_start, t_end))
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("target"))
        )
        out = out.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--n-anchors", type=int, default=20, help="train+val anchors, stride 14d back from VAL_ANCHOR")
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--test", action="store_true", help="build the test anchor instead of train anchors")
    args = ap.parse_args()

    out_dir = FEATURES_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading train.parquet ...", flush=True)
    df = pl.read_parquet(TRAIN_PARQUET)
    value_cols = [c for c in df.columns if c not in ("event_date", "user_id")]
    print(f"value cols ({len(value_cols)}): {value_cols}", flush=True)
    all_users = df.select(pl.col("user_id").unique().sort())

    if args.test:
        anchors = [(TEST_ANCHOR, False)]
    else:
        anchors = [
            (VAL_ANCHOR - timedelta(days=args.stride * i), True) for i in range(args.n_anchors)
        ][::-1]

    for anchor, with_target in anchors:
        path = out_dir / f"anchor_{anchor.isoformat()}.parquet"
        if path.exists():
            print(f"  {anchor}  exists, skip", flush=True)
            continue
        t0 = time.time()
        out = build_anchor(df, all_users, anchor, value_cols, with_target)
        out.write_parquet(path)
        print(f"  {anchor}  rows={out.height:,}  cols={out.width}  {time.time() - t0:.1f}s", flush=True)

    print("done")


if __name__ == "__main__":
    main()

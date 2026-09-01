"""Rhythm & repeat-amount features per anchor (user idea 15.08): approximate repeated purchase
amounts (10%-wide log bins) and purchase periodicity (modal gap, daily-buy autocorrelation).

History = 365 days before anchor. Writes FEATURES_DIR/v2_rhythm/anchor_*.parquet (user_id + cols).
Usage: python build_rhythm_feats.py [--tag v2 --out-tag v2_rhythm]
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import numpy as np
import polars as pl

from common import FEATURES_DIR, TRAIN_PARQUET

ACF_WIN = 182
ACF_LAGS = (7, 14, 28, 30)


def acf_feats(h: pl.DataFrame, users: np.ndarray, anchor: date) -> pl.DataFrame:
    """Autocorrelation of the daily buy indicator over the last ACF_WIN days."""
    uidx = {u: i for i, u in enumerate(users)}
    w = h.filter(pl.col("age") < ACF_WIN)
    X = np.zeros((len(users), ACF_WIN), dtype=np.float32)
    rows = np.fromiter((uidx[u] for u in w["user_id"].to_numpy()), dtype=np.int64, count=len(w))
    X[rows, w["age"].to_numpy().astype(np.int64)] = 1.0
    p = X.mean(1)
    var = p * (1 - p)
    out = {"user_id": users}
    best, best_lag = np.full(len(users), -1.0, dtype=np.float32), np.zeros(len(users), dtype=np.float32)
    for L in range(2, 36):
        m = (X[:, L:] * X[:, :-L]).mean(1)
        a = np.where(var > 0, (m - p * p) / np.where(var > 0, var, 1), np.nan)
        if L in ACF_LAGS:
            out[f"acf_buy_{L}"] = a
        upd = np.nan_to_num(a, nan=-1.0) > best
        best = np.where(upd, np.nan_to_num(a, nan=-1.0), best)
        best_lag = np.where(upd, L, best_lag)
    out["acf_buy_max"] = np.where(best > -1, best, np.nan)
    out["acf_buy_max_lag"] = np.where(best > -1, best_lag, np.nan)
    return pl.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out-tag", default="v2_rhythm")
    args = ap.parse_args()

    t0 = time.time()
    buys = pl.scan_parquet(TRAIN_PARQUET).filter(pl.col("gmv") > 0).select(
        ["event_date", "user_id", "gmv"]).collect().sort(["user_id", "event_date"])
    buys = buys.with_columns((pl.col("gmv").log1p() / 0.1).floor().alias("bin"))
    files = sorted((FEATURES_DIR / args.tag).glob("anchor_*.parquet"))
    base_users = pl.read_parquet(files[0], columns=["user_id"]).sort("user_id")
    users = base_users["user_id"].to_numpy()
    out_dir = FEATURES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        a = date.fromisoformat(f.stem.removeprefix("anchor_"))
        h = buys.filter((pl.col("event_date") <= a) & (pl.col("event_date") > a - timedelta(days=365)))
        h = h.with_columns((pl.lit(a) - pl.col("event_date")).dt.total_days().alias("age"))
        # --- repeat amounts: 10%-wide log bins ---
        c = h.group_by(["user_id", "bin"]).agg([pl.len().alias("cnt"), pl.col("age").min().alias("bin_last_age")])
        hb = h.join(c, on=["user_id", "bin"])
        rep = hb.group_by("user_id").agg([
            pl.len().alias("_n"),
            (pl.col("cnt") >= 2).mean().alias("rep_share_10pct"),
            pl.col("cnt").max().alias("rep_modal_cnt"),
            pl.col("bin").n_unique().alias("_nbins"),
        ]).with_columns([
            (pl.col("rep_modal_cnt") / pl.col("_n")).alias("rep_modal_share"),
            (pl.col("_nbins") / pl.col("_n")).alias("rep_bin_diversity"),
        ])
        modal = c.sort(["user_id", "cnt"], descending=[False, True]).group_by("user_id").agg([
            pl.col("bin").first().alias("_mbin"), pl.col("bin_last_age").first().alias("rep_modal_last_age")])
        modal = modal.with_columns((pl.col("_mbin") * 0.1 + 0.05).alias("rep_modal_log_gmv")).drop("_mbin")
        rep = rep.join(modal, on="user_id", how="left").drop(["_n", "_nbins"])
        # --- gaps ---
        gaps = h.with_columns(pl.col("event_date").diff().over("user_id").dt.total_days().alias("gap")).drop_nulls("gap")
        gm = gaps.group_by(["user_id", "gap"]).len().sort(["user_id", "len"], descending=[False, True]).group_by("user_id").agg([
            pl.col("gap").first().alias("gap_modal"), pl.col("len").first().alias("_mc"), pl.col("len").sum().alias("gap_n")])
        gm = gm.with_columns((pl.col("_mc") / pl.col("gap_n")).alias("gap_modal_share")).drop("_mc")
        gs = gaps.join(gm.select(["user_id", "gap_modal"]), on="user_id").group_by("user_id").agg([
            ((pl.col("gap") - pl.col("gap_modal")).abs() <= 2).mean().alias("gap_near_modal_share"),
            (pl.col("gap") <= 2).mean().alias("gap_bursty_share"),
            pl.col("gap").median().alias("gap_median"),
        ])
        last = h.group_by("user_id").agg(pl.col("age").min().alias("_last_age"))
        gm = gm.join(gs, on="user_id", how="left").join(last, on="user_id", how="left").with_columns([
            (pl.col("gap_modal") - pl.col("_last_age")).alias("gap_due_in_modal"),
            (pl.col("_last_age") / pl.col("gap_median").clip(lower_bound=1)).alias("gap_cycle_pos_median"),
        ]).drop("_last_age")
        # --- autocorrelation ---
        acf = acf_feats(h, users, a)
        full = base_users.join(rep, on="user_id", how="left").join(gm, on="user_id", how="left").join(acf, on="user_id", how="left")
        full.write_parquet(out_dir / f.name)
        print(f"  {a}: {full.width-1} cols ({time.time()-t0:.0f}s)", flush=True)
    print(f"saved {len(files)} extras files to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

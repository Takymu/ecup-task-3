"""Train LightGBM on log1p(target) across anchor files; validate on VAL_ANCHOR.

Usage:
  python train_lgbm.py --tag v1 [--gap-days 30] [--full]

--gap-days: exclude train anchors whose 30d target window overlaps the val window
--full: after validation, retrain on ALL anchors (incl. val) with best_iter and save as *_full
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta

import os

import lightgbm as lgb
import numpy as np
import polars as pl

from common import FEATURES_DIR, MODELS_DIR, VAL_ANCHOR, lean_load, rmsle

N_THREADS = int(os.environ.get("ML_THREADS", "14"))

DROP_COLS = ["user_id", "anchor_date", "target", "anchor_month", "anchor_doy"]  # anchor calendar = regime memorization, +0.005 val rank (15.08)


def load_anchors(tag: str) -> pl.DataFrame:
    files = sorted((FEATURES_DIR / tag).glob("anchor_*.parquet"))
    files = [f for f in files if f.stem != f"anchor_"]
    dfs = [pl.read_parquet(f) for f in files if "2026-02-13" not in f.stem]
    df = pl.concat(dfs, how="vertical_relaxed")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--max-rounds", type=int, default=5000)
    ap.add_argument("--min-data", type=int, default=200)
    ap.add_argument("--min-anchor-date", default=None, help="ISO date; exclude older train anchors")
    ap.add_argument("--drop-anchor-cal", action="store_true", help="(default now) drop anchor_month/anchor_doy")
    ap.add_argument("--keep-anchor-cal", action="store_true", help="keep anchor_month/anchor_doy (legacy 269-col runs)")
    ap.add_argument("--half-life", type=float, default=0.0, help="anchor age half-life in days for sample weights (0=off)")
    ap.add_argument("--name", default=None, help="model name suffix (default = tag)")
    ap.add_argument("--fast", action="store_true",
                    help="proxy protocol: recent anchors only, max_bin 63, lr 0.04 — for A/B deltas")
    ap.add_argument("--objective", default="regression",
                    help="regression | regression_l1 | tweedie (tweedie is on log1p target too)")
    ap.add_argument("--boosting", default="gbdt", help="gbdt | dart")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-anchor", default=None, help="ISO date; override VAL_ANCHOR (2nd validation)")
    ap.add_argument("--max-recency", type=int, default=0,
                    help="drop train rows with days_since_last_event > N (0=off); test users all have <=30")
    ap.add_argument("--extras", default=None,
                    help="features subdir (e.g. v2_trait) to join on anchor_date+user_id")
    args = ap.parse_args()
    if args.fast:
        args.lr = 0.04
        args.num_leaves = 96
        if args.min_anchor_date is None:
            args.min_anchor_date = "2025-09-01"

    t0 = time.time()
    from datetime import date as _date
    val_anchor = _date.fromisoformat(args.val_anchor) if args.val_anchor else VAL_ANCHOR
    train_cutoff = val_anchor - timedelta(days=args.gap_days)
    drop = list(DROP_COLS)
    if args.keep_anchor_cal:
        drop = [c for c in drop if c not in ("anchor_month", "anchor_doy")]
    data = lean_load(
        args.tag,
        cutoff=train_cutoff,
        min_anchor=_date.fromisoformat(args.min_anchor_date) if args.min_anchor_date else None,
        val_anchor=val_anchor,
        drop_cols=tuple(drop),
        extras_tag=args.extras,
    )
    feat_cols = data["feat_cols"]
    X_tr, y_tr_raw = data["X_tr"], data["y_tr_raw"]
    X_va, y_va_raw = data["X_va"], data["y_va_raw"]
    va_users = data["va_users"]
    print(f"lean-loaded train={X_tr.shape} val={X_va.shape} in {time.time()-t0:.1f}s", flush=True)
    if args.max_recency > 0:
        rec = X_tr[:, feat_cols.index("days_since_last_event")]
        keep = ~(rec > args.max_recency)  # NaN recency (no events yet) is kept
        print(f"max-recency {args.max_recency}: keeping {keep.mean():.3%} of train rows", flush=True)
        X_tr, y_tr_raw = X_tr[keep], y_tr_raw[keep]
        data["age_days_tr"] = data["age_days_tr"][keep]

    raw_target = args.objective == "tweedie"  # tweedie models raw zero-inflated spend
    y_tr = y_tr_raw if raw_target else np.log1p(y_tr_raw)
    y_va = y_va_raw if raw_target else np.log1p(y_va_raw)

    w_tr = None
    if args.half_life > 0:
        w_tr = np.power(0.5, data["age_days_tr"].astype(np.float64) / args.half_life)
        print(f"sample weights: half-life={args.half_life}d, w range [{w_tr.min():.3f}, {w_tr.max():.3f}]")

    # naive baselines for calibration
    gmv30 = np.clip(X_va[:, feat_cols.index("gmv_sum_30d")].astype(np.float64), 0, None)
    print(f"[baseline] predict 0        : RMSLE = {rmsle(y_va_raw, np.zeros_like(y_va_raw)):.5f}")
    print(f"[baseline] predict gmv_30d  : RMSLE = {rmsle(y_va_raw, gmv30):.5f}")
    best_k, best_s = 0.0, 1e9
    for k in np.linspace(0.1, 1.5, 29):
        s = rmsle(y_va_raw, k * gmv30)
        if s < best_s:
            best_k, best_s = k, s
    print(f"[baseline] k*gmv_30d (k={best_k:.2f}): RMSLE = {best_s:.5f}", flush=True)

    params = dict(
        objective=args.objective,
        boosting=args.boosting,
        metric="rmse",
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        num_threads=N_THREADS,
        verbosity=-1,
        seed=args.seed,
    )
    if args.objective == "tweedie":
        params["tweedie_variance_power"] = 1.3
    if args.boosting == "dart":
        params.update(drop_rate=0.1, skip_drop=0.5)
    ds_params = {"max_bin": 63} if args.fast else {}
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, weight=w_tr, params=ds_params)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

    feval = None
    if raw_target:
        params["metric"] = "None"
        def feval(preds, ds):  # noqa: ANN001
            yt = ds.get_label()
            return "rmsle", float(np.sqrt(np.mean(
                (np.log1p(np.clip(yt, 0, None)) - np.log1p(np.clip(preds, 0, None))) ** 2))), False

    model = lgb.train(
        params,
        dtr,
        num_boost_round=args.max_rounds,
        valid_sets=[dva],
        valid_names=["val"],
        feval=feval,
        callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(200)],
    )

    pred_va = model.predict(X_va, num_iteration=model.best_iteration)
    if not raw_target:
        pred_va = np.expm1(pred_va)
    score = rmsle(y_va_raw, np.clip(pred_va, 0, None))
    _ly = np.log1p(y_va_raw); _lp = np.log1p(np.clip(pred_va, 0, None))
    rank_score = min(float(np.sqrt(np.mean((_ly - np.clip(_lp + d, 0, None)) ** 2)))
                     for d in np.linspace(-0.4, 0.2, 61))
    print(f"\n[MODEL] {args.name or args.tag}: val RMSLE = {score:.5f}  rank-score = {rank_score:.5f}"
          f"  (best_iter={model.best_iteration})", flush=True)

    val_pred = pl.DataFrame(
        {"user_id": va_users, "pred": np.clip(pred_va, 0, None), "target": y_va_raw}
    )
    val_pred.write_parquet(MODELS_DIR / f"lgbm_{args.name or args.tag}_valpred.parquet")

    imp = sorted(
        zip(feat_cols, model.feature_importance("gain")), key=lambda x: -x[1]
    )
    print("\ntop-30 features by gain:")
    for name, g in imp[:30]:
        print(f"  {name:<28} {g:,.0f}")

    name = args.name or args.tag
    out = MODELS_DIR / f"lgbm_{name}.txt"
    model.save_model(str(out), num_iteration=model.best_iteration)
    meta = dict(
        tag=args.tag,
        name=name,
        val_rmsle=score,
        rank_score=rank_score,
        best_iter=model.best_iteration,
        feat_cols=feat_cols,
        params=params,
        gap_days=args.gap_days,
        half_life=args.half_life,
        val_anchor=str(val_anchor),
        max_recency=args.max_recency,
        extras=args.extras,
    )
    (MODELS_DIR / f"lgbm_{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved {out}")

    if args.full:
        n_full = int((model.best_iteration or args.max_rounds) * 1.1)  # DART has no early stopping -> best_iteration=0
        del X_tr, y_tr, dtr, dva
        full = lean_load(
            args.tag,
            min_anchor=_date.fromisoformat(args.min_anchor_date) if args.min_anchor_date else None,
            drop_cols=tuple(drop),
            extras_tag=args.extras,
        )
        X_all, y_all_raw = full["X_tr"], full["y_tr_raw"]
        y_all = y_all_raw if raw_target else np.log1p(y_all_raw)
        print(f"\nretraining on ALL anchors ({len(y_all):,} rows) for {n_full} rounds ...", flush=True)
        w_all = None
        if args.half_life > 0:
            w_all = np.power(0.5, full["age_days_tr"].astype(np.float64) / args.half_life)
        d_all = lgb.Dataset(X_all, label=y_all, feature_name=feat_cols, weight=w_all)
        model_full = lgb.train(params, d_all, num_boost_round=n_full)
        out_full = MODELS_DIR / f"lgbm_{name}_full.txt"
        model_full.save_model(str(out_full))
        print(f"saved {out_full}")


if __name__ == "__main__":
    main()

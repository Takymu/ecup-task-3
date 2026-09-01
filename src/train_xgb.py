"""XGBoost hist on log1p target — third GBM family for the blend.

Usage: python train_xgb.py --tag v2 --name xgb_v2 --min-anchor-date 2025-04-01
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import xgboost as xgb

from common import MODELS_DIR, VAL_ANCHOR, rmsle
from train_lgbm import load_anchors

DROP_COLS = ["user_id", "anchor_date", "target", "anchor_month", "anchor_doy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--name", default="xgb_v2")
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--min-anchor-date", default="2025-04-01")
    ap.add_argument("--lr", type=float, default=0.03)
    args = ap.parse_args()

    t0 = time.time()
    df = load_anchors(args.tag)
    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    tr = df.filter(
        (pl.col("anchor_date") <= cutoff)
        & pl.col("days_active_total").is_not_null()
        & (pl.col("anchor_date") >= date.fromisoformat(args.min_anchor_date))
    )
    va = df.filter(pl.col("anchor_date") == VAL_ANCHOR)
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X_tr = tr.select(feat_cols).to_numpy().astype(np.float32)
    y_tr = np.log1p(np.clip(tr["target"].to_numpy(), 0, None))
    X_va = va.select(feat_cols).to_numpy().astype(np.float32)
    y_va_raw = np.clip(va["target"].to_numpy(), 0, None)
    print(f"loaded in {time.time()-t0:.0f}s train={X_tr.shape}", flush=True)

    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr, feature_names=feat_cols)
    dva = xgb.QuantileDMatrix(X_va, label=np.log1p(y_va_raw), ref=dtr, feature_names=feat_cols)
    params = dict(
        objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
        eta=args.lr, max_depth=8, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, nthread=int(os.environ.get("ML_THREADS", "14")), seed=42,
    )
    model = xgb.train(params, dtr, num_boost_round=4000, evals=[(dva, "val")],
                      early_stopping_rounds=150, verbose_eval=200)
    pred = np.clip(np.expm1(model.predict(dva, iteration_range=(0, model.best_iteration + 1))), 0, None)
    score = rmsle(y_va_raw, pred)
    print(f"\n[XGB] val RMSLE = {score:.5f} (best_iter={model.best_iteration})", flush=True)

    model.save_model(str(MODELS_DIR / f"{args.name}.json"))
    pl.DataFrame({"user_id": va["user_id"], "pred": pred, "target": y_va_raw}).write_parquet(
        MODELS_DIR / f"{args.name}_valpred.parquet"
    )
    (MODELS_DIR / f"{args.name}.meta.json").write_text(
        json.dumps(dict(name=args.name, val_rmsle=score, best_iter=int(model.best_iteration), feat_cols=feat_cols), indent=2)
    )


if __name__ == "__main__":
    main()

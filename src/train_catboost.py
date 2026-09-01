"""CatBoost on log1p(target), same anchor protocol as train_lgbm.

Usage: python train_catboost.py --tag v2 [--full] [--gpu]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

from common import FEATURES_DIR, MODELS_DIR, VAL_ANCHOR, rmsle
from train_lgbm import load_anchors

DROP_COLS = ["user_id", "anchor_date", "target", "anchor_month", "anchor_doy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--min-anchor-date", default=None)
    args = ap.parse_args()

    t0 = time.time()
    train_cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    files = sorted((FEATURES_DIR / args.tag).glob("anchor_*.parquet"))
    files = [f for f in files if "2026-02-13" not in f.stem]
    if args.min_anchor_date:
        files = [f for f in files if f.stem.removeprefix("anchor_") >= args.min_anchor_date]

    feat_cols = None
    Xs, ys, val_part = [], [], None
    for f in files:
        part = pl.read_parquet(f).filter(pl.col("days_active_total").is_not_null())
        if feat_cols is None:
            feat_cols = [c for c in part.columns if c not in DROP_COLS]
        a = part["anchor_date"][0]
        if a == VAL_ANCHOR:
            val_part = part
        elif a <= train_cutoff:
            Xs.append(part.select(feat_cols).to_numpy().astype(np.float32))
            ys.append(np.log1p(np.clip(part["target"].to_numpy(), 0, None)).astype(np.float32))
        del part
    X_tr = np.vstack(Xs); del Xs
    y_tr = np.concatenate(ys); del ys
    X_va = val_part.select(feat_cols).to_numpy().astype(np.float32)
    y_va_raw = np.clip(val_part["target"].to_numpy(), 0, None)
    val_df = val_part.select(["user_id"])
    print(f"loaded in {time.time()-t0:.0f}s; train={X_tr.shape} val={X_va.shape}", flush=True)

    params = dict(
        loss_function="RMSE",
        learning_rate=args.lr,
        depth=args.depth,
        l2_leaf_reg=3.0,
        iterations=args.iters,
        random_seed=42,
        od_type="Iter",
        od_wait=150,
        verbose=200,
    )
    if args.gpu:
        params.update(task_type="GPU", devices="0")
    else:
        import os
        params.update(thread_count=int(os.environ.get("ML_THREADS", "14")))

    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_va, np.log1p(y_va_raw)), use_best_model=True)

    pred_va = np.clip(np.expm1(model.predict(X_va)), 0, None)
    score = rmsle(y_va_raw, pred_va)
    print(f"\n[CATBOOST] val RMSLE = {score:.5f}  (best_iter={model.get_best_iteration()})", flush=True)

    name = args.name or f"cb_{args.tag}"
    model.save_model(str(MODELS_DIR / f"{name}.cbm"))
    (MODELS_DIR / f"{name}.meta.json").write_text(
        json.dumps(dict(name=name, val_rmsle=score, best_iter=int(model.get_best_iteration() or 0),
                        feat_cols=feat_cols, params={k: str(v) for k, v in params.items()}), indent=2)
    )
    pl.DataFrame({"user_id": val_df["user_id"], "pred": pred_va, "target": y_va_raw}).write_parquet(
        MODELS_DIR / f"{name}_valpred.parquet"
    )

    if args.full:
        n_full = int(model.get_best_iteration() * 1.1)
        Xs, ys = [], []
        for f in files:
            part = pl.read_parquet(f).filter(pl.col("days_active_total").is_not_null())
            Xs.append(part.select(feat_cols).to_numpy().astype(np.float32))
            ys.append(np.log1p(np.clip(part["target"].to_numpy(), 0, None)).astype(np.float32))
            del part
        X_all = np.vstack(Xs); del Xs
        y_all = np.concatenate(ys); del ys
        print(f"full refit {n_full} iters on {X_all.shape} ...", flush=True)
        p_full = {**params, "iterations": n_full}
        p_full.pop("od_type"); p_full.pop("od_wait")
        model_f = CatBoostRegressor(**p_full)
        model_f.fit(X_all, y_all)
        model_f.save_model(str(MODELS_DIR / f"{name}_full.cbm"))
        print("saved full model")


if __name__ == "__main__":
    main()

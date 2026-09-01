"""Two-stage LightGBM: P(target>0) classifier x E[log1p(target)|>0] regressor.

Optimal RMSLE prediction: expm1( p * E[log1p(y) | y>0] ), since E[log1p] = p * E[log1p | pos].

Usage: python train_two_stage.py --tag v2 [--half-life 90] [--full]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import polars as pl

from common import FEATURES_DIR, MODELS_DIR, VAL_ANCHOR, rmsle
from train_lgbm import load_anchors

DROP_COLS = ["user_id", "anchor_date", "target", "anchor_month", "anchor_doy"]


def make_weights(df: pl.DataFrame, half_life: float) -> np.ndarray | None:
    if half_life <= 0:
        return None
    age = df.select((pl.lit(VAL_ANCHOR) - pl.col("anchor_date")).dt.total_days()).to_numpy().ravel()
    return np.power(0.5, age.astype(np.float64) / half_life)


def fit_pair(X_tr, y_raw_tr, w_tr, X_va, y_raw_va, feat_cols, lr, seed=42, es_data=None):
    """Returns (clf, reg, best iters). es_data: (X, y_raw) for early stopping."""
    pos_tr = y_raw_tr > 0
    X_es, y_es = es_data
    pos_es = y_es > 0

    import os
    n_threads = int(os.environ.get("ML_THREADS", "14"))
    common = dict(
        learning_rate=lr, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=n_threads, verbosity=-1, seed=seed,
    )
    clf_params = dict(objective="binary", metric="binary_logloss", **common)
    dtr = lgb.Dataset(X_tr, label=pos_tr.astype(np.float32), weight=w_tr, feature_name=feat_cols)
    dva = lgb.Dataset(X_es, label=pos_es.astype(np.float32), reference=dtr)
    clf = lgb.train(clf_params, dtr, num_boost_round=3000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(150, verbose=False)])

    reg_params = dict(objective="regression", metric="rmse", **common)
    w_pos = w_tr[pos_tr] if w_tr is not None else None
    dtr_r = lgb.Dataset(X_tr[pos_tr], label=np.log1p(y_raw_tr[pos_tr]), weight=w_pos, feature_name=feat_cols)
    dva_r = lgb.Dataset(X_es[pos_es], label=np.log1p(y_es[pos_es]), reference=dtr_r)
    reg = lgb.train(reg_params, dtr_r, num_boost_round=3000, valid_sets=[dva_r],
                    callbacks=[lgb.early_stopping(150, verbose=False)])
    return clf, reg


def predict_two_stage(clf, reg, X):
    p = clf.predict(X)
    mu = reg.predict(X)
    return np.expm1(np.clip(p * mu, 0, None))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--half-life", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--min-anchor-date", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extras", default=None, help="extras feature tag joined on user_id per anchor")
    args = ap.parse_args()

    t0 = time.time()
    from datetime import date as _date
    from common import lean_load
    train_cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    data = lean_load(
        args.tag,
        cutoff=train_cutoff,
        min_anchor=_date.fromisoformat(args.min_anchor_date) if args.min_anchor_date else None,
        val_anchor=VAL_ANCHOR,
        drop_cols=tuple(DROP_COLS),
        extras_tag=args.extras,
    )
    feat_cols = data["feat_cols"]
    X_tr, y_tr = data["X_tr"], data["y_tr_raw"]
    X_va, y_va = data["X_va"], data["y_va_raw"]
    va_users = data["va_users"]
    print(f"lean-loaded train={X_tr.shape} val={X_va.shape} in {time.time()-t0:.0f}s", flush=True)
    w_tr = None
    if args.half_life > 0:
        w_tr = np.power(0.5, data["age_days_tr"].astype(np.float64) / args.half_life)

    clf, reg = fit_pair(X_tr, y_tr, w_tr, X_va, y_va, feat_cols, args.lr, seed=args.seed, es_data=(X_va, y_va))
    pred = predict_two_stage(clf, reg, X_va)
    score = rmsle(y_va, pred)
    print(f"[TWO-STAGE] val RMSLE = {score:.5f} (clf {clf.best_iteration} it, reg {reg.best_iteration} it)")

    p_va = clf.predict(X_va)
    from sklearn.metrics import roc_auc_score
    print(f"  classifier AUC = {roc_auc_score(y_va > 0, p_va):.5f}")

    name = args.name or f"two_stage_{args.tag}"
    clf.save_model(str(MODELS_DIR / f"{name}_clf.txt"))
    reg.save_model(str(MODELS_DIR / f"{name}_reg.txt"))
    meta = dict(name=name, val_rmsle=score, feat_cols=feat_cols, half_life=args.half_life, lr=args.lr)
    (MODELS_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # save val predictions for stacking/blending analysis
    val_pred = pl.DataFrame({"user_id": va_users, "pred": pred, "p_buy": p_va, "target": y_va})
    val_pred.write_parquet(MODELS_DIR / f"{name}_valpred.parquet")

    if args.full:
        del X_tr, y_tr
        full = lean_load(
            args.tag,
            min_anchor=_date.fromisoformat(args.min_anchor_date) if args.min_anchor_date else None,
            drop_cols=tuple(DROP_COLS),
            extras_tag=args.extras,
        )
        n_clf = int(clf.best_iteration * 1.1)
        n_reg = int(reg.best_iteration * 1.1)
        X_all, y_all = full["X_tr"], full["y_tr_raw"]
        w_all = None
        if args.half_life > 0:
            w_all = np.power(0.5, full["age_days_tr"].astype(np.float64) / args.half_life)
        pos = y_all > 0
        import os as _os
        common = dict(
            learning_rate=args.lr, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
            num_threads=int(_os.environ.get("ML_THREADS", "14")), verbosity=-1, seed=args.seed,
        )
        print(f"full refit: clf {n_clf} it on {len(y_all):,}, reg {n_reg} it on {pos.sum():,}", flush=True)
        clf_f = lgb.train(dict(objective="binary", **common),
                          lgb.Dataset(X_all, label=pos.astype(np.float32), weight=w_all, feature_name=feat_cols),
                          num_boost_round=n_clf)
        w_pos = w_all[pos] if w_all is not None else None
        reg_f = lgb.train(dict(objective="regression", **common),
                          lgb.Dataset(X_all[pos], label=np.log1p(y_all[pos]), weight=w_pos, feature_name=feat_cols),
                          num_boost_round=n_reg)
        clf_f.save_model(str(MODELS_DIR / f"{name}_full_clf.txt"))
        reg_f.save_model(str(MODELS_DIR / f"{name}_full_reg.txt"))
        print("saved full models")


if __name__ == "__main__":
    main()

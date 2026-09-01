"""Optuna sweep for LightGBM on the fast protocol, then refit best on full protocol.

Usage: python tune_optuna.py --tag v2 --trials 60 --name opt_lgbm
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from common import MODELS_DIR, VAL_ANCHOR, rmsle
from train_lgbm import load_anchors

DROP_COLS = ["user_id", "anchor_date", "target", "anchor_month", "anchor_doy"]
N_THREADS = int(os.environ.get("ML_THREADS", "14"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--name", default="opt_lgbm")
    ap.add_argument("--fast-min-anchor", default="2025-09-01")
    ap.add_argument("--full-min-anchor", default="2025-04-01")
    args = ap.parse_args()

    from common import lean_load
    cutoff = VAL_ANCHOR - timedelta(days=30)

    fast = lean_load(args.tag, cutoff=cutoff, min_anchor=date.fromisoformat(args.fast_min_anchor),
                     val_anchor=VAL_ANCHOR, drop_cols=tuple(DROP_COLS))
    feat_cols = fast["feat_cols"]
    X_f = fast["X_tr"]
    y_f = np.log1p(fast["y_tr_raw"])
    X_va, y_va_raw, va_users = fast["X_va"], fast["y_va_raw"], fast["va_users"]
    y_va = np.log1p(y_va_raw)
    print(f"fast train: {X_f.shape}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="regression", metric="rmse", verbosity=-1, num_threads=N_THREADS, seed=42,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 1500, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=1,
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            max_bin=63,
        )
        dtr = lgb.Dataset(X_f, label=y_f, params={"max_bin": 63})
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(120, verbose=False)])
        pred = np.expm1(model.predict(X_va, num_iteration=model.best_iteration))
        trial.set_user_attr("best_iter", model.best_iteration)
        return rmsle(y_va_raw, np.clip(pred, 0, None))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    print(f"\nsweep done in {(time.time()-t0)/60:.0f} min; best fast RMSLE = {study.best_value:.5f}")
    print("best params:", study.best_params, flush=True)

    # refit best config on the full protocol (all dense anchors, full bins)
    del X_f, y_f
    full = lean_load(args.tag, cutoff=cutoff, min_anchor=date.fromisoformat(args.full_min_anchor),
                     drop_cols=tuple(DROP_COLS))
    X, y = full["X_tr"], np.log1p(full["y_tr_raw"])
    print(f"full train: {X.shape}", flush=True)
    params = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=N_THREADS,
                  seed=42, bagging_freq=1, **study.best_params)
    dtr = lgb.Dataset(X, label=y, feature_name=feat_cols)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=6000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(400)])
    pred = np.clip(np.expm1(model.predict(X_va, num_iteration=model.best_iteration)), 0, None)
    score = rmsle(y_va_raw, pred)
    print(f"[OPT-LGBM full] val RMSLE = {score:.5f} (best_iter={model.best_iteration})", flush=True)

    model.save_model(str(MODELS_DIR / f"lgbm_{args.name}.txt"), num_iteration=model.best_iteration)
    (MODELS_DIR / f"lgbm_{args.name}.meta.json").write_text(json.dumps(dict(
        name=args.name, val_rmsle=score, best_iter=model.best_iteration,
        feat_cols=feat_cols, params={k: (v if not isinstance(v, float) else round(v, 6)) for k, v in params.items()},
        study_best=study.best_params,
    ), indent=2, default=str))
    pl.DataFrame({"user_id": va_users, "pred": pred, "target": y_va_raw}).write_parquet(
        MODELS_DIR / f"lgbm_{args.name}_valpred.parquet"
    )
    print("saved model, meta, valpred")


if __name__ == "__main__":
    main()

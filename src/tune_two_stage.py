"""Optuna sweep for the two-stage pair (clf x reg), objective = rank-score (opt-shift RMSLE).

Fast protocol (anchors >= --fast-min-anchor, max_bin 63) for the sweep, then a full-protocol
refit of the best config with early stopping on val.

Usage: python tune_two_stage.py --trials 40 --jobs 2 --name opt_two_stage
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

from common import MODELS_DIR, VAL_ANCHOR, lean_load, rmsle

DROP_COLS = ("user_id", "anchor_date", "target", "anchor_month", "anchor_doy")


def rank_score(y_raw: np.ndarray, pred_raw: np.ndarray) -> tuple[float, float]:
    """(rank-score after optimal additive shift in log space, raw rmsle)."""
    ly = np.log1p(np.clip(y_raw, 0, None))
    lp = np.log1p(np.clip(pred_raw, 0, None))
    raw = float(np.sqrt(np.mean((ly - lp) ** 2)))
    best = min(
        float(np.sqrt(np.mean((ly - np.clip(lp + d, 0, None)) ** 2)))
        for d in np.linspace(-0.4, 0.2, 61)
    )
    return best, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--name", default="opt_two_stage")
    ap.add_argument("--fast-min-anchor", default="2025-09-01")
    ap.add_argument("--full-min-anchor", default="2025-04-01")
    args = ap.parse_args()

    n_threads = max(4, int(os.environ.get("ML_THREADS", "28")) // args.jobs)
    cutoff = VAL_ANCHOR - timedelta(days=30)

    t0 = time.time()
    fast = lean_load(args.tag, cutoff=cutoff, min_anchor=date.fromisoformat(args.fast_min_anchor),
                     val_anchor=VAL_ANCHOR, drop_cols=DROP_COLS)
    feat_cols = fast["feat_cols"]
    X_f, y_f_raw, age_f = fast["X_tr"], fast["y_tr_raw"], fast["age_days_tr"]
    X_va, y_va_raw, va_users = fast["X_va"], fast["y_va_raw"], fast["va_users"]
    pos_f = y_f_raw > 0
    pos_va = y_va_raw > 0
    print(f"fast train {X_f.shape} ({pos_f.mean():.3f} pos), loaded in {time.time()-t0:.0f}s", flush=True)

    def fit_eval(params_c, params_r, w, max_bin=63, rounds=3000, es=120):
        dtr = lgb.Dataset(X_f, label=pos_f.astype(np.float32), weight=w,
                          params={"max_bin": max_bin})
        dva = lgb.Dataset(X_va, label=pos_va.astype(np.float32), reference=dtr)
        clf = lgb.train(params_c, dtr, num_boost_round=rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(es, verbose=False)])
        w_pos = w[pos_f] if w is not None else None
        dtr_r = lgb.Dataset(X_f[pos_f], label=np.log1p(y_f_raw[pos_f]), weight=w_pos,
                            params={"max_bin": max_bin})
        dva_r = lgb.Dataset(X_va[pos_va], label=np.log1p(y_va_raw[pos_va]), reference=dtr_r)
        reg = lgb.train(params_r, dtr_r, num_boost_round=rounds, valid_sets=[dva_r],
                        callbacks=[lgb.early_stopping(es, verbose=False)])
        pred = np.expm1(np.clip(clf.predict(X_va) * reg.predict(X_va), 0, None))
        return clf, reg, pred

    def objective(trial: optuna.Trial) -> float:
        common = dict(
            verbosity=-1, num_threads=n_threads, seed=42, bagging_freq=1,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 1500, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
        )
        params_c = dict(objective="binary", metric="binary_logloss",
                        num_leaves=trial.suggest_int("leaves_clf", 31, 255, log=True), **common)
        params_r = dict(objective="regression", metric="rmse",
                        num_leaves=trial.suggest_int("leaves_reg", 31, 255, log=True), **common)
        w = None
        if trial.suggest_categorical("use_hl", [False, True]):
            hl = trial.suggest_float("half_life", 30, 400, log=True)
            w = np.power(0.5, age_f.astype(np.float64) / hl)
        clf, reg, pred = fit_eval(params_c, params_r, w)
        rs, raw = rank_score(y_va_raw, pred)
        trial.set_user_attr("raw_rmsle", raw)
        trial.set_user_attr("iters", (clf.best_iteration, reg.best_iteration))
        print(f"  trial {trial.number}: rank={rs:.5f} raw={raw:.5f} "
              f"it=({clf.best_iteration},{reg.best_iteration})", flush=True)
        return rs

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, show_progress_bar=False)
    bp = study.best_params
    print(f"\nsweep done in {(time.time()-t0)/60:.0f} min; best rank-score = {study.best_value:.5f}")
    print("best params:", bp, flush=True)

    # full-protocol refit of the best config, ES on val
    del X_f, y_f_raw, age_f, fast
    full = lean_load(args.tag, cutoff=cutoff, min_anchor=date.fromisoformat(args.full_min_anchor),
                     val_anchor=VAL_ANCHOR, drop_cols=DROP_COLS)
    X_f, y_f_raw, age_f = full["X_tr"], full["y_tr_raw"], full["age_days_tr"]
    pos_f = y_f_raw > 0
    print(f"full train {X_f.shape}", flush=True)

    common = dict(verbosity=-1, num_threads=int(os.environ.get("ML_THREADS", "28")),
                  seed=42, bagging_freq=1,
                  learning_rate=bp["learning_rate"], min_data_in_leaf=bp["min_data_in_leaf"],
                  feature_fraction=bp["feature_fraction"], bagging_fraction=bp["bagging_fraction"],
                  lambda_l2=bp["lambda_l2"])
    params_c = dict(objective="binary", metric="binary_logloss", num_leaves=bp["leaves_clf"], **common)
    params_r = dict(objective="regression", metric="rmse", num_leaves=bp["leaves_reg"], **common)
    w = None
    if bp.get("use_hl"):
        w = np.power(0.5, age_f.astype(np.float64) / bp["half_life"])

    dtr = lgb.Dataset(X_f, label=pos_f.astype(np.float32), weight=w, feature_name=feat_cols)
    dva = lgb.Dataset(X_va, label=pos_va.astype(np.float32), reference=dtr)
    clf = lgb.train(params_c, dtr, num_boost_round=4000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(200, verbose=False)])
    w_pos = w[pos_f] if w is not None else None
    dtr_r = lgb.Dataset(X_f[pos_f], label=np.log1p(y_f_raw[pos_f]), weight=w_pos, feature_name=feat_cols)
    dva_r = lgb.Dataset(X_va[pos_va], label=np.log1p(y_va_raw[pos_va]), reference=dtr_r)
    reg = lgb.train(params_r, dtr_r, num_boost_round=4000, valid_sets=[dva_r],
                    callbacks=[lgb.early_stopping(200, verbose=False)])

    pred = np.expm1(np.clip(clf.predict(X_va) * reg.predict(X_va), 0, None))
    rs, raw = rank_score(y_va_raw, pred)
    print(f"[OPT-2S full] val RMSLE = {raw:.5f}  rank-score = {rs:.5f} "
          f"(clf {clf.best_iteration}, reg {reg.best_iteration})", flush=True)

    name = args.name
    clf.save_model(str(MODELS_DIR / f"{name}_clf.txt"))
    reg.save_model(str(MODELS_DIR / f"{name}_reg.txt"))
    (MODELS_DIR / f"{name}.meta.json").write_text(json.dumps(dict(
        name=name, val_rmsle=raw, rank_score=rs, feat_cols=feat_cols,
        clf_iter=clf.best_iteration, reg_iter=reg.best_iteration,
        params=bp, study_best_value=study.best_value), indent=2, default=str))
    pl.DataFrame({"user_id": va_users, "pred": pred, "target": y_va_raw}).write_parquet(
        MODELS_DIR / f"{name}_valpred.parquet")
    print("saved model, meta, valpred", flush=True)


if __name__ == "__main__":
    main()

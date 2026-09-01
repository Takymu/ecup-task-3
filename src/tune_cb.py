"""Overnight Optuna for CatBoost on GPU, objective = rank-score (opt-shift RMSLE).

Fast protocol sweep -> full-protocol refit of the best config -> opt_cb model+valpred+meta.
Usage: python tune_cb.py [--trials 36]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta

import numpy as np
import optuna
import polars as pl
from catboost import CatBoostRegressor

from common import MODELS_DIR, VAL_ANCHOR, lean_load

DROP_COLS = ("user_id", "anchor_date", "target", "anchor_month", "anchor_doy")


def rank_score(y_raw, pred_raw):
    ly = np.log1p(np.clip(y_raw, 0, None))
    lp = np.log1p(np.clip(pred_raw, 0, None))
    raw = float(np.sqrt(np.mean((ly - lp) ** 2)))
    best = min(float(np.sqrt(np.mean((ly - np.clip(lp + d, 0, None)) ** 2)))
               for d in np.linspace(-0.4, 0.2, 61))
    return best, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--trials", type=int, default=36)
    ap.add_argument("--name", default="opt_cb")
    ap.add_argument("--fast-min-anchor", default="2025-09-01")
    ap.add_argument("--full-min-anchor", default="2025-04-01")
    args = ap.parse_args()

    t0 = time.time()
    cutoff = VAL_ANCHOR - timedelta(days=30)
    fast = lean_load(args.tag, cutoff=cutoff, min_anchor=date.fromisoformat(args.fast_min_anchor),
                     val_anchor=VAL_ANCHOR, drop_cols=DROP_COLS)
    feat_cols = fast["feat_cols"]
    X_f, y_f = fast["X_tr"], np.log1p(fast["y_tr_raw"])
    X_va, y_va_raw, va_users = fast["X_va"], fast["y_va_raw"], fast["va_users"]
    y_va = np.log1p(y_va_raw)
    print(f"fast train {X_f.shape} in {time.time()-t0:.0f}s", flush=True)

    def objective(trial: optuna.Trial) -> float:
        m = CatBoostRegressor(
            task_type="GPU", devices="0", random_seed=42, verbose=0,
            loss_function="RMSE", iterations=3000,
            od_type="Iter", od_wait=100,
            depth=trial.suggest_int("depth", 6, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
            random_strength=trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            border_count=128,
        )
        m.fit(X_f, y_f, eval_set=(X_va, y_va), use_best_model=True)
        pred = np.clip(np.expm1(m.predict(X_va)), 0, None)
        rs, raw = rank_score(y_va_raw, pred)
        trial.set_user_attr("best_iter", int(m.get_best_iteration() or 0))
        print(f"  trial {trial.number}: rank={rs:.5f} raw={raw:.5f} "
              f"it={m.get_best_iteration()}", flush=True)
        return rs

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    bp = study.best_params
    best_iter = study.best_trial.user_attrs["best_iter"]
    print(f"\n[OPT-CB] sweep done {(time.time()-t0)/60:.0f} min; "
          f"best rank-score = {study.best_value:.5f}; params {bp}; it {best_iter}", flush=True)

    del X_f, y_f, fast
    full = lean_load(args.tag, cutoff=cutoff, min_anchor=date.fromisoformat(args.full_min_anchor),
                     val_anchor=VAL_ANCHOR, drop_cols=DROP_COLS)
    X, y = full["X_tr"], np.log1p(full["y_tr_raw"])
    print(f"full train {X.shape}", flush=True)
    m = CatBoostRegressor(task_type="GPU", devices="0", random_seed=42, verbose=200,
                          loss_function="RMSE", iterations=3000,
                          od_type="Iter", od_wait=150, border_count=128, **bp)
    m.fit(X, y, eval_set=(X_va, y_va), use_best_model=True)
    pred = np.clip(np.expm1(m.predict(X_va)), 0, None)
    rs, raw = rank_score(y_va_raw, pred)
    print(f"[OPT-CB full] val RMSLE = {raw:.5f}  rank-score = {rs:.5f} "
          f"(it={m.get_best_iteration()})", flush=True)

    m.save_model(str(MODELS_DIR / f"{args.name}.cbm"))
    (MODELS_DIR / f"{args.name}.meta.json").write_text(json.dumps(dict(
        name=args.name, val_rmsle=raw, rank_score=rs, feat_cols=feat_cols,
        best_iter=int(m.get_best_iteration() or 0), params=bp), indent=2, default=str))
    pl.DataFrame({"user_id": va_users, "pred": pred, "target": y_va_raw}).write_parquet(
        MODELS_DIR / f"{args.name}_valpred.parquet")
    print("saved model, meta, valpred", flush=True)


if __name__ == "__main__":
    main()

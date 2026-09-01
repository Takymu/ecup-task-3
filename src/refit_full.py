"""Refit an already-validated model on ALL anchors (incl. val) for submission.

Usage:
  python refit_full.py --model lgbm_expD --kind lgbm [--iters-mult 1.1]
  python refit_full.py --model cb_v2_d10 --kind cb --gpu
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import polars as pl

from common import FEATURES_DIR, MODELS_DIR
from train_lgbm import load_anchors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["lgbm", "cb", "two_stage"], required=True)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--min-anchor-date", default="2025-04-01")
    ap.add_argument("--iters-mult", type=float, default=1.1)
    ap.add_argument("--iters", type=int, default=None, help="override best_iter (cb: from train log)")
    ap.add_argument("--iters-reg", type=int, default=None, help="two_stage regressor iters")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    meta = json.loads((MODELS_DIR / f"{args.model}.meta.json").read_text())
    feat_cols = meta["feat_cols"]

    from datetime import date
    from common import lean_load
    data = lean_load(
        args.tag,
        min_anchor=date.fromisoformat(args.min_anchor_date) if args.min_anchor_date else None,
    )
    assert data["feat_cols"] == feat_cols, "feature set drift between meta and anchor files"
    X, y_raw = data["X_tr"], data["y_tr_raw"]
    y = np.log1p(y_raw)
    print(f"refit on {X.shape}", flush=True)

    if args.kind == "lgbm":
        import lightgbm as lgb
        params = dict(meta["params"])
        params.pop("metric", None)
        n = int((args.iters or meta["best_iter"]) * args.iters_mult)
        model = lgb.train(params, lgb.Dataset(X, label=y, feature_name=feat_cols), num_boost_round=n)
        out = MODELS_DIR / f"{args.model}_full.txt"
        model.save_model(str(out))
    elif args.kind == "two_stage":
        import lightgbm as lgb
        common = dict(
            learning_rate=float(meta["lr"]), num_leaves=127, min_data_in_leaf=200,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
            num_threads=int(os.environ.get("ML_THREADS", "14")), verbosity=-1, seed=42,
        )
        pos = y_raw > 0
        n_clf = int(args.iters * args.iters_mult)
        n_reg = int(args.iters_reg * args.iters_mult)
        print(f"two-stage refit: clf {n_clf} it, reg {n_reg} it on {pos.sum():,} pos", flush=True)
        clf = lgb.train(dict(objective="binary", **common),
                        lgb.Dataset(X, label=pos.astype(np.float32), feature_name=feat_cols),
                        num_boost_round=n_clf)
        reg = lgb.train(dict(objective="regression", **common),
                        lgb.Dataset(X[pos], label=y[pos], feature_name=feat_cols),
                        num_boost_round=n_reg)
        clf.save_model(str(MODELS_DIR / f"{args.model}_full_clf.txt"))
        reg.save_model(str(MODELS_DIR / f"{args.model}_full_reg.txt"))
        out = MODELS_DIR / f"{args.model}_full_(clf|reg).txt"
    else:
        from catboost import CatBoostRegressor
        p = meta["params"]
        best_iter = args.iters or meta.get("best_iter") or 500
        n = int(best_iter * args.iters_mult)
        params = dict(
            loss_function="RMSE", learning_rate=float(p["learning_rate"]), depth=int(p["depth"]),
            l2_leaf_reg=float(p["l2_leaf_reg"]), iterations=n, random_seed=42, verbose=200,
        )
        if args.gpu:
            params.update(task_type="GPU", devices="0")
        model = CatBoostRegressor(**params)
        model.fit(X, y)
        out = MODELS_DIR / f"{args.model}_full.cbm"
        model.save_model(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

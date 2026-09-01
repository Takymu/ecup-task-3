"""Feature importance dump + top-K selection retrain.

1) Fast gbdt (anchors >= fast-min-anchor, max_bin 63) -> gain importances -> {name}_imp.json
2) DART retrain on the top-K features only (full protocol) -> valpred + rank-score.

Usage: python feat_select.py [--top-k 150] [--name dart_top150]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl

from common import MODELS_DIR, VAL_ANCHOR, lean_load, rmsle

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
    ap.add_argument("--top-k", type=int, default=150)
    ap.add_argument("--name", default=None)
    ap.add_argument("--fast-min-anchor", default="2025-09-01")
    ap.add_argument("--full-min-anchor", default="2025-04-01")
    ap.add_argument("--dart-iters", type=int, default=350)
    ap.add_argument("--extras", default=None)
    args = ap.parse_args()
    name = args.name or f"dart_top{args.top_k}"
    n_threads = int(os.environ.get("ML_THREADS", "48"))

    t0 = time.time()
    cutoff = VAL_ANCHOR - timedelta(days=30)
    fast = lean_load(args.tag, cutoff=cutoff, min_anchor=date.fromisoformat(args.fast_min_anchor),
                     val_anchor=VAL_ANCHOR, drop_cols=DROP_COLS, extras_tag=args.extras)
    feat_cols = fast["feat_cols"]
    X_f, y_f = fast["X_tr"], np.log1p(fast["y_tr_raw"])
    X_va, y_va_raw, va_users = fast["X_va"], fast["y_va_raw"], fast["va_users"]
    print(f"fast train {X_f.shape} in {time.time()-t0:.0f}s", flush=True)

    params = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=n_threads,
                  seed=42, learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0)
    dtr = lgb.Dataset(X_f, label=y_f, feature_name=feat_cols, params={"max_bin": 63})
    dva = lgb.Dataset(X_va, label=np.log1p(y_va_raw), reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(120, verbose=False)])
    gain = m.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1]
    ranked = [(feat_cols[i], float(gain[i])) for i in order]
    (MODELS_DIR / f"{name}_imp.json").write_text(json.dumps(ranked, indent=1))
    print("top-25 by gain:", flush=True)
    for c, g in ranked[:25]:
        print(f"  {g:14.0f}  {c}", flush=True)
    zero_gain = sum(1 for _, g in ranked if g == 0)
    print(f"features with ZERO gain: {zero_gain}/{len(feat_cols)}", flush=True)

    keep = [c for c, _ in ranked[:args.top_k]]
    keep_idx = [feat_cols.index(c) for c in keep]
    del X_f, y_f, fast, dtr, dva, m

    full = lean_load(args.tag, cutoff=cutoff, min_anchor=date.fromisoformat(args.full_min_anchor),
                     val_anchor=VAL_ANCHOR, drop_cols=DROP_COLS, extras_tag=args.extras)
    X, y = full["X_tr"][:, keep_idx], np.log1p(full["y_tr_raw"])
    X_va = full["X_va"][:, keep_idx]
    print(f"full train (top-{args.top_k}) {X.shape}", flush=True)
    params["boosting"] = "dart"
    dtr = lgb.Dataset(X, label=y, feature_name=keep)
    m = lgb.train(params, dtr, num_boost_round=args.dart_iters)
    pred = np.clip(np.expm1(m.predict(X_va)), 0, None)
    rs, raw = rank_score(y_va_raw, pred)
    print(f"[{name}] val RMSLE = {raw:.5f}  rank-score = {rs:.5f}", flush=True)

    m.save_model(str(MODELS_DIR / f"lgbm_{name}.txt"))
    (MODELS_DIR / f"lgbm_{name}.meta.json").write_text(json.dumps(dict(
        name=name, val_rmsle=raw, rank_score=rs, best_iter=args.dart_iters,
        feat_cols=keep, top_k=args.top_k, extras=args.extras), indent=2))
    pl.DataFrame({"user_id": va_users, "pred": pred, "target": y_va_raw}).write_parquet(
        MODELS_DIR / f"lgbm_{name}_valpred.parquet")
    print("saved model, imp, valpred", flush=True)


if __name__ == "__main__":
    main()

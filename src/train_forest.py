"""Diversity learners for the rank-blend: ridge / random forest / extra trees.

Each kind: fit on fast-protocol anchors -> val rank-score -> refit incl. val anchor ->
test-anchor prediction saved as {name}_testpred.parquet (no giant model pickles to evacuate).

Usage: python train_forest.py --kind ridge|rf|et [--min-anchor 2025-09-01]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta

import numpy as np
import polars as pl

from common import FEATURES_DIR, MODELS_DIR, TEST_ANCHOR, VAL_ANCHOR, lean_load, rmsle

DROP_COLS = ("user_id", "anchor_date", "target", "anchor_month", "anchor_doy")


def rank_score(y_raw, pred_raw):
    ly = np.log1p(np.clip(y_raw, 0, None))
    lp = np.log1p(np.clip(pred_raw, 0, None))
    raw = float(np.sqrt(np.mean((ly - lp) ** 2)))
    best = min(float(np.sqrt(np.mean((ly - np.clip(lp + d, 0, None)) ** 2)))
               for d in np.linspace(-0.4, 0.2, 61))
    return best, raw


def make_model(kind: str, n_jobs: int, alpha: float = 3.0):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=alpha, random_state=42)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=100, max_features=0.3,
            n_jobs=n_jobs, random_state=42)
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=100, max_features=0.5,
        n_jobs=n_jobs, random_state=42)


def transform(X: np.ndarray, kind: str, mu=None, sd=None):
    """Ridge needs symmetric-log + standardization; trees eat raw."""
    if kind != "ridge":
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), mu, sd
    X = np.sign(X) * np.log1p(np.abs(X))
    if mu is None:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0) + 1e-6
    return np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0), mu, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["ridge", "rf", "et"], required=True)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--min-anchor", default="2025-09-01")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or f"{args.kind}_{args.tag}"
    n_jobs = int(os.environ.get("ML_THREADS", "60"))

    t0 = time.time()
    cutoff = VAL_ANCHOR - timedelta(days=30)
    data = lean_load(args.tag, cutoff=cutoff, min_anchor=date.fromisoformat(args.min_anchor),
                     val_anchor=VAL_ANCHOR, drop_cols=DROP_COLS)
    feat_cols = data["feat_cols"]
    X_tr, mu, sd = transform(data["X_tr"], args.kind)
    y_tr = np.log1p(data["y_tr_raw"])
    X_va, _, _ = transform(data["X_va"], args.kind, mu, sd)
    y_va_raw, va_users = data["y_va_raw"], data["va_users"]
    print(f"[{name}] train {X_tr.shape} loaded {time.time()-t0:.0f}s", flush=True)

    model = make_model(args.kind, n_jobs)
    model.fit(X_tr, y_tr)
    pred = np.clip(np.expm1(model.predict(X_va)), 0, None)
    rs, raw = rank_score(y_va_raw, pred)
    print(f"[{name}] val RMSLE = {raw:.5f}  rank-score = {rs:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    pl.DataFrame({"user_id": va_users, "pred": pred, "target": y_va_raw}).write_parquet(
        MODELS_DIR / f"{name}_valpred.parquet")
    (MODELS_DIR / f"{name}.meta.json").write_text(json.dumps(dict(
        name=name, kind=args.kind, val_rmsle=raw, rank_score=rs,
        min_anchor=args.min_anchor, n_feat=len(feat_cols)), indent=2))

    # full refit incl. val anchor, then test prediction — testpred saved right here
    del X_tr, y_tr, data
    full = lean_load(args.tag, min_anchor=date.fromisoformat(args.min_anchor), drop_cols=DROP_COLS)
    X_all, mu, sd = transform(full["X_tr"], args.kind)
    y_all = np.log1p(full["y_tr_raw"])
    del full
    print(f"[{name}] refit on {X_all.shape}", flush=True)
    model = make_model(args.kind, n_jobs)
    model.fit(X_all, y_all)
    del X_all, y_all

    test_df = pl.read_parquet(FEATURES_DIR / args.tag / f"anchor_{TEST_ANCHOR}.parquet").sort("user_id")
    X_te, _, _ = transform(test_df.select(feat_cols).to_numpy().astype(np.float32), args.kind, mu, sd)
    pred_te = np.clip(np.expm1(model.predict(X_te)), 0, None)
    pl.DataFrame({"user_id": test_df["user_id"].to_numpy(), "pred": pred_te}).write_parquet(
        MODELS_DIR / f"{name}_full_testpred.parquet")
    print(f"[{name}] testpred saved, mean_lp={np.log1p(pred_te).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()

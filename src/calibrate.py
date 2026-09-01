"""E15: monotone level recalibration in log-space, tested by anchor-to-anchor transfer.

Fit: quantile-binned piecewise-linear map log_pred -> mean(log_target) on --cal-anchor.
Apply: to --eval-anchor predictions; report RMSLE before/after (+ self-fit upper bound).

Usage:
  python calibrate.py --model lgbm_expD60 --meta lgbm_expD60 --tag v2 \
      --cal-anchor 2025-12-17 --eval-anchor 2026-01-14
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import lightgbm as lgb
import numpy as np
import polars as pl

from common import FEATURES_DIR, MODELS_DIR, rmsle


def fit_map(lp: np.ndarray, lt: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    qs = np.quantile(lp, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    xs, ys = [], []
    for i in range(n_bins):
        m = (lp >= qs[i]) & (lp < qs[i + 1])
        if m.sum() < 50:
            continue
        xs.append(lp[m].mean())
        ys.append(lt[m].mean())
    xs, ys = np.array(xs), np.array(ys)
    ys = np.maximum.accumulate(ys)  # enforce monotonicity
    return xs, ys


def apply_map(lp: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = np.interp(lp, xs, ys)
    # linear extrapolation on the right tail (keep identity slope beyond last bin)
    hi = lp > xs[-1]
    out[hi] = ys[-1] + (lp[hi] - xs[-1])
    return np.clip(out, 0, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model file stem")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--cal-anchor", required=True)
    ap.add_argument("--eval-anchor", default="2026-01-14")
    ap.add_argument("--bins", type=int, default=25)
    args = ap.parse_args()

    model = lgb.Booster(model_file=str(MODELS_DIR / f"{args.model}.txt"))
    feat_cols = json.loads((MODELS_DIR / f"{args.model}.meta.json").read_text())["feat_cols"]

    def load(anchor: str):
        df = pl.read_parquet(FEATURES_DIR / args.tag / f"anchor_{anchor}.parquet")
        X = df.select(feat_cols).to_numpy().astype(np.float32)
        y = np.clip(df["target"].to_numpy(), 0, None)
        lp = np.clip(model.predict(X), 0, None)
        return lp, np.log1p(y), y

    lp_c, lt_c, _ = load(args.cal_anchor)
    lp_e, lt_e, y_e = load(args.eval_anchor)

    print(f"raw eval RMSLE      = {np.sqrt(np.mean((lt_e - lp_e) ** 2)):.5f}")
    print(f"cal-anchor raw bias = {np.mean(lt_c - lp_c):+.4f}; eval raw bias = {np.mean(lt_e - lp_e):+.4f}")

    xs, ys = fit_map(lp_c, lt_c, args.bins)
    lp_e_cal = apply_map(lp_e, xs, ys)
    print(f"transfer-cal RMSLE  = {np.sqrt(np.mean((lt_e - lp_e_cal) ** 2)):.5f}  (fit@{args.cal_anchor})")

    xs2, ys2 = fit_map(lp_e, lt_e, args.bins)
    lp_e_self = apply_map(lp_e, xs2, ys2)
    print(f"self-cal RMSLE      = {np.sqrt(np.mean((lt_e - lp_e_self) ** 2)):.5f}  (upper bound, circular)")

    print("\nmap sample (log-pred -> log-target):")
    for x, y_ in zip(xs[::4], ys[::4]):
        print(f"  {x:6.3f} -> {y_:6.3f}  (delta {y_-x:+.3f})")


if __name__ == "__main__":
    main()

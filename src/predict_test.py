"""Write {stem}_testpred.parquet (raw-scale preds) for trained GBM models at TEST_ANCHOR.

Runs where the dense test features live (server). Spec: comma-separated kind:stem[:extras_tag],
kind in {lgbm, cb, two_stage}; stem is the model file stem (with _full suffix if a refit model).

Usage:
  python predict_test.py --tag v2 --spec "lgbm:lgbm_srv_dart,cb:srv_cb_full,lgbm:lgbm_srv_trait:v2_trait"
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from common import FEATURES_DIR, MODELS_DIR, TEST_ANCHOR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()

    base = pl.read_parquet(FEATURES_DIR / args.tag / f"anchor_{TEST_ANCHOR.isoformat()}.parquet")

    for part in args.spec.split(","):
        bits = part.strip().split(":")
        kind, stem = bits[0], bits[1]
        extras = bits[2] if len(bits) > 2 else None
        test_df = base
        if extras:
            ex = pl.read_parquet(FEATURES_DIR / extras / f"anchor_{TEST_ANCHOR.isoformat()}.parquet")
            ex = ex.drop([c for c in ("anchor_date", "target") if c in ex.columns])
            test_df = base.join(ex, on="user_id", how="left")

        meta_stem = stem.removesuffix("_full")
        feat_cols = json.loads((MODELS_DIR / f"{meta_stem}.meta.json").read_text())["feat_cols"]
        X = test_df.select(feat_cols).to_numpy().astype(np.float32)

        if kind == "lgbm":
            import lightgbm as lgb
            lp = np.clip(lgb.Booster(model_file=str(MODELS_DIR / f"{stem}.txt")).predict(X), 0, None)
        elif kind == "cb":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(str(MODELS_DIR / f"{stem}.cbm"))
            lp = np.clip(m.predict(X), 0, None)
        elif kind == "two_stage":
            import lightgbm as lgb
            clf = lgb.Booster(model_file=str(MODELS_DIR / f"{stem}_clf.txt"))
            reg = lgb.Booster(model_file=str(MODELS_DIR / f"{stem}_reg.txt"))
            lp = np.clip(clf.predict(X) * reg.predict(X), 0, None)
        else:
            raise ValueError(kind)

        pl.DataFrame({"user_id": test_df["user_id"], "pred": np.expm1(lp)}).write_parquet(
            MODELS_DIR / f"{stem}_testpred.parquet"
        )
        print(f"{stem}: mean_lp={lp.mean():.4f} zeros={(lp < 1e-6).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()

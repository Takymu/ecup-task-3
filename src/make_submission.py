"""Predict at TEST_ANCHOR with a weighted log-space blend of full models, write submission CSV.

Usage:
  python make_submission.py --tag v2 --name sub_v2_blend \
      --models "lgbm:lgbm_expD_full:0.65,cb:cb_v2_d10_full:0.35" [--mult 1.0]

Model spec: type:stem:weight  (type in {lgbm, cb, two_stage}; two_stage stem expects _clf/_reg pair)
--mult: raw multiplier applied to final predictions (seasonal level probe).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from common import FEATURES_DIR, MODELS_DIR, SAMPLE_SUBMIT, SUBMISSIONS_DIR, TEST_ANCHOR


def load_feat_cols(meta_stem: str) -> list[str]:
    return json.loads((MODELS_DIR / f"{meta_stem}.meta.json").read_text())["feat_cols"]


def predict_one(kind: str, stem: str, test_df: pl.DataFrame) -> np.ndarray:
    """Returns log1p-space predictions."""
    if kind == "lgbm":
        import lightgbm as lgb
        meta_stem = stem.removesuffix("_full")
        X = test_df.select(load_feat_cols(meta_stem)).to_numpy().astype(np.float32)
        model = lgb.Booster(model_file=str(MODELS_DIR / f"{stem}.txt"))
        return np.clip(model.predict(X), 0, None)
    if kind == "cb":
        from catboost import CatBoostRegressor
        meta_stem = stem.removesuffix("_full")
        X = test_df.select(load_feat_cols(meta_stem)).to_numpy().astype(np.float32)
        model = CatBoostRegressor()
        model.load_model(str(MODELS_DIR / f"{stem}.cbm"))
        return np.clip(model.predict(X), 0, None)
    if kind == "two_stage":
        import lightgbm as lgb
        meta_stem = stem.removesuffix("_full")
        X = test_df.select(load_feat_cols(meta_stem)).to_numpy().astype(np.float32)
        clf = lgb.Booster(model_file=str(MODELS_DIR / f"{stem}_clf.txt"))
        reg = lgb.Booster(model_file=str(MODELS_DIR / f"{stem}_reg.txt"))
        return np.clip(clf.predict(X) * reg.predict(X), 0, None)
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--mult", type=float, default=1.0)
    args = ap.parse_args()

    test_df = pl.read_parquet(FEATURES_DIR / args.tag / f"anchor_{TEST_ANCHOR.isoformat()}.parquet")

    specs = []
    for part in args.models.split(","):
        kind, stem, w = part.strip().split(":")
        specs.append((kind, stem, float(w)))
    wsum = sum(w for _, _, w in specs)

    log_pred = np.zeros(test_df.height)
    for kind, stem, w in specs:
        lp = predict_one(kind, stem, test_df)
        print(f"  {kind}:{stem} w={w:.2f}  log-pred mean={lp.mean():.4f}")
        log_pred += (w / wsum) * lp

    pred = np.clip(np.expm1(log_pred), 0, None) * args.mult

    sub = pl.DataFrame({"user_id": test_df["user_id"], "predict": pred})
    sample = pl.read_csv(SAMPLE_SUBMIT)
    sub = sample.select("user_id").join(sub, on="user_id", how="left")
    assert sub["predict"].null_count() == 0
    out = SUBMISSIONS_DIR / f"{args.name}.csv"
    sub.write_csv(out)
    print(f"saved {out}")
    print(f"pred stats: mean={pred.mean():.2f} p50={np.median(pred):.2f} p99={np.percentile(pred,99):.2f} "
          f"max={pred.max():.2f} ~zeros={(pred<0.5).mean():.3f}")


if __name__ == "__main__":
    main()

"""Predict at TEST_ANCHOR with a saved model and write a submission CSV.

Usage:
  python predict.py --tag v1 [--model lgbm_v1_full] [--name v1_full]
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from common import FEATURES_DIR, MODELS_DIR, SAMPLE_SUBMIT, SUBMISSIONS_DIR, TEST_ANCHOR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--model", default=None, help="model file stem in artifacts/models (default lgbm_<tag>)")
    ap.add_argument("--name", default=None, help="submission name (default = model stem)")
    args = ap.parse_args()

    stem = args.model or f"lgbm_{args.tag}"
    model = lgb.Booster(model_file=str(MODELS_DIR / f"{stem}.txt"))
    meta_path = MODELS_DIR / f"lgbm_{args.tag}.meta.json"
    feat_cols = json.loads(meta_path.read_text())["feat_cols"]

    test_path = FEATURES_DIR / args.tag / f"anchor_{TEST_ANCHOR.isoformat()}.parquet"
    test_df = pl.read_parquet(test_path)
    X = test_df.select(feat_cols).to_numpy().astype(np.float32)
    pred = np.expm1(model.predict(X))
    pred = np.clip(pred, 0, None)

    sub = pl.DataFrame({"user_id": test_df["user_id"], "predict": pred})

    # align with sample submission user set/order
    sample = pl.read_csv(SAMPLE_SUBMIT)
    sub = sample.select("user_id").join(sub, on="user_id", how="left")
    assert sub["predict"].null_count() == 0, "missing users in prediction!"

    name = args.name or stem
    out = SUBMISSIONS_DIR / f"{name}.csv"
    sub.write_csv(out)
    print(f"saved {out}  rows={sub.height:,}")
    print(f"pred stats: mean={pred.mean():.2f} p50={np.median(pred):.2f} "
          f"p99={np.percentile(pred, 99):.2f} max={pred.max():.2f} zeros={(pred < 0.5).mean():.3f}")


if __name__ == "__main__":
    main()

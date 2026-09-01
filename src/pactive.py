"""P(active in next 30d | features) model, trained at test-like anchors (all users), and the
multiplicative correction lp_corrected = lp * P_active^alpha for selection-biased predictions.

Modes:
  --eval  : train on anchors in [train-lo, train-hi], evaluate correction on --eval-anchor for given valpreds
  --fit   : train on anchors in [train-lo, train-hi], predict P_active for --apply-anchor (e.g. TEST) -> parquet
Usage:
  python pactive.py --eval --train-lo 2025-06-01 --train-hi 2025-08-31 --eval-anchor 2025-10-01 --valpreds l_gru_proxy,lgbm_dart_proxy
  python pactive.py --fit --train-lo 2025-06-01 --train-hi 2025-10-15 --apply-anchor 2026-02-13
"""
from __future__ import annotations
import argparse, os
from datetime import date, timedelta
import numpy as np, polars as pl, lightgbm as lgb
from common import FEATURES_DIR, MODELS_DIR, TRAIN_PARQUET

DROP = {"user_id", "anchor_date", "target", "anchor_month", "anchor_doy"}


def active_label(anchor: date) -> pl.DataFrame:
    return (pl.scan_parquet(TRAIN_PARQUET).select(["event_date", "user_id"])
            .filter((pl.col("event_date") > anchor) & (pl.col("event_date") <= anchor + timedelta(days=30)))
            .group_by("user_id").len().rename({"len": "act"}).collect())


def load_anchor(anchor: date, tag: str, with_label=True):
    df = pl.read_parquet(FEATURES_DIR / tag / f"anchor_{anchor}.parquet").filter(pl.col("days_active_total").is_not_null())
    if with_label:
        df = df.join(active_label(anchor), on="user_id", how="left").with_columns((pl.col("act").fill_null(0) > 0).cast(pl.Int8).alias("y_act"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2"); ap.add_argument("--eval", action="store_true"); ap.add_argument("--fit", action="store_true")
    ap.add_argument("--train-lo", default="2025-06-01"); ap.add_argument("--train-hi", default="2025-08-31")
    ap.add_argument("--eval-anchor", default="2025-10-01"); ap.add_argument("--apply-anchor", default="2026-02-13")
    ap.add_argument("--valpreds", default=""); ap.add_argument("--name", default="pactive")
    a = ap.parse_args()
    lo, hi = date.fromisoformat(a.train_lo), date.fromisoformat(a.train_hi)
    anchors = sorted(date.fromisoformat(f.stem.removeprefix("anchor_")) for f in (FEATURES_DIR / a.tag).glob("anchor_*.parquet"))
    tr = [d for d in anchors if lo <= d <= hi]
    parts = [load_anchor(d, a.tag) for d in tr]
    cols = [c for c in parts[0].columns if c not in DROP and c not in ("act", "y_act")]
    X = np.vstack([p.select(cols).to_numpy().astype(np.float32) for p in parts]); y = np.concatenate([p["y_act"].to_numpy() for p in parts])
    print(f"P_active train: {len(tr)} anchors {tr[0]}..{tr[-1]}, rows {len(y):,}, active share {y.mean():.3f}", flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbosity=-1, num_threads=int(os.environ.get("ML_THREADS", "32")), seed=0)
    m = lgb.train(params, lgb.Dataset(X, label=y, feature_name=cols), num_boost_round=400)
    del X
    if a.eval:
        ev = date.fromisoformat(a.eval_anchor)
        E = load_anchor(ev, a.tag).sort("user_id")
        p = m.predict(E.select(cols).to_numpy().astype(np.float32))
        from sklearn.metrics import roc_auc_score
        print(f"eval @ {ev}: active share {E['y_act'].mean():.3f}, AUC(P_active) = {roc_auc_score(E['y_act'].to_numpy(), p):.4f}, mean p {p.mean():.3f}, p q05/q50 {np.quantile(p,[.05,.5]).round(3)}", flush=True)
        pa = pl.DataFrame({"user_id": E["user_id"], "p_act": p})
        for stem in [s for s in a.valpreds.split(",") if s]:
            vp = pl.read_parquet(MODELS_DIR / f"{stem}_valpred.parquet").join(pa, on="user_id", how="left").sort("user_id")
            yy = np.log1p(np.clip(vp["target"].to_numpy(), 0, None)); lp = np.log1p(np.clip(vp["pred"].to_numpy(), 0, None)); pp = vp["p_act"].fill_null(1.0).to_numpy()
            def best(v):
                g = np.linspace(-0.8, 0.4, 121); return min(np.sqrt(np.mean((yy - np.clip(v + d, 0, None)) ** 2)) for d in g)
            base = best(lp)
            line = f"  {stem}: base {base:.5f}"
            for al in (0.5, 1.0, 1.5, 2.0):
                line += f" | a={al}: {best(lp * pp ** al):.5f}"
            # additive variant: lp + beta*log(p)
            for be in (0.5, 1.0):
                line += f" | add b={be}: {best(lp + be * np.log(np.clip(pp, 1e-3, 1))):.5f}"
            print(line, flush=True)
    if a.fit:
        ap_ = date.fromisoformat(a.apply_anchor)
        E = pl.read_parquet(FEATURES_DIR / a.tag / f"anchor_{ap_}.parquet").sort("user_id")
        p = m.predict(E.select(cols).to_numpy().astype(np.float32))
        pl.DataFrame({"user_id": E["user_id"], "p_act": p}).write_parquet(MODELS_DIR / f"{a.name}_{ap_}.parquet")
        m.save_model(str(MODELS_DIR / f"{a.name}.txt"))
        print(f"saved {a.name}_{ap_}.parquet: mean p {p.mean():.3f}, q05/q25/q50 {np.quantile(p,[.05,.25,.5]).round(3)}", flush=True)


if __name__ == "__main__":
    main()

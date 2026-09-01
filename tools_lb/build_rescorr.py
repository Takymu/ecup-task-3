"""Full residual-correction probe: small LGBM trained on full-val residual of v24ind blend,
applied to test features + test lp. Saves probe_form_rescorr.npy + sub_probe_rescorr_L233.csv."""
import numpy as np, polars as pl, lightgbm as lgb
from pathlib import Path

T = Path(__file__).resolve().parents[1]
d = np.load(T/"artifacts/v24ind_val_blend.npz")
uid, y, lp = d["uid"], d["y"], d["lp"]
r = y - lp
DROP = ["user_id", "anchor_date", "target", "anchor_month", "anchor_doy"]

fv = pl.read_parquet(T/"artifacts/features/v2/anchor_2026-01-14.parquet").sort("user_id")
assert (fv["user_id"].to_numpy() == uid).all()
cols = [c for c in fv.columns if c not in DROP]
Xv = np.hstack([fv.select(cols).to_numpy().astype(np.float32),
                (lp - lp.mean()).reshape(-1, 1).astype(np.float32)])

params = dict(objective="regression", learning_rate=0.03, num_leaves=15, max_depth=4,
              min_data_in_leaf=2000, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=10.0, seed=42, verbose=-1, num_threads=8)
mdl = lgb.train(params, lgb.Dataset(Xv, r), num_boost_round=200)
pv = mdl.predict(Xv); pv -= pv.mean()
print("in-sample val gain:", np.sqrt(np.mean(r**2)) - np.sqrt(np.mean((r - pv)**2)), "rms(d)_val:", np.sqrt(np.mean(pv**2)))

ft = pl.read_parquet(T/"artifacts/features/v2/anchor_2026-02-13.parquet").sort("user_id")
sub = pl.read_csv(T/"artifacts/submissions/sub_v24ind_comb5_L233.csv").sort("user_id")
assert (ft["user_id"].to_numpy() == sub["user_id"].to_numpy()).all()
lpt = np.log1p(sub["predict"].to_numpy())
Xt = np.hstack([ft.select(cols).to_numpy().astype(np.float32),
                (lpt - lpt.mean()).reshape(-1, 1).astype(np.float32)])
dt = mdl.predict(Xt); dt -= dt.mean()
print("test form rms:", np.sqrt(np.mean(dt**2)),
      "corr with mined form:", np.corrcoef(dt, np.load(T/"artifacts/forms/probe_form_mined.npy"))[0, 1])
np.save(T/"artifacts/forms/probe_form_rescorr.npy", dt)

x = lpt + dt; x += 2.33 - x.mean()
pred = np.clip(np.expm1(np.clip(x, 0, None)), 0, None)
out = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(
    pl.DataFrame({"user_id": sub["user_id"].to_numpy(), "predict": pred}), on="user_id", how="left")
assert out["predict"].null_count() == 0
out.write_csv(T/"artifacts/submissions/sub_probe_rescorr_L233.csv")
mdl.save_model(str(T/"artifacts/rescorr_lgbm.txt"))
print("saved sub_probe_rescorr_L233.csv + rescorr_lgbm.txt")

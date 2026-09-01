"""Production panel-corrector: small LGBM on pz-blend residuals pooled over TWO CLEAN panels
(10-15, 11-26), applied to the real test anchor on top of sub_v24ind. Probe sub + form."""
import numpy as np, polars as pl, lightgbm as lgb
from pathlib import Path

T = Path(__file__).resolve().parents[2]; EVAC = T/"artifacts/models"; S = T/"artifacts/submissions"
W = {"pz_a32_s42": 0.3, "pz_a32_s1": 0.25, "pz_noctx_s42": 0.15, "pz_gru_s42": 0.15, "pz_dart": 0.15}
DROP = ["user_id", "anchor_date", "target", "anchor_month", "anchor_doy"]

def blend(files):
    acc = None
    for s, w in W.items():
        df = pl.read_parquet(EVAC / files(s)).sort("user_id")
        lp = np.log1p(np.clip(df["pred"].to_numpy(), 0, None))
        acc = w * lp if acc is None else acc + w * lp
    return acc

def shift(y, lp, m):
    g = np.linspace(-0.9, 0.9, 181)
    return g[int(np.argmin([np.sqrt(np.mean((y[m] - np.clip(lp[m] + d, 0, None)) ** 2)) for d in g]))]

Xs, rs = [], []
# panel-1 ONLY: October structure has proven forward transfer (+0.0013 to P2); December (P2) is
# season-specific (reverse transfer fails) - keep it out of the production corrector.
for anchor, panel_file, val_files in (
    ("2025-10-15", "pseudo_panel_2025-10-15.parquet", lambda s: f"{s}_valpred.parquet"),
):
    p = pl.read_parquet(T/f"artifacts/{panel_file}").sort("user_id")
    on = p["on"].to_numpy() == 1
    if "t" in p.columns:
        y = np.log1p(np.clip(p["t"].to_numpy(), 0, None))
    else:
        y = np.log1p(np.clip(pl.read_parquet(EVAC/"pz_a32_s42_valpred.parquet").sort("user_id")["target"].to_numpy(), 0, None))
    lp = blend(val_files)
    lps = np.clip(lp + shift(y, lp, on), 0, None); r = y - lps
    F = pl.read_parquet(T/f"artifacts/features/v2/anchor_{anchor}.parquet").sort("user_id")
    X = np.hstack([F.drop([c for c in DROP if c in F.columns]).to_numpy().astype(np.float32),
                   (lp - lp.mean()).reshape(-1, 1).astype(np.float32)])
    Xs.append(X[on]); rs.append(r[on])
    print(f"anchor {anchor}: panel rows {on.sum()}, resid rms {np.sqrt(np.mean(r[on]**2)):.4f}")

Xtr = np.vstack(Xs); rtr = np.concatenate(rs)
P = dict(objective="regression", learning_rate=0.03, num_leaves=15, max_depth=4, min_data_in_leaf=4000,
         feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, seed=42, verbose=-1, num_threads=8)
mdl = lgb.train(P, lgb.Dataset(Xtr, rtr), num_boost_round=200)
mdl.save_model(str(T/"artifacts/pcorr_lgbm.txt"))

base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id")
lpt = np.log1p(base["predict"].to_numpy())
Ft = pl.read_parquet(T/"artifacts/features/v2/anchor_2026-02-13.parquet").sort("user_id")
assert (Ft["user_id"].to_numpy() == base["user_id"].to_numpy()).all()
Xt = np.hstack([Ft.drop([c for c in DROP if c in Ft.columns]).to_numpy().astype(np.float32),
                (lpt - lpt.mean()).reshape(-1, 1).astype(np.float32)])
dt = mdl.predict(Xt); dt -= dt.mean()
res = np.load(T/"artifacts/forms/probe_form_rescorr.npy")
print("form rms:", round(float(np.sqrt(np.mean(dt**2))), 4), "corr with rescorr form:", round(float(np.corrcoef(dt, res)[0, 1]), 3))
np.save(T/"artifacts/forms/probe_form_pcorr.npy", dt)
x = lpt + dt; x += 2.33 - x.mean()
pred = np.clip(np.expm1(np.clip(x, 0, None)), 0, None)
out = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(
    pl.DataFrame({"user_id": base["user_id"].to_numpy(), "predict": pred}), on="user_id", how="left")
assert out["predict"].null_count() == 0
out.write_csv(S/"sub_probe_pcorr_L233.csv")
print("saved sub_probe_pcorr_L233.csv")

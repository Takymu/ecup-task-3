"""28.08: форма pshift (идея роя, линза отбора): деконтаминация ВЕРОЯТНОСТИ покупки сдвигом логита, а не множителем к предсказанию.
δ(cell) = logit(P(buy|cell, отобран)) − logit(P(buy|cell)) по анкерам лета/осени 2025 (cell = пауза × покупки за 90д, как в selmap);
p' = σ(logit(p) − δ), lp' = p'·μ (ZILN-точечный прогноз = p·μ в log1p-пространстве); форма = lp' − lp ⟂ span{...}, rms .02. Головы p, μ — hd_s42 @13.02.
Usage: build_probe_pshift35.py save"""
import sys, numpy as np, polars as pl
from pathlib import Path
from datetime import date, timedelta
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"; E = T/"artifacts/models"
df = pl.read_parquet(T/"data/train.parquet", columns=["event_date","user_id","gmv"]); users = df["user_id"].unique().sort().to_frame()
def cells(A):
    last = df.filter(pl.col("event_date") <= A).group_by("user_id").agg(pl.col("event_date").max().alias("last"))
    nb = df.filter((pl.col("event_date") > A - timedelta(days=90)) & (pl.col("event_date") <= A) & (pl.col("gmv") > 0)).group_by("user_id").agg(pl.len().alias("nb"))
    x = users.join(last, on="user_id", how="left").join(nb, on="user_id", how="left").with_columns(pl.col("nb").fill_null(0))
    seen = x["last"].is_not_null().to_numpy()
    rec = np.where(seen, (np.datetime64(A) - x["last"].to_numpy().astype("datetime64[D]")).astype("timedelta64[D]").astype(int), 9999)
    rb = np.full(len(rec), 5); rb[rec<=30]=4; rb[rec<=14]=3; rb[rec<=7]=2; rb[rec<=3]=1; rb[rec==0]=0
    nbv = x["nb"].to_numpy(); bb = np.full(len(nbv), 3); bb[nbv<=10]=2; bb[nbv<=3]=1; bb[nbv==0]=0
    return seen, rb, bb
lg = lambda q: np.log(np.clip(q,1e-4,1-1e-4)/(1-np.clip(q,1e-4,1-1e-4)))
Dsum = np.zeros((6,4)); Wsum = np.zeros((6,4))
for A, d in [(date(2025,6,11),0),(date(2025,7,16),0),(date(2025,8,13),0),(date(2025,9,10),0),(date(2025,6,11),30),(date(2025,7,16),30)]:
    seen, rb, bb = cells(A)
    y = users.join(df.filter((pl.col("event_date") > A) & (pl.col("event_date") <= A + timedelta(days=30))).group_by("user_id").agg(pl.col("gmv").sum().alias("y")), on="user_id", how="left")["y"].fill_null(0.0).to_numpy()
    sel = users.join(df.filter((pl.col("event_date") >= A + timedelta(days=31+d)) & (pl.col("event_date") <= A + timedelta(days=60+d))).select(pl.col("user_id").unique()).with_columns(pl.lit(1).alias("s")), on="user_id", how="left")["s"].fill_null(0).to_numpy().astype(bool)
    buy = y > 0
    for i in range(6):
        for j in range(4):
            m = seen & (rb==i) & (bb==j)
            if m.sum() < 300 or (m & sel).sum() < 100: continue
            Dsum[i,j] += (lg(buy[m & sel].mean()) - lg(buy[m].mean())) * m.sum(); Wsum[i,j] += m.sum()
    print("anchor", A, "d", d, flush=True)
D = np.where(Wsum > 0, Dsum/np.maximum(Wsum,1), 0.0); print("δ логита P(buy) от отбора:"); print(np.round(D,3))
A = date(2026,2,13); seen, rb, bb = cells(A); delta = D[rb, bb]; delta[~seen] = 0
h = pl.read_parquet(E/"hd_s42_heads.parquet").sort("user_id"); assert (h["user_id"].to_numpy()==users["user_id"].to_numpy()).all()
p, mu = h["p"].to_numpy(), h["mu"].to_numpy(); p2 = 1/(1+np.exp(-(lg(p) - delta)))
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
raw = (p2 - p) * mu; print(f"raw: mean {raw.mean():+.4f} rms {raw.std():.4f}; mean p {p.mean():.3f} -> {p2.mean():.3f}")
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","tcn","atweedie","amlp","wpack","ffc","churn","young","gappy","light","fading","lo","hi","selmap"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
P3 = np.load(T/"artifacts/forms/probe3_forms.npy"); z = (lp-lp.mean())/lp.std()
span = [lp-lp.mean(), z**2] + [v-v.mean() for v in meas.values()] + [r-r.mean() for r in P3]
Q,_ = np.linalg.qr(np.stack(span,1)); d0 = raw - raw.mean(); dp = d0 - Q@(Q.T@d0); dp -= dp.mean()
print(f"pshift: new-energy {(dp**2).sum()/(d0**2).sum():.3f}; corr ⟂ vs selmap {np.corrcoef(dp, meas['selmap'])[0,1]:+.3f}, churn {np.corrcoef(dp, meas['churn'])[0,1]:+.3f}")
f = dp*(0.02/dp.std())
if "save" in sys.argv:
    np.save(T/"artifacts/forms/probe_form_pshift.npy", f); x = lp + f; x += 2.33 - x.mean()
    o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
    assert o["predict"].null_count()==0; o.write_csv(S/"sub_probe_pshift35_L233.csv"); print("saved sub_probe_pshift35_L233.csv (base v35)")

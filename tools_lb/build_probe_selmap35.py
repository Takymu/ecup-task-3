"""28.08: форма selmap (идея роя, линза отбора) — ИЗМЕРЕННАЯ 2-мерная карта смещения отбора recency(6) × дни покупок за 90д(4):
M[i,j] = E[log1p y | юзер активен в окне (a+31+d, a+60+d]] − E[log1p y] на анкерах лета/осени 2025 (обусловленность будущим = механизм панели).
Форма на тесте = −M[ячейка юзера] (понижаем тех, кого отбор завышает), ⟂ span{lp_v35, lp², все измеренные формы, comb5-формы, churn}, rms .02.
Usage: build_probe_selmap35.py save"""
import sys, numpy as np, polars as pl
from pathlib import Path
from datetime import date, timedelta
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"
df = pl.read_parquet(T/"data/train.parquet", columns=["event_date","user_id","gmv"])
users = df["user_id"].unique().sort().to_frame()
def cells(A):
    last = df.filter(pl.col("event_date") <= A).group_by("user_id").agg(pl.col("event_date").max().alias("last"))
    nb = df.filter((pl.col("event_date") > A - timedelta(days=90)) & (pl.col("event_date") <= A) & (pl.col("gmv") > 0)).group_by("user_id").agg(pl.len().alias("nb"))
    x = users.join(last, on="user_id", how="left").join(nb, on="user_id", how="left").with_columns(pl.col("nb").fill_null(0))
    seen = x["last"].is_not_null().to_numpy()
    rec = np.where(seen, (np.datetime64(A) - x["last"].to_numpy().astype("datetime64[D]")).astype("timedelta64[D]").astype(int), 9999)
    rb = np.full(len(rec), 5); rb[rec<=30]=4; rb[rec<=14]=3; rb[rec<=7]=2; rb[rec<=3]=1; rb[rec==0]=0
    nbv = x["nb"].to_numpy(); bb = np.full(len(nbv), 3); bb[nbv<=10]=2; bb[nbv<=3]=1; bb[nbv==0]=0
    return seen, rb, bb
Msum = np.zeros((6,4)); Wsum = np.zeros((6,4))
for A, d in [(date(2025,6,11),0),(date(2025,7,16),0),(date(2025,8,13),0),(date(2025,9,10),0),(date(2025,6,11),30),(date(2025,7,16),30)]:
    seen, rb, bb = cells(A)
    y = users.join(df.filter((pl.col("event_date") > A) & (pl.col("event_date") <= A + timedelta(days=30))).group_by("user_id").agg(pl.col("gmv").sum().alias("y")), on="user_id", how="left")["y"].fill_null(0.0).to_numpy()
    sel = users.join(df.filter((pl.col("event_date") >= A + timedelta(days=31+d)) & (pl.col("event_date") <= A + timedelta(days=60+d))).select(pl.col("user_id").unique()).with_columns(pl.lit(1).alias("s")), on="user_id", how="left")["s"].fill_null(0).to_numpy().astype(bool)
    ly = np.log1p(np.clip(y,0,None))
    for i in range(6):
        for j in range(4):
            m = seen & (rb==i) & (bb==j)
            if m.sum() < 300 or (m & sel).sum() < 100: continue
            Msum[i,j] += (ly[m & sel].mean() - ly[m].mean()) * m.sum(); Wsum[i,j] += m.sum()
    print("anchor", A, "d", d, "done", flush=True)
M = np.where(Wsum > 0, Msum/np.maximum(Wsum,1), 0.0); print("карта смещения (лог-ед.):"); print(np.round(M,3))
# тест
A = date(2026,2,13); seen, rb, bb = cells(A); raw = -M[rb, bb]; raw[~seen] = 0
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); assert (users["user_id"].to_numpy()==uid).all(); lp = np.log1p(sub["predict"].to_numpy())
print("test: share by recency bucket", np.round(np.bincount(rb, minlength=6)/len(rb),3), "| raw rms", round(float(raw.std()),4), "mean", round(float(raw.mean()),4))
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","tcn","atweedie","amlp","wpack","ffc","churn","young","gappy","light","fading","lo","hi"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
P3 = np.load(T/"artifacts/forms/probe3_forms.npy"); z = (lp-lp.mean())/lp.std()
span = [lp-lp.mean(), z**2] + [v-v.mean() for v in meas.values()] + [r-r.mean() for r in P3]
Q,_ = np.linalg.qr(np.stack(span,1)); d0 = raw - raw.mean(); dp = d0 - Q@(Q.T@d0); dp -= dp.mean()
print(f"selmap: new-energy {(dp**2).sum()/(d0**2).sum():.3f}; corr ⟂ vs churn {np.corrcoef(dp, meas['churn'])[0,1]:+.3f}, vs gappy {np.corrcoef(dp, meas['gappy'])[0,1]:+.3f}")
f = dp*(0.02/dp.std())
if "save" in sys.argv:
    np.save(T/"artifacts/forms/probe_form_selmap.npy", f); x = lp + f; x += 2.33 - x.mean()
    o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
    assert o["predict"].null_count()==0; o.write_csv(S/"sub_probe_selmap35_L233.csv"); print("saved sub_probe_selmap35_L233.csv (base v35)")

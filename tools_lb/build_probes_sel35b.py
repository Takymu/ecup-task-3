"""27.08: ещё две формы «склонности уходить» (механизм отбора панели) на базе v35: light = 1[<=3 активных дней за последние 30],
fading = 1[активность за 30д < половины темпа предыдущих 60д]. Знак −. Ортогонализация как в build_probes_sel35.py (+young/gappy). Usage: ... save [names]"""
import sys, numpy as np, polars as pl
from pathlib import Path
from datetime import date, timedelta
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"; A = date(2026,2,13)
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","tcn","atweedie","amlp","wpack","ffc","lo","hi","young","gappy"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
P3 = np.load(T/"artifacts/forms/probe3_forms.npy")
df = pl.read_parquet(T/"data/train.parquet", columns=["event_date","user_id"])
g = df.group_by("user_id").agg((pl.col("event_date")>A-timedelta(30)).sum().alias("a30"), ((pl.col("event_date")>A-timedelta(90))&(pl.col("event_date")<=A-timedelta(30))).sum().alias("a3090"))
x = pl.DataFrame({"user_id":uid}).join(g, on="user_id", how="left").fill_null(0)
a30 = x["a30"].to_numpy(); tr = a30/(x["a3090"].to_numpy()/2+0.5)
raw = {"light": -(a30<=3).astype(float), "fading": -(tr<0.5).astype(float)}
for n,d in raw.items(): print(f"{n}: share {np.mean(d<0):.3f}  mean lp in seg {lp[d<0].mean():.3f}")
z = (lp-lp.mean())/lp.std(); span = [lp-lp.mean(), z**2] + [v-v.mean() for v in meas.values()] + [r-r.mean() for r in P3]
Q,_ = np.linalg.qr(np.stack(span,1)); out = {}
for n, d in raw.items():
    d = d-d.mean(); dp = d - Q@(Q.T@d); dp -= dp.mean(); print(f"{n}: new-energy {(dp**2).sum()/(d**2).sum():.3f}"); out[n] = dp*(0.02/dp.std())
print("corr light/fading:", round(float(np.corrcoef(out["light"],out["fading"])[0,1]),3))
if "save" in sys.argv:
    for n in sys.argv[2:]:
        f = out[n]; np.save(T/f"artifacts/forms/probe_form_{n}.npy", f); xx = lp + f; xx += 2.33 - xx.mean()
        o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(xx,0,None)),0,None)}), on="user_id", how="left")
        assert o["predict"].null_count()==0; o.write_csv(S/f"sub_probe_{n}35_L233.csv"); print(f"saved sub_probe_{n}35_L233.csv (base v35)")

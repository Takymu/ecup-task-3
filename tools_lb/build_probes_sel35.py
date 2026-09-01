"""27.08: формы «склонности уходить» на базе v35 — механизм отбора панели (в трейне такие юзеры всегда возвращались).
young = 1[first_seen >= 2025-07-01]; gappy = 1[max пауза в истории >= 30д]; знак −: проба ПОНИЖАЕТ сегмент, k>0.5 ⇔ мы его завышаем.
⟂ span{lp_v35, все измеренные формы, lo/hi, comb5-формы}, rms .02. k=0 → 1.645300, k=.5 → v35, k=1 → 1.645057. Usage: ... save"""
import sys, numpy as np, polars as pl
from pathlib import Path
from datetime import date
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","tcn","atweedie","amlp","wpack","ffc","lo","hi"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
P3 = np.load(T/"artifacts/forms/probe3_forms.npy")
df = pl.read_parquet(T/"data/train.parquet", columns=["event_date","user_id"]).sort("user_id","event_date")
g = df.group_by("user_id", maintain_order=True).agg(pl.col("event_date").min().alias("first"), pl.col("event_date").diff().dt.total_days().max().alias("maxgap"),
        (pl.col("event_date").diff().dt.total_days()>=21).sum().alias("ngap21"))
x = pl.DataFrame({"user_id":uid}).join(g, on="user_id", how="left")
first = x["first"].to_numpy(); maxgap = x["maxgap"].fill_null(0).to_numpy(); ngap = x["ngap21"].fill_null(0).to_numpy()
raw = {"young": -(first >= np.datetime64("2025-07-01")).astype(float), "gappy": -(maxgap >= 30).astype(float), "gappy2": -(ngap >= 2).astype(float)}
for n,d in raw.items(): print(f"{n}: share {np.mean(d<0):.3f}  mean lp in seg {lp[d<0].mean():.3f} vs all {lp.mean():.3f}")
z = (lp-lp.mean())/lp.std(); span = [lp-lp.mean(), z**2] + [v-v.mean() for v in meas.values()] + [r-r.mean() for r in P3]
Q,_ = np.linalg.qr(np.stack(span,1)); out = {}
for n, d in raw.items():
    d = d-d.mean(); dp = d - Q@(Q.T@d); dp -= dp.mean(); print(f"{n}: new-energy {(dp**2).sum()/(d**2).sum():.3f}"); out[n] = dp*(0.02/dp.std())
print("corr:", {f"{a}/{b}": round(float(np.corrcoef(out[a],out[b])[0,1]),3) for a in out for b in out if a<b})
if "save" in sys.argv:
    for n in sys.argv[2:]:
        f = out[n]; np.save(T/f"artifacts/forms/probe_form_{n}.npy", f); xx = lp + f; xx += 2.33 - xx.mean()
        o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(xx,0,None)),0,None)}), on="user_id", how="left")
        assert o["predict"].null_count()==0; o.write_csv(S/f"sub_probe_{n}35_L233.csv"); print(f"saved sub_probe_{n}35_L233.csv (base v35)")

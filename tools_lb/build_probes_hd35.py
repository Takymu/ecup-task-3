"""29.08: пробы hd-семейства (аугментация длины истории, стенд V/P3: full +1.01/+1.10, L10 +1.34/+1.29, L16 +1.21/+1.05) на базе v35.
Направления = log1p(testpred) − lp_v35 (stale-модель hd_s42 @13.02: полный вид и виды с усечённой историей L10/L16), ⟂ span всего измеренного.
Формы: hd10 (вид L10), hdfam (full+L10+L16, сумма ⟂-частей). Usage: build_probes_hd35.py save [hd10 hdfam]"""
import sys, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"; E = T/"artifacts/models"
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","tcn","atweedie","amlp","wpack","ffc","churn","young","gappy","light","fading","lo","hi","selmap","pshift"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
P3 = np.load(T/"artifacts/forms/probe3_forms.npy"); z = (lp-lp.mean())/lp.std()
span = [lp-lp.mean(), z**2] + [v-v.mean() for v in meas.values()] + [r-r.mean() for r in P3]
Q,_ = np.linalg.qr(np.stack(span,1))
def lpt(fn):
    d = pl.read_parquet(E/fn).sort("user_id"); assert (d["user_id"].to_numpy()==uid).all(); return np.log1p(np.clip(d["pred"].to_numpy(),0,None))
views = {"full": lpt("hd_s42_heads0_testpred.parquet"), "L10": lpt("hd_s42_T10_testpred.parquet"), "L16": lpt("hd_s42_T16_testpred.parquet")}
perp = {}
for n, v in views.items():
    d = v - lp; d -= d.mean(); dp = d - Q@(Q.T@d); dp -= dp.mean(); perp[n] = dp
    print(f"{n:5s} raw rms {d.std():.4f} new-energy {(dp**2).sum()/(d**2).sum():.3f}")
print("corr ⟂ views:", {f"{a}/{b}": round(float(np.corrcoef(perp[a],perp[b])[0,1]),3) for a in perp for b in perp if a<b})
# L10 = дубль измеренных sm10/sm8/mspec (новая энергия 6%) — не проба; формы: hd (полный вид, .30), hd16 (.46), hdf2 = hd+hd16
out = {"hd": perp["full"]*(0.02/perp["full"].std()), "hd16": perp["L16"]*(0.02/perp["L16"].std())}
fam = perp["full"]/perp["full"].std() + perp["L16"]/perp["L16"].std(); fam -= fam.mean(); out["hdf2"] = fam*(0.02/fam.std())
print("corr hd/hd16", round(float(np.corrcoef(out["hd"], out["hd16"])[0,1]),3))
if "save" in sys.argv:
    for n in (sys.argv[2:] or list(out)):
        f = out[n]; np.save(T/f"artifacts/forms/probe_form_{n}.npy", f); x = lp + f; x += 2.33 - x.mean()
        o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
        assert o["predict"].null_count()==0; o.write_csv(S/f"sub_probe_{n}35_L233.csv"); print(f"saved sub_probe_{n}35_L233.csv (base v35)")

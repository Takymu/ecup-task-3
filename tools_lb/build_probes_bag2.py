"""30.08 (финальный день): де-нойзинг тяжёлых форм, часть 2 — f3s3 (бэг 3 рефит-сидов f3s, corr сида с бэгом 0.828 → ~30% шума в
векторе с k=1.145) и f3saux4 (бэг 4 рефит-сидов, когда доучатся). Формы как aux4: ⟂ span{lp_v35, z², все измеренные КРОМЕ заменяемой,
P3-формы}; пробы на базе v39 (нуль 1.644558, k=.5 ⇔ 1.644436). Usage: build_probes_bag2.py [save] [f3s3 f3saux4]"""
import sys, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"; E = T/"artifacts/models"
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","tcn","atweedie","amlp","wpack","ffc","churn","young","gappy","light","fading","lo","hi","selmap","pshift","hd","hd16","auxcnt","ms24","pm2","segxk","aux4"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
P3 = np.load(T/"artifacts/forms/probe3_forms.npy"); z = (lp-lp.mean())/lp.std()
def lpt(fn):
    d = pl.read_parquet(E/fn).sort("user_id"); assert (d["user_id"].to_numpy()==uid).all(); return np.log1p(np.clip(d["pred"].to_numpy(),0,None))
def clean_bag(files, excl):
    sp = [lp-lp.mean(), z**2] + [v-v.mean() for n2, v in meas.items() if n2 not in excl] + [r-r.mean() for r in P3]
    Q,_ = np.linalg.qr(np.stack(sp,1))
    d = np.mean([lpt(f) for f in files],0) - lp; d -= d.mean(); d = d - Q@(Q.T@d); d -= d.mean()
    return d * (0.02/d.std())
out = {}
out["f3s3"] = clean_bag([f"f3s_{s}_full_avg2_testpred.parquet" for s in ("s42","s1","s2")], {"f3s"})
print("f3s3: corr с формой f3s =", round(float(np.corrcoef(out["f3s3"], meas["f3s"]/meas["f3s"].std())[0,1]),3))
if "f3saux4" in sys.argv:
    out["f3saux4"] = clean_bag([f"f3saux_{s}_full_avg2_testpred.parquet" for s in ("s42","s1","s2","s3")], {"f3saux"})
    print("f3saux4: corr с формой f3saux =", round(float(np.corrcoef(out["f3saux4"], meas["f3saux"]/meas["f3saux"].std())[0,1]),3),
          "| corr(один сид, бэг) =", round(float(np.corrcoef(clean_bag(["f3saux_s42_full_avg2_testpred.parquet"], {"f3saux"}), out["f3saux4"])[0,1]),3))
if "save" in sys.argv:
    v39 = pl.read_csv(S/"sub_v39_L233.csv").sort("user_id"); lp39 = np.log1p(v39["predict"].to_numpy())
    for n, f in out.items():
        np.save(T/f"artifacts/forms/probe_form_{n}.npy", f); x = lp39 + f; x += 2.33 - x.mean()
        o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
        assert o["predict"].null_count()==0; o.write_csv(S/f"sub_probe_{n}39_L233.csv"); print(f"saved sub_probe_{n}39_L233.csv (base v39)")

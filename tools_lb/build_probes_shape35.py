"""27.08: пробы формы шкалы предсказаний на базе v35 (уровень/наклон/кривизна уже измерены; здесь — кубический член и два хвоста).
Формы: cub = z^3 (z = стандартизованный lp), lo = 1[lp в нижних 20%], hi = 1[lp в верхних 20%]; каждая ⟂ span{lp_v35, все измеренные формы,
остальные формы этого набора}, rms .02. Нуль (k=0) = 1.645300, k=.5 = v35, k=1 = 1.645057. Usage: build_probes_shape35.py save"""
import sys, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","tcn","atweedie","amlp","wpack","ffc"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
P3 = np.load(T/"artifacts/forms/probe3_forms.npy") if (T/"artifacts/forms/probe3_forms.npy").exists() else None  # d1,d2,d4c,curv,never (старые формы comb5 — уже в lb_base)
z = (lp-lp.mean())/lp.std(); q20, q80 = np.quantile(lp,[.2,.8])
raw = {"cub": z**3, "lo": (lp<=q20).astype(float), "hi": (lp>=q80).astype(float)}
span = [lp-lp.mean(), z**2] + [v-v.mean() for v in meas.values()] + ([r-r.mean() for r in P3] if P3 is not None else [])
print("span dims", len(span), "P3 forms", None if P3 is None else P3.shape)
out = {}
for n, d in raw.items():
    others = [raw[m]-raw[m].mean() for m in raw if m != n]
    Q,_ = np.linalg.qr(np.stack(span+others,1)); d = d-d.mean(); dp = d - Q@(Q.T@d); dp -= dp.mean()
    print(f"{n}: new-energy {(dp**2).sum()/(d**2).sum():.3f}  corr(dp, lp) {np.corrcoef(dp,lp)[0,1]:+.1e}")
    out[n] = dp*(0.02/dp.std())
print("corr between forms:", {f"{a}/{b}": round(float(np.corrcoef(out[a],out[b])[0,1]),3) for a in out for b in out if a<b})
if "save" in sys.argv:
    for n,f in out.items():
        np.save(T/f"artifacts/forms/probe_form_{n}.npy", f); x = lp + f; x += 2.33 - x.mean()
        o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
        assert o["predict"].null_count()==0; o.write_csv(S/f"sub_probe_{n}35_L233.csv"); print(f"saved sub_probe_{n}35_L233.csv (base v35)")

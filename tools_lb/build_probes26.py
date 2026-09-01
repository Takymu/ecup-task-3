"""26.08 ночь: пробы на базе alstm (sub_probe_alstm_L233, ЛБ 1.6452318). span = lp_alstm + все измеренные формы + ожидающие (auemb/tcn/atweedie).
amlp = −(бэг 5 сидов tab-MLP − lp) ⟂ span (стенд вал k⟂ +.75, P3 нет); wpack = (auemb+tcn+atweedie+amlp)/2 — сумма четырёх ⟂ форм rms .02:
если хотя бы половина реальна (k_i≈.7), k_pack≈1.4 → ~1.64501. Usage: build_probes26.py save"""
import sys, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[1]; E = T/"artifacts/models"; S = T/"artifacts/submissions"
sub = pl.read_csv(S/"sub_probe_alstm_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","tcn","atweedie"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
Q,_ = np.linalg.qr(np.stack([lp-lp.mean()] + [v-v.mean() for v in meas.values()],1))
def lpt(fn):
    d = pl.read_parquet(E/fn).sort("user_id"); assert (d["user_id"].to_numpy()==uid).all(); return np.log1p(np.clip(d["pred"].to_numpy(),0,None))
mlp = np.mean([lpt("l_tab_mlp_a16_full_testpred.parquet")]+[lpt(f"l_tab_mlp_a16_s{s}_full_testpred.parquet") for s in (1,2,3,4)],0)
d = -(mlp-lp); d -= d.mean(); dp = d - Q@(Q.T@d); dp -= dp.mean()
print(f"amlp: rms raw {d.std():.4f} new-energy {(dp**2).sum()/(d**2).sum():.3f}")
out = {"amlp": dp*(0.02/dp.std())}
pack = sum(meas[n] for n in ("auemb","tcn","atweedie")) + out["amlp"]; pack -= pack.mean(); print("pack rms before norm", pack.std().round(4), "(4 ⟂ форм по .02 → .04)")
out["wpack"] = pack*(0.02/pack.std())
print("corr amlp vs auemb/tcn/atweedie:", [round(float(np.corrcoef(out["amlp"],meas[n])[0,1]),3) for n in ("auemb","tcn","atweedie")])
if "save" in sys.argv:
    for n,f in out.items():
        np.save(T/f"artifacts/forms/probe_form_{n}.npy", f)
        x = lp + f; x += 2.33 - x.mean()
        o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
        assert o["predict"].null_count()==0; o.write_csv(S/f"sub_probe_{n}_L233.csv"); print("saved sub_probe_%s_L233.csv (base alstm)"%n)

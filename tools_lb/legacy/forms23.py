"""23.08: формы на базе v32. span = lp32 + все измеренные формы (включая sres42/sm10/f3s/sm8).
Кандидаты: f3sb (бэг f3s 3 сида), sm10b (бэг 3 сида), sm12, sm8b. Usage: forms23.py [save:name ...]"""
import sys, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[2]; E = T/"artifacts/models"; S = T/"artifacts/submissions"
sub = pl.read_csv(S/"sub_v32_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp32 = np.log1p(sub["predict"].to_numpy())
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8"]: meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
Q,_ = np.linalg.qr(np.stack([lp32-lp32.mean()] + [v-v.mean() for v in meas.values()],1))
def lpt(s):
    d = pl.read_parquet(E/f"{s}_full_avg2_testpred.parquet").sort("user_id"); assert (d["user_id"].to_numpy()==uid).all(); return np.log1p(np.clip(d["pred"].to_numpy(),0,None))
cands = {"f3sb": np.mean([lpt(f"f3s_s{s}") for s in (42,1,2)],0) - lp32,
         "sm10b": np.mean([lpt(f"sm10_s{s}") for s in (42,1,2)],0) - lp32,
         "sm8b": np.mean([lpt(f"sm8_s{s}") for s in (42,1)],0) - lp32,
         "sm12": lpt("sm12_s42") - lp32}
out = {}
for n,d in cands.items():
    d = d-d.mean(); dp = d - Q@(Q.T@d); dp -= dp.mean(); ne = float((dp**2).sum()/(d**2).sum())
    print(f"{n:6s} rms raw {d.std():.4f} new-energy {ne:.3f}  corr(raw: f3s {np.corrcoef(d,meas['f3s'])[0,1]:+.2f} sm10 {np.corrcoef(d,meas['sm10'])[0,1]:+.2f} mspec {np.corrcoef(d,meas['mspec'])[0,1]:+.2f})")
    out[n] = dp*(0.02/dp.std())
names=list(out); print(names); print(np.round(np.corrcoef(np.stack([out[n] for n in names])),2))
for a in sys.argv[1:]:
    n = a.split(":")[1]; np.save(T/f"artifacts/forms/probe_form_{n}.npy", out[n])
    x = lp32 + out[n]; x += 2.33 - x.mean()
    o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
    assert o["predict"].null_count()==0; o.write_csv(S/f"sub_probe_{n}_L233.csv"); print("saved probe", n, "(base v32)")

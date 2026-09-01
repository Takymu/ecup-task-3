"""30.08 финал: бэг-20 и «лучший сид» для форм hd (stale) и hd16 (вид L16) — для сборки двух финальных решений.
Печатает грязность (corr один сид ↔ бэг20) и сохраняет probe_form_{hd,hd16}_{bag20,best}.npy."""
import os, re
import numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"; E = T/"artifacts/models"
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy")
v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy())
dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","atweedie","wpack","ffc","churn","young","gappy","light","fading","lo","hi","selmap","pshift","hd","hd16","auxcnt","ms24","pm2","segxk","segslope","segcarr","ebshrink"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
P3 = np.load(T/"artifacts/forms/probe3_forms.npy"); z = (lp-lp.mean())/lp.std()
def lpt(f):
    d = pl.read_parquet(E/f).sort("user_id"); assert (d["user_id"].to_numpy()==uid).all()
    return np.log1p(np.clip(d["pred"].to_numpy(),0,None))
for form, suf in (("hd", "_stale_testpred.parquet"), ("hd16", "_T16_testpred.parquet")):
    pat = re.compile(rf"^hd_s(\d+){re.escape(suf)}$")
    files = sorted((f for f in os.listdir(E) if pat.match(f)), key=lambda f: int(pat.match(f).group(1)))
    sp = [lp-lp.mean(), z**2] + [v-v.mean() for n2, v in meas.items() if n2 not in ("hd","hd16")] + [r-r.mean() for r in P3]
    Q,_ = np.linalg.qr(np.stack(sp,1))
    def cl(v):
        v = v - v.mean(); v = v - Q@(Q.T@v); return v - v.mean()
    vecs = [lpt(f) for f in files]
    bag = cl(np.mean(vecs,0) - lp)
    singles = [cl(v - lp) for v in vecs]
    proj = [float(np.dot(x,bag)/np.linalg.norm(x)/np.linalg.norm(bag)) for x in singles]
    bi = int(np.argmax(proj))
    print(f"{form:5s} сидов {len(vecs):2d} | corr(s42, бэг20) {np.corrcoef(singles[-1] if files[-1].startswith('hd_s42') else singles[0], bag)[0,1]:.3f} | лучший {files[bi].split('_')[1]} (proj {proj[bi]:.3f})")
    np.save(T/f"artifacts/forms/probe_form_{form}_bag20.npy", bag*(0.02/bag.std()))
    np.save(T/f"artifacts/forms/probe_form_{form}_best.npy", singles[bi]*(0.02/singles[bi].std()))

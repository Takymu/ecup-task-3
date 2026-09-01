"""30.08 финал: пробы на бэгах-20 для трёх грязных форм (f3saux k=1.10, f3s k=1.145, auxcnt k=0.81).
Для каждой семьи: форма = ⟂-часть (бэг всех сидов − lp_v35) к span{lp, z², все измеренные КРОМЕ своей, P3};
печатает corr(бэг4, бэг20) — диагностика «хватило ли 20», и corr(лучший сид, бэг). Пробы на базе v39.
Usage: build_probes_bag20.py [save]"""
import sys, glob, os, re
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
def lpt(fn):
    d = pl.read_parquet(E/fn).sort("user_id"); assert (d["user_id"].to_numpy()==uid).all()
    return np.log1p(np.clip(d["pred"].to_numpy(),0,None))
FAM = {"f3saux4": ("f3saux", "_full_avg2_testpred.parquet", "f3saux"),
       "f3s3":    ("f3s",    "_full_avg2_testpred.parquet", "f3s"),
       "aux4":    ("auxcnt", "_stale_testpred.parquet",     "auxcnt")}
out, best_seed = {}, {}
for probe, (fam, suf, form) in FAM.items():
    pat = re.compile(rf"^{fam}_s(\d+){re.escape(suf)}$")
    files = sorted((f for f in os.listdir(E) if pat.match(f)), key=lambda f: int(pat.match(f).group(1)))
    files = [str(E/f) for f in files]
    sp = [lp-lp.mean(), z**2] + [v-v.mean() for n2, v in meas.items() if n2 != form] + [r-r.mean() for r in P3]
    Q,_ = np.linalg.qr(np.stack(sp,1))
    def cl(v):
        v = v - v.mean(); v = v - Q@(Q.T@v); return v - v.mean()
    vecs = [lpt(os.path.basename(f)) for f in files]
    bag20 = cl(np.mean(vecs,0) - lp)
    bag4  = cl(np.mean(vecs[:4],0) - lp)
    proj = [float(np.dot(cl(v - lp), bag20)/np.linalg.norm(cl(v - lp))/np.linalg.norm(bag20)) for v in vecs]
    bi = int(np.argmax(proj)); best_seed[probe] = (os.path.basename(files[bi]), proj[bi])
    print(f"{probe:8s} сидов {len(vecs):2d} | corr(бэг4,бэг20) {np.corrcoef(bag4,bag20)[0,1]:.3f} | "
          f"corr(один сид,бэг20) {np.corrcoef(cl(vecs[0]-lp),bag20)[0,1]:.3f} | лучший сид {os.path.basename(files[bi]).split('_full')[0].split('_stale')[0]} (proj {proj[bi]:.3f})")
    out[probe] = bag20 * (0.02/bag20.std())
    np.save(T/f"artifacts/forms/probe_form_{probe}_bag20.npy", out[probe])
    np.save(T/f"artifacts/forms/probe_form_{probe}_best.npy", (lambda w: w*(0.02/w.std()))(cl(vecs[bi] - lp)))
if "save" in sys.argv:
    v39 = pl.read_csv(S/"sub_v39_L233.csv").sort("user_id"); lp39 = np.log1p(v39["predict"].to_numpy())
    for probe, f in out.items():
        np.save(T/f"artifacts/forms/probe_form_{probe}.npy", f)
        x = lp39 + f; x += 2.33 - x.mean()
        o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(
            pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
        assert o["predict"].null_count()==0
        o.write_csv(S/f"sub_probe_{probe}39_L233.csv"); print(f"saved sub_probe_{probe}39_L233.csv (bag20, base v39)")

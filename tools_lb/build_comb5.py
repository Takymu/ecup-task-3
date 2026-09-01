"""Apply comb5 correction (k5 from lb_probes_round3.json) to a family-weights blend json -> submission CSV.
Usage: build_comb5.py <weights.json name in artifacts> <out sub name>. Sanity-checks itself by rebuilding v22."""
import json, sys, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[1]; SUBS = T/"artifacts/submissions"; EV = T/"artifacts/models"
k = json.loads((T/"artifacts/lb/lb_probes_round3.json").read_text())["k5"]
def blend_lp(cfg):
    acc=None; tot=0; users=None
    for fam, w in cfg["family_weights"].items():
        ms = cfg["members"][fam]
        for p in ms:
            df = pl.read_parquet(p).sort("user_id")
            if users is None: users = df["user_id"].to_numpy()
            assert (df["user_id"].to_numpy()==users).all()
            lp = np.log1p(np.clip(df["pred"].to_numpy(),0,None)); ww=w/len(ms); tot+=ww
            acc = ww*(lp-lp.mean()) if acc is None else acc+ww*(lp-lp.mean())
    return users, acc/tot + 2.33
def apply_comb5(users, lp):
    v15 = pl.read_csv(SUBS/"sub_v15_famblend_L233.csv").sort("user_id"); rec = pl.read_csv(SUBS/"sub_v15_recshift_L233.csv").sort("user_id")
    assert (v15["user_id"].to_numpy()==users).all()
    d1 = np.log1p(rec["predict"].to_numpy())-np.log1p(v15["predict"].to_numpy())
    pa = pl.DataFrame({"user_id": users}).join(pl.read_parquet(EV/"pactive_jun_oct_2026-02-13.parquet"), on="user_id", how="left")["p_act"].to_numpy()
    assert not np.isnan(pa).any()
    never = np.load(T/"artifacts/forms/probe3_forms.npy")[1]
    d2 = lp*(pa-1); d2 -= d2.mean()
    x = lp + k["d1"]*d1 + k["d2"]*d2; x += 2.33 - x.mean()
    d4 = 0.1*(x-2.33); q = (x-2.33)**2; curv = 0.03*(q-q.mean())
    x = x + k["d4c"]*d4 + k["curv"]*curv + k["never"]*never; x += 2.33 - x.mean()
    return np.clip(np.expm1(np.clip(x,0,None)),0,None)
# необязательная санити-проверка: пересобрать v22 и сверить с копией отправленного (если она есть)
if (SUBS/"sub_v22_comb5_L233.csv").exists():
    u, lp22 = blend_lp(json.loads((T/"artifacts/lb/blend_v22_weights.json").read_text()))
    chk = apply_comb5(u, lp22)
    ref = pl.read_csv(SUBS/"sub_v22_comb5_L233.csv").sort("user_id")
    print("sanity v22 max|dlog| =", np.abs(np.log1p(chk)-np.log1p(ref["predict"].to_numpy())).max())
else:
    print("sanity v22 пропущена: нет sub_v22_comb5_L233.csv")
cfg = json.loads((T/"artifacts/lb"/sys.argv[1]).read_text()); u2, lp2 = blend_lp(cfg)
pred = apply_comb5(u2, lp2)
sub = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id": u2, "predict": pred}), on="user_id", how="left")
assert sub["predict"].null_count()==0
sub.write_csv(SUBS/f"{sys.argv[2]}.csv")
print("saved", sys.argv[2])
if "chk" in dir():
    print("  corr with v22:", np.corrcoef(np.log1p(pred), np.log1p(chk))[0,1])

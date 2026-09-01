"""29.08 (рой r3, segxk): интеракции измеренных направлений с сегментами маргинальности членства в панели — 0 GPU.
Сегменты (окно отбора (14.01, 13.02]): marg = ≤3 активных дня, dense = ≥15, dorm = recency ≥15 на 13.02. Носители: ctx, churn, auxcnt, hd16. Пакет со знаками +marg −dense +dorm.
Каждая интеракция F_c·(1[seg]−mean) ⟂ span{lp_v35, z², все измеренные формы, P3-формы}; печатается новая энергия; пакет = сумма ⟂-частей.
Usage: build_probe_segxk.py [save]  → probe_form_segxk.npy + sub_probe_segxk38_L233.csv (база v38)"""
import sys, numpy as np, polars as pl
from datetime import date
from pathlib import Path
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id"); lb_base = np.log1p(base["predict"].to_numpy())
F2 = np.load(T/"artifacts/forms/probe_forms2.npy"); v27 = np.log1p(pl.read_csv(S/"sub_v27_comb5_L233.csv").sort("user_id")["predict"].to_numpy()); dec = np.log1p(pl.read_csv(S/"sub_probe_decon_L233.csv").sort("user_id")["predict"].to_numpy())
meas = {"ctx":F2[0], "rescorr":np.load(T/"artifacts/forms/probe_form_rescorr.npy"), "v27":v27-lb_base, "decon":dec-v27}
for n in ["pxc","pcorr","gift","visc","boost","ctxv2","mspec","sres42","sm10","f3s","sm8","f3s12","f3saux","f3spring","f3sauxx","distill","alstm","aknn","auemb","tcn","atweedie","amlp","wpack","ffc","churn","young","gappy","light","fading","lo","hi","selmap","pshift","hd","hd16","auxcnt"]:
    meas[n] = np.load(T/f"artifacts/forms/probe_form_{n}.npy")
P3 = np.load(T/"artifacts/forms/probe3_forms.npy"); z = (lp-lp.mean())/lp.std()
span = [lp-lp.mean(), z**2] + [v-v.mean() for v in meas.values()] + [r-r.mean() for r in P3]
Q,_ = np.linalg.qr(np.stack(span,1))
# segments from the selection window
df = pl.scan_parquet(T/"data/train.parquet").select("user_id","event_date").filter(pl.col("event_date") > date(2026,1,14)).collect()
g = df.group_by("user_id").agg(pl.len().alias("nd"), pl.col("event_date").max().alias("last"))
x = pl.DataFrame({"user_id": uid}).join(g, on="user_id", how="left")
nd = x["nd"].fill_null(0).to_numpy(); rec = (np.datetime64("2026-02-13") - x["last"].to_numpy().astype("datetime64[D]")).astype("timedelta64[D]").astype(np.int64)
segs = {"marg": nd <= 3, "dense": nd >= 15, "dorm": rec >= 15}
print({k: round(float(v.mean()),3) for k,v in segs.items()})
carriers = {"ctx": meas["ctx"], "churn": meas["churn"], "auxcnt": meas["auxcnt"], "hd16": meas["hd16"]}  # probe3_forms = curv/never/..., d2 там нет
parts = {}
for cn, F in carriers.items():
    for sn, m in segs.items():
        d = F * (m.astype(float) - m.mean()); d -= d.mean(); dp = d - Q@(Q.T@d); dp -= dp.mean()
        parts[f"{cn}|{sn}"] = dp; print(f"{cn:7s}x{sn:6s} raw rms {d.std():.4f} new-energy {(dp**2).sum()/(d**2).sum():.3f}")
names = list(parts); C = np.corrcoef(np.stack([parts[n]/parts[n].std() for n in names]))
print("max |corr| между интеракциями:", round(float(np.abs(C - np.eye(len(names))).max()),2))
SIGN = {"marg": +1, "dense": -1, "dorm": +1}  # прайор: глобальные k недооценивают маргинальных/дремлющих, переоценивают плотных
pack = sum(SIGN[n.split("|")[1]] * parts[n]/parts[n].std() for n in names); pack -= pack.mean(); pack = pack - Q@(Q.T@pack); pack -= pack.mean(); pack *= 0.02/pack.std()
if "save" in sys.argv:
    np.save(T/"artifacts/forms/probe_form_segxk.npy", pack)
    v38 = pl.read_csv(S/"sub_v38_L233.csv").sort("user_id"); lp38 = np.log1p(v38["predict"].to_numpy())
    xx = lp38 + pack; xx += 2.33 - xx.mean()
    o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(xx,0,None)),0,None)}), on="user_id", how="left")
    assert o["predict"].null_count()==0; o.write_csv(S/"sub_probe_segxk38_L233.csv"); print("saved sub_probe_segxk38_L233.csv (base v38)")

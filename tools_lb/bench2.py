"""Двухпротокольный стенд представлений (воспроизводимая версия того, что 24-25.08 гонялось инлайном).
Протокол V: вал 14.01, база = v15-прокси на валпредах; протокол P3: псевдо-панель 31.12, matched-контроль
0.8*f3p3(s42,s1)+0.2*pz-бленд. Направление кандидата ортогонализуется к базе + решённым семействам
(sm10, sm8, f3s, f3saux, spring, ctx, hyb, lstm(анти), knn) и оценивается k_scaled = k*rms/0.02.
Использование: python bench2.py name=stemV[,stemV2]:stemP3[,..][:anti] ...   (anti => знак минус)
"""
import sys, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[1]; E = T/"artifacts/models"
def lp(fn, key="pred"):
    d = pl.read_parquet(E/fn).sort("user_id"); return np.log1p(np.clip(d[key].to_numpy(),0,None))
def bag(stems, suf="_valpred.parquet"):
    vs=[]
    for s in stems:
        for sf in (suf,"_pa1231_testpred.parquet","_pred.parquet"):
            try: vs.append(lp(f"{s}{sf}")); break
            except Exception: pass
    if not vs: raise FileNotFoundError(stems)
    return np.mean(vs,0)
def resid(y,p,m):
    g=np.linspace(-0.9,0.9,361); d0=g[np.argmin([np.mean((y[m]-np.clip(p[m]+d,0,None))**2) for d in g])]; return y-np.clip(p+d0,0,None)
def perp(d, basis, m):
    d = d - d[m].mean(); B = np.stack([b - b[m].mean() for b in basis],1); return d - B@np.linalg.lstsq(B[m], d[m], rcond=None)[0]
def kst(r, dd, m):
    k = np.dot(r[m],dd[m])/np.dot(dd[m],dd[m]); mm=np.mean(r[m]**2); gain=np.dot(r[m],dd[m])**2/np.dot(dd[m],dd[m])/m.sum()
    return k*dd[m].std()/0.02, np.sqrt(mm)-np.sqrt(max(mm-gain,0)), k
# ---- V
yv = lp("rk_ctrl_valpred.parquet","target"); mv = np.ones(len(yv),bool)
gru = bag(["l_gru_h256x2_a16"]+[f"l_gru_a16_s{i}" for i in (1,2,3,4,5,6)]+["l_gru_a16_c9","l_gru_a16_aux"])
day = bag(["l_day413_gru_h256"]+[f"l_day413_s{i}" for i in (1,2,3,4)])
tab = bag(["l_tab_p131","l_tab_p131_s1","l_tab_p131_s2"]); two = bag(["two_stage_s1","two_stage_s2"])
baseV = 0.4*gru+0.4*day+0.1*tab+0.1*two; rV = resid(yv,baseV,mv)
famV = {"sm10":["sm10_s42","sm10_s1","sm10_s2"],"sm8":["sm8_s42","sm8_s1"],"f3s":["f3s_s42","f3s_s1","f3s_s2"],"f3saux":["f3saux_s42"],
        "ctx":[f"f3_ctx_s{i}" for i in range(6,13)],"hyb":["l_gru_hyb14_lr1_a16"]+[f"l_gru_hyb14_lr1_a16_s{i}" for i in (1,2,3)],
        "lstm":["l_lstm_a16_s0","l_lstm_a16_s1"],"knn":["knn"]}
basisV=[baseV]+[bag(v)-baseV for v in famV.values()]+[lp("f3spring_s42_pval_testpred.parquet")-baseV]
# ---- P3
p3 = pl.read_parquet(T/"artifacts/pseudo_panel_2025-12-31.parquet").sort("user_id"); on = p3["on"].to_numpy()==1; y3 = np.log1p(np.clip(p3["t"].to_numpy(),0,None))
W = {"pz_a32_s42":0.3,"pz_a32_s1":0.25,"pz_noctx_s42":0.15,"pz_gru_s42":0.15,"pz_dart":0.15}
pz = sum(w*lp(f"{s}_pa1231_testpred.parquet") for s,w in W.items()); f3 = bag(["f3p3_s42","f3p3_s1"]); ctrl = 0.8*f3+0.2*pz; r3 = resid(y3,ctrl,on)
famP = {"sm10":["sm10p3_s42","sm10p3_s1"],"sm8":["sm8p3_s42"],"f3s":["f3sp3_s42","f3sp3_s1"],"f3saux":["f3sauxp3_s42"],"lstm":["lstmp3_s42"],"knn":["knnp3"]}
basisP=[ctrl]+[bag(v)-ctrl for v in famP.values()]+[lp("f3spring_s42_pa1231_testpred.parquet")-ctrl]
print(f"V база {np.sqrt(np.mean(rV**2)):.5f}  P3 контроль {np.sqrt(np.mean(r3[on]**2)):.5f}")
for arg in sys.argv[1:]:
    name, spec = arg.split("="); parts = spec.split(":"); anti = len(parts)>2 and parts[2]=="anti"; sg = -1 if anti else 1
    try:
        dv = sg*(bag(parts[0].split(","))-baseV); dp = None if parts[1]=="-" else sg*(bag(parts[1].split(","))-ctrl)
    except FileNotFoundError as ex: print(name, "нет файлов", ex); continue
    kv,gv,_ = kst(rV, perp(dv,basisV,mv), mv); kv0,_,_ = kst(rV, perp(dv,[baseV],mv), mv)
    kp=gp=kp0=float("nan")
    if dp is not None: kp,gp,_ = kst(r3, perp(dp,basisP,on), on); kp0,_,_ = kst(r3, perp(dp,[ctrl],on), on)
    print(f"{name:14s} V: k⟂={kv:+.2f} (raw {kv0:+.2f}) gain={gv:.5f} | P3: k⟂={kp:+.2f} (raw {kp0:+.2f}) gain={gp:.5f}  {'ANTI' if anti else ''}")

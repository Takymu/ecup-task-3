"""ФАБРИКА НОВЫХ ИСТОЧНИКОВ ИНФОРМАЦИИ (задача юзера: >=10 подтверждённых, порог вал +0.0001).
Каждая группа фичей считается из сырого train.parquet на 19 train-анкерах + вал + тест, кэшируется;
тест = парный GBM (feature_fraction=1.0, одинаковый сид) control vs control+группа; Δ>=0.0001 = CONFIRMED.
В конце: контроль + все подтверждённые группы вместе -> valpred/testpred (factory_all)."""
import numpy as np, polars as pl, lightgbm as lgb, time, json
from pathlib import Path
from datetime import date, timedelta

T = Path("/root/ecup"); M = T/"artifacts/models"; CA = T/"artifacts/factory_cache"; CA.mkdir(exist_ok=True)
DROP = ["user_id","anchor_date","target","anchor_month","anchor_doy"]
df = pl.read_parquet(T/"data/train.parquet")
users = df["user_id"].unique().sort().to_numpy(); N = len(users)
UID = pl.DataFrame({"user_id": users})
FE = sorted((T/"artifacts/features/v2").glob("anchor_*.parquet"))
anchors = [date.fromisoformat(f.stem.split("_")[1]) for f in FE if date(2025,4,1) <= date.fromisoformat(f.stem.split("_")[1]) <= date(2025,12,17)][::2]
VAL, TEST = date(2026,1,14), date(2026,2,13)
ALL_A = anchors + [VAL, TEST]
L = np.log1p
def col(out, c, fill=0.0): return out[c].fill_null(fill).to_numpy().astype(np.float64)
def days_since(out, c, a, fill=99.0):
    return np.array([(a-x).days if x else fill for x in out[c].to_list()], dtype=np.float64)
def agg(h, exprs): return UID.join(h.group_by("user_id").agg(exprs), on="user_id", how="left")

def groups_at(a):
    f = CA/f"g_{a}.npz"
    if f.exists(): return dict(np.load(f))
    h = df.filter(pl.col("event_date") <= a)
    h = h.with_columns(((pl.lit(a)-pl.col("event_date")).dt.total_days()).alias("ago"),
                       (pl.col("search")+pl.col("cat")).alias("act"),
                       pl.col("event_date").dt.weekday().alias("wd"), pl.col("event_date").dt.day().alias("dom"))
    G = {}
    # G1 funnel_rates (30/90)
    cols = []
    for w in (30, 90):
        o = agg(h.filter(pl.col("ago") < w), [pl.len().alias("d"), pl.col("to_cart").sum().alias("c"), pl.col("to_ord").sum().alias("o"),
              pl.col("search").sum().alias("s"), pl.col("cat").sum().alias("k"), pl.col("gmv").sum().alias("g"),
              pl.col("search_to_ord").sum().alias("so"), pl.col("cat_to_ord").sum().alias("ko"), pl.col("has_search_to_ord").sum().alias("hso")])
        d, c, oo, s, k, g, so, ko, hso = [col(o, x) for x in ("d","c","o","s","k","g","so","ko","hso")]
        cols += [c/np.maximum(d,1), oo/np.maximum(c,1), so/np.maximum(s,1), ko/np.maximum(k,1), L(g/np.maximum(oo,1)), hso/np.maximum(d,1)]
    G["funnel"] = np.stack(cols,1)
    # G2 weekday profile 180d + entropy
    o = h.filter(pl.col("ago") < 180).group_by(["user_id","wd"]).agg(pl.len().alias("n")).pivot(on="wd", index="user_id", values="n").fill_null(0)
    o = UID.join(o, on="user_id", how="left").fill_null(0)
    Wm = np.stack([col(o, str(i)) if str(i) in o.columns else np.zeros(N) for i in range(1,8)],1)
    tot = Wm.sum(1, keepdims=True); P_ = Wm/np.maximum(tot,1)
    ent = -(P_*np.log(np.maximum(P_,1e-9))).sum(1)
    G["weekday"] = np.hstack([P_, ent.reshape(-1,1)])
    # G3 activity gaps + streaks (365d)
    s365 = h.filter(pl.col("ago") < 365).sort(["user_id","event_date"]).with_columns(pl.col("event_date").diff().over("user_id").dt.total_days().alias("gap"))
    o = agg(s365, [pl.col("gap").mean().alias("gm"), pl.col("gap").std().alias("gs"), pl.col("gap").max().alias("gx"),
                   pl.col("gap").last().alias("gl"), (pl.col("gap")==1).cast(pl.Int32).sum().alias("consec"), pl.len().alias("n")])
    # текущий стрик: дни подряд до анкера
    cur = agg(h.filter(pl.col("ago") < 60).sort(["user_id","ago"]), [pl.col("ago").alias("ago_list")])
    curs = np.array([next((i for i,v in enumerate(lst) if v != i), len(lst)) if lst is not None else 0 for lst in cur["ago_list"].to_list()], dtype=np.float64)
    G["gaps"] = np.stack([L(col(o,"gm",30)), L(col(o,"gs",0)), L(col(o,"gx",365)), L(col(o,"gl",365)), col(o,"consec")/np.maximum(col(o,"n"),1), L(curs)],1)
    # G4 tail5: последние 5 активных дней
    t5 = h.filter(pl.col("ago") < 120).sort(["user_id","event_date"], descending=[False, True]).with_columns(pl.int_range(0, pl.len()).over("user_id").alias("i")).filter(pl.col("i") < 5)
    cols = []
    for i in range(5):
        o = UID.join(t5.filter(pl.col("i")==i).select("user_id","gmv","to_cart","to_ord","act","ago"), on="user_id", how="left")
        cols += [L(col(o,"gmv")), L(col(o,"to_cart")), L(col(o,"to_ord")), L(col(o,"act")), L(col(o,"ago",120))]
    G["tail5"] = np.stack(cols,1)
    # G5 platform-relative (28d): доля юзера в платформе по активности и gmv; платф. тренд 28 vs 90 (анкер-уровень)
    h28 = h.filter(pl.col("ago") < 28); h90 = h.filter(pl.col("ago") < 90)
    pa28, pg28 = float(h28["act"].sum()), float(h28["gmv"].sum()); pa90, pg90 = float(h90["act"].sum()), float(h90["gmv"].sum())
    o = agg(h28, [pl.col("act").sum().alias("a"), pl.col("gmv").sum().alias("g")])
    G["platrel"] = np.stack([L(col(o,"a")/pa28*N), L(col(o,"g")/max(pg28,1)*N), np.full(N, np.log(pa28/(pa90/90*28))), np.full(N, np.log(max(pg28,1)/(pg90/90*28)))],1)
    # G6 holiday_hist: март-2025 (14.02-15.03) gmv/дни/доля (в прошлом для всех анкеров >= 01.04)
    hm = df.filter((pl.col("event_date")>=date(2025,2,14))&(pl.col("event_date")<=date(2025,3,15)))
    o = agg(hm, [pl.col("gmv").sum().alias("g"), pl.len().alias("d"), pl.col("to_ord").sum().alias("o")])
    o365 = agg(h.filter(pl.col("ago") < 365), [pl.col("gmv").sum().alias("g")])
    G["holiday"] = np.stack([L(col(o,"g")), L(col(o,"d")), L(col(o,"o")), col(o,"g")/np.maximum(col(o365,"g"),1)],1)
    # G7 burstiness 180d
    o = agg(h.filter(pl.col("ago") < 180), [pl.col("gmv").max().alias("gx"), pl.col("gmv").mean().alias("gm"), (pl.col("to_ord")>=2).cast(pl.Int32).sum().alias("multi"),
            pl.col("gmv").filter(pl.col("gmv")>0).quantile(0.9).alias("q90"), pl.col("gmv").filter(pl.col("gmv")>0).median().alias("q50"), pl.col("act").max().alias("ax"), pl.col("act").mean().alias("am")])
    G["burst"] = np.stack([L(col(o,"gx")/np.maximum(col(o,"gm"),1e-3)), L(col(o,"multi")), L(col(o,"q90")/np.maximum(col(o,"q50"),1)), L(col(o,"ax")/np.maximum(col(o,"am"),1e-3))],1)
    # G8 search intensity
    cols = []
    for w in (30, 90):
        o = agg(h.filter(pl.col("ago") < w), [pl.col("searches").sum().alias("ss"), (pl.col("search")>0).cast(pl.Int32).sum().alias("sd"), ((pl.col("search")>0)&(pl.col("to_cart")==0)).cast(pl.Int32).sum().alias("snc")])
        cols += [L(col(o,"ss")/np.maximum(col(o,"sd"),1)), col(o,"snc")/np.maximum(col(o,"sd"),1)]
    G["searchint"] = np.stack(cols + [cols[0]-cols[2]],1)
    # G9 month profile 180d: доля активности по третям месяца
    o = h.filter(pl.col("ago") < 180).with_columns((pl.col("dom")//11).clip(0,2).alias("th")).group_by(["user_id","th"]).agg(pl.len().alias("n")).pivot(on="th", index="user_id", values="n").fill_null(0)
    o = UID.join(o, on="user_id", how="left").fill_null(0)
    Tm = np.stack([col(o,str(i)) if str(i) in o.columns else np.zeros(N) for i in range(3)],1); Tm = Tm/np.maximum(Tm.sum(1,keepdims=True),1)
    G["monthprof"] = Tm
    # G11 conv lag (cart->ord asof)
    carts = h.filter(pl.col("to_cart")>0).select("user_id", pl.col("event_date").alias("cd")).sort(["user_id","cd"])
    ords = h.filter(pl.col("to_ord")>0).select("user_id", pl.col("event_date").alias("od")).sort(["user_id","od"])
    j = carts.join_asof(ords, left_on="cd", right_on="od", by="user_id", strategy="forward").with_columns((pl.col("od")-pl.col("cd")).dt.total_days().alias("lag"))
    o = agg(j, [pl.col("lag").filter(pl.col("lag")<=21).median().alias("ml"), (pl.col("lag")<=14).cast(pl.Float64).mean().alias("c14"), pl.col("cd").max().alias("lc")])
    lo_ = agg(ords, [pl.col("od").max().alias("lo")])
    lc = days_since(o,"lc",a,9999); lod = days_since(lo_,"lo",a,9999); pend = ((lc<lod)&(lc<=14)).astype(np.float64)
    G["convlag"] = np.stack([L(col(o,"ml",30)), col(o,"c14"), pend, L(np.where(pend>0, lc, 30.0)), pend*col(o,"c14")],1)
    # G12 viscosity: активность в 1-7д после дней покупки (180д)
    buys = h.filter((pl.col("gmv")>0)&(pl.col("ago")<180)).select("user_id", pl.col("event_date").alias("bd"))
    acts = h.filter(pl.col("ago")<190).select("user_id", pl.col("event_date").alias("ad"))
    jj = buys.join(acts, on="user_id").with_columns((pl.col("ad")-pl.col("bd")).dt.total_days().alias("dt")).filter((pl.col("dt")>=1)&(pl.col("dt")<=7))
    va = agg(jj, [pl.col("ad").n_unique().alias("na")]); nb = agg(buys, [pl.len().alias("nb")])
    G["visc"] = np.stack([col(va,"na")/np.maximum(col(nb,"nb")*7,1), L(col(nb,"nb"))],1)
    # G15 ctx tabular (анкер-уровень): платф. активность/gmv/заказы за 7/28/90 (одинаково для всех юзеров)
    vals = []
    for w in (7, 28, 90):
        hw = h.filter(pl.col("ago") < w); vals += [np.log(float(hw["act"].sum())/w), np.log(float(hw["gmv"].sum())/w+1), np.log(float(hw["to_ord"].sum())/w+1)]
    G["ctxtab"] = np.tile(np.array(vals, dtype=np.float64), (N,1))
    # G16 per-channel recency
    o = agg(h, [pl.col("event_date").filter(pl.col("search")>0).max().alias("ls"), pl.col("event_date").filter(pl.col("cat")>0).max().alias("lk"),
                pl.col("event_date").filter(pl.col("to_cart")>0).max().alias("lc"), pl.col("event_date").filter(pl.col("to_ord")>0).max().alias("lo"), pl.col("event_date").filter(pl.col("gmv")>0).max().alias("lb")])
    r = {c: L(np.clip(days_since(o,c,a,999),0,999)) for c in ("ls","lk","lc","lo","lb")}
    G["chrecency"] = np.stack([r["ls"], r["lk"], r["lc"], r["lo"], r["lb"], r["lc"]-r["lo"], r["ls"]-r["lk"]],1)
    # G17 daily variability + weekly rhythm (autocorr lag 7) по сетке 56д
    h56 = h.filter(pl.col("ago") < 56)
    A = np.zeros((N, 56), dtype=np.float32)
    A[np.searchsorted(users, h56["user_id"].to_numpy()), h56["ago"].to_numpy().astype(np.int64)] = L(h56["act"].to_numpy().astype(np.float64)).astype(np.float32)
    Ac = A - A.mean(1, keepdims=True); den = (Ac**2).sum(1)
    ac7 = (Ac[:, :-7]*Ac[:, 7:]).sum(1)/np.maximum(den, 1e-6)
    G["rhythm"] = np.stack([A.std(1), ac7, (A>0).sum(1)/56.0, A[:, :28].std(1)-A[:, 28:].std(1)],1)
    # G18 channel mix
    cols = []
    for w in (30, 90):
        o = agg(h.filter(pl.col("ago") < w), [pl.col("search").sum().alias("s"), pl.col("cat").sum().alias("k"), ((pl.col("search")>0)&(pl.col("cat")>0)).cast(pl.Int32).sum().alias("both"), pl.len().alias("d")])
        cols += [col(o,"s")/np.maximum(col(o,"s")+col(o,"k"),1), col(o,"both")/np.maximum(col(o,"d"),1)]
    G["chmix"] = np.stack(cols + [cols[0]-cols[2]],1)
    # G19 order size
    ob = h.filter((pl.col("gmv")>0)&(pl.col("ago")<365)).with_columns((pl.col("gmv")/pl.col("to_ord").clip(lower_bound=1)).alias("aov"))
    o = agg(ob, [pl.col("aov").mean().alias("am"), pl.col("aov").max().alias("ax"), pl.col("aov").last().alias("al"), pl.col("aov").quantile(0.75).alias("a75")])
    G["ordsize"] = np.stack([L(col(o,"am")), L(col(o,"ax")), L(col(o,"al")), L(col(o,"al"))-L(col(o,"a75"))],1)
    # G20 tenure & buy rate
    o = agg(h, [pl.col("event_date").min().alias("fe"), pl.col("event_date").filter(pl.col("gmv")>0).min().alias("fb"), (pl.col("gmv")>0).cast(pl.Int32).sum().alias("nb")])
    ten = days_since(o,"fe",a,0); fb = days_since(o,"fb",a,0)
    G["tenure"] = np.stack([L(ten), L(fb), col(o,"nb")/np.maximum(ten,1)*30, L(np.maximum(fb-ten,0))],1)
    np.savez(f, **{k: v.astype(np.float32) for k, v in G.items()})
    return {k: v.astype(np.float32) for k, v in G.items()}

t0 = time.time()
GA = {}
for a in ALL_A:
    GA[a] = groups_at(a); print(f"groups {a} ok ({time.time()-t0:.0f}s)", flush=True)
names = list(GA[VAL].keys())
print("groups:", names, flush=True)

def load(a):
    F = pl.read_parquet(T/f"artifacts/features/v2/anchor_{a}.parquet").sort("user_id")
    X = np.nan_to_num(F.drop([c for c in DROP if c in F.columns]).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = L(np.clip(F["target"].to_numpy(),0,None)).astype(np.float32) if "target" in F.columns else None
    return F, X, y
Xs, ys = [], []
for a in anchors:
    _, X, y = load(a); Xs.append(X); ys.append(y)
X = np.concatenate(Xs); y = np.concatenate(ys); del Xs, ys
Fv, Xv, yv = load(VAL); Ft, Xt, _ = load(TEST)
def rank_score(p):
    g = np.linspace(-0.6,0.6,121)
    return min(np.sqrt(np.mean((yv-np.clip(p+d0,0,None))**2)) for d0 in g)
P = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
         feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1, num_threads=46)
def stack(gn):
    return (np.concatenate([GA[a][gn] for a in anchors]), GA[VAL][gn], GA[TEST][gn])
ctrl = lgb.train(P, lgb.Dataset(X, y), 400); s0 = rank_score(ctrl.predict(Xv)); del ctrl
print(f"CONTROL: {s0:.5f}", flush=True)
res = {}
for gn in names:
    Gtr, Gv, Gt = stack(gn)
    m = lgb.train(P, lgb.Dataset(np.hstack([X, Gtr]), y), 400)
    s = rank_score(m.predict(np.hstack([Xv, Gv]))); del m
    res[gn] = s0 - s
    print(f"GROUP {gn:10s} ({Gtr.shape[1]:2d} фич): {s:.5f}  Δ={s0-s:+.5f}  {'CONFIRMED' if s0-s >= 1e-4 else '-'}", flush=True)
conf = [g for g, d in res.items() if d >= 1e-4]
print("CONFIRMED:", conf, flush=True)
json.dump({"control": s0, "delta": res, "confirmed": conf}, open(M/"factory_results.json","w"), indent=1)
if conf:
    Gtr = np.hstack([stack(g)[0] for g in conf]); Gv = np.hstack([stack(g)[1] for g in conf]); Gt = np.hstack([stack(g)[2] for g in conf])
    m = lgb.train(P, lgb.Dataset(np.hstack([X, Gtr]), y), 400)
    pv = m.predict(np.hstack([Xv, Gv])); print(f"ALL CONFIRMED together: {rank_score(pv):.5f} (control {s0:.5f})", flush=True)
    pl.DataFrame({"user_id": Fv["user_id"], "pred": np.expm1(np.clip(pv,0,None)), "target": Fv["target"]}).write_parquet(M/"factory_all_valpred.parquet")
    pl.DataFrame({"user_id": Ft["user_id"], "pred": np.expm1(np.clip(m.predict(np.hstack([Xt, Gt])),0,None))}).write_parquet(M/"factory_all_full_testpred.parquet")
print("FACTORY_DONE", flush=True)

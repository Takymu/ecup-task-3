"""Источник: день недели анкера. Все train-анкеры — среды, тест-анкер 13.02.2026 — ПЯТНИЦА
(окно 30д содержит 5 пар выходных vs 4; рецентность на пятницу распределена иначе).
Строим пятничную сетку анкеров тем же build_anchor (фичи идентичны), учим GBM Fri vs Wed,
оцениваем оба на пятничном вал-анкере 09.01.2026 (окно до 08.02 наблюдаемо) и на средовом 14.01.
Тест-предикт Fri-модели на 13.02 (тест-фичи уже пятничные)."""
import sys, time
sys.path.insert(0, "/root/ecup/src")
import numpy as np, polars as pl, lightgbm as lgb
from pathlib import Path
from datetime import date, timedelta
import build_features as bf

T = Path("/root/ecup"); M = T/"artifacts/models"; OUT = T/"artifacts/features/v2fri"; OUT.mkdir(parents=True, exist_ok=True)
DROP = ["user_id","anchor_date","target","anchor_month","anchor_doy"]
df = pl.read_parquet(T/"data/train.parquet")
value_cols = [c for c in df.columns if c not in ("event_date","user_id")]
all_users = df.select(pl.col("user_id").unique().sort())
FRI_VAL = date(2026,1,9)
fri_train = [FRI_VAL - timedelta(days=14*i) for i in range(1, 22) if FRI_VAL - timedelta(days=14*i) >= date(2025,4,1)]
fri_train = sorted(a for a in fri_train if a <= date(2025,12,12))
for a in fri_train + [FRI_VAL]:
    p = OUT/f"anchor_{a}.parquet"
    if p.exists(): continue
    t0 = time.time(); bf.build_anchor(df, all_users, a, value_cols, True).write_parquet(p)
    print(f"built {a} ({time.time()-t0:.0f}s)", flush=True)
print("fri anchors:", len(fri_train), fri_train[0], "..", fri_train[-1], flush=True)
def load(path):
    F = pl.read_parquet(path).sort("user_id")
    X = np.nan_to_num(F.drop([c for c in DROP if c in F.columns]).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.log1p(np.clip(F["target"].to_numpy(),0,None)).astype(np.float32) if "target" in F.columns and F["target"].null_count()==0 else None
    return F, X, y
FE = sorted((T/"artifacts/features/v2").glob("anchor_*.parquet"))
wed_train = [f for f in FE if date(2025,4,1) <= date.fromisoformat(f.stem.split("_")[1]) <= date(2025,12,17)][::2]
def stack(paths):
    Xs, ys = [], []
    for p in paths:
        _, X, y = load(p); Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)
Xw, yw = stack(wed_train); Xf, yf = stack([OUT/f"anchor_{a}.parquet" for a in fri_train])
print("wed rows", Xw.shape, "fri rows", Xf.shape, flush=True)
Fvf, Xvf, yvf = load(OUT/f"anchor_{FRI_VAL}.parquet")
Fvw, Xvw, yvw = load(T/"artifacts/features/v2/anchor_2026-01-14.parquet")
Ft, Xt, _ = load(T/"artifacts/features/v2/anchor_2026-02-13.parquet")
def rank_score(y, p):
    g = np.linspace(-0.6,0.6,121)
    return min(np.sqrt(np.mean((y-np.clip(p+d0,0,None))**2)) for d0 in g)
P = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
         feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1, num_threads=46)
mw = lgb.train(P, lgb.Dataset(Xw, yw), 400); mf = lgb.train(P, lgb.Dataset(Xf, yf), 400)
print(f"FRI-val 09.01: wed-model {rank_score(yvf, mw.predict(Xvf)):.5f}  fri-model {rank_score(yvf, mf.predict(Xvf)):.5f}", flush=True)
print(f"WED-val 14.01: wed-model {rank_score(yvw, mw.predict(Xvw)):.5f}  fri-model {rank_score(yvw, mf.predict(Xvw)):.5f}", flush=True)
mb = lgb.train(P, lgb.Dataset(np.vstack([Xw, Xf]), np.concatenate([yw, yf])), 400)
print(f"BOTH-model: fri-val {rank_score(yvf, mb.predict(Xvf)):.5f}  wed-val {rank_score(yvw, mb.predict(Xvw)):.5f}", flush=True)
for nm, m in [("fri_gbm", mf), ("wed_gbm", mw), ("both_gbm", mb)]:
    pl.DataFrame({"user_id": Fvw["user_id"], "pred": np.expm1(np.clip(m.predict(Xvw),0,None)), "target": Fvw["target"]}).write_parquet(M/f"{nm}_valpred.parquet")
    pl.DataFrame({"user_id": Ft["user_id"], "pred": np.expm1(np.clip(m.predict(Xt),0,None))}).write_parquet(M/f"{nm}_full_testpred.parquet")
print("FRIDAY_DONE", flush=True)

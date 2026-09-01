"""Источник: положение юзера относительно популяции в тот же день (ранговая нормализация внутри анкера)
— убирает дрейф платформы (+45%/год) между тренировочными анкерами и тестом.
Парный GBM: raw (контроль) vs raw+ranks (все 267 фич -> перцентиль внутри анкера) vs ranks-only."""
import numpy as np, polars as pl, lightgbm as lgb
from pathlib import Path
from datetime import date
from scipy.stats import rankdata

T = Path("/root/ecup"); M = T/"artifacts/models"
DROP = ["user_id","anchor_date","target","anchor_month","anchor_doy"]
FE = sorted((T/"artifacts/features/v2").glob("anchor_*.parquet"))
anchors = [f for f in FE if date(2025,4,1) <= date.fromisoformat(f.stem.split("_")[1]) <= date(2025,12,17)][::2]
def load(path):
    F = pl.read_parquet(path).sort("user_id")
    X = np.nan_to_num(F.drop([c for c in DROP if c in F.columns]).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.log1p(np.clip(F["target"].to_numpy(),0,None)).astype(np.float32) if "target" in F.columns else None
    return F, X, y
def ranks(X):
    R = np.empty_like(X)
    for j in range(X.shape[1]):
        R[:, j] = rankdata(X[:, j], method="average") / len(X)
    return R.astype(np.float32)
Xs, Rs, ys = [], [], []
for f in anchors:
    _, X, y = load(f); Xs.append(X); Rs.append(ranks(X)); ys.append(y)
    print("ranked", f.stem, flush=True)
X = np.concatenate(Xs); R = np.concatenate(Rs); y = np.concatenate(ys); del Xs, Rs, ys
Fv, Xv, yv = load(T/"artifacts/features/v2/anchor_2026-01-14.parquet"); Rv = ranks(Xv)
Ft, Xt, _ = load(T/"artifacts/features/v2/anchor_2026-02-13.parquet"); Rt = ranks(Xt)
def rank_score(p):
    g = np.linspace(-0.6,0.6,121)
    return min(np.sqrt(np.mean((yv-np.clip(p+d0,0,None))**2)) for d0 in g)
P = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
         feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1, num_threads=46)
for name, A, Av, At in [("rk_ctrl", X, Xv, Xt), ("rk_plus", np.hstack([X, R]), np.hstack([Xv, Rv]), np.hstack([Xt, Rt])), ("rk_only", R, Rv, Rt)]:
    m = lgb.train(P, lgb.Dataset(A, y), 400)
    pv = m.predict(Av); print(f"{name}: val rank-score {rank_score(pv):.5f}", flush=True)
    pl.DataFrame({"user_id": Fv["user_id"], "pred": np.expm1(np.clip(pv,0,None)), "target": Fv["target"]}).write_parquet(M/f"{name}_valpred.parquet")
    pl.DataFrame({"user_id": Ft["user_id"], "pred": np.expm1(np.clip(m.predict(At),0,None))}).write_parquet(M/f"{name}_full_testpred.parquet")
    del m
print("RANK_DONE", flush=True)

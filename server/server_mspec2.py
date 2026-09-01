"""mspec2: gift-season специалист v2 — 42-дневные фичи (анкеры 12.02-05.03.2025, история влезает),
LGBM + GRU-бэг; in-season холдаут (train 3 -> val 05.03); ОФФ-СЕЗОННЫЙ тест специфичности: предикт
на 31.12.2025 (окно янв-2026, не подарочное) -> mspec2_pa1231; тест-предикт на 13.02.2026."""
import numpy as np, polars as pl, lightgbm as lgb, torch, torch.nn as nn
from pathlib import Path
from datetime import date, timedelta

T = Path("/root/ecup"); M = T/"artifacts/models"
dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pl.read_parquet(T/"data/train.parquet",
                     columns=["user_id","event_date","search","cat","to_cart","to_ord","gmv"])
users = df["user_id"].unique().sort().to_numpy(); N = len(users)
ANCH = [date(2025,2,12), date(2025,2,19), date(2025,2,26), date(2025,3,5)]
TEST = date(2026,2,13); OFF = date(2025,12,31)

def target(a):
    g = (df.filter((pl.col("event_date")>a)&(pl.col("event_date")<=a+timedelta(days=30)))
           .group_by("user_id").agg(pl.col("gmv").sum().alias("t")))
    return np.log1p(np.clip(pl.DataFrame({"user_id":users}).join(g,on="user_id",how="left")["t"].fill_null(0).to_numpy(),0,None)).astype(np.float32)

def short_feats(a):
    h = df.filter((pl.col("event_date")<=a)&(pl.col("event_date")>a-timedelta(days=42)))
    cols = []
    for w in (7,14,28,42):
        hw = h.filter(pl.col("event_date")>a-timedelta(days=w))
        g = hw.group_by("user_id").agg([
            pl.len().alias(f"da{w}"), pl.col("gmv").sum().alias(f"g{w}"),
            pl.col("to_ord").sum().alias(f"o{w}"), pl.col("to_cart").sum().alias(f"c{w}"),
            pl.col("search").sum().alias(f"s{w}"), pl.col("cat").sum().alias(f"k{w}")])
        cols.append(g)
    last = h.group_by("user_id").agg([
        pl.col("event_date").max().alias("le"),
        pl.col("event_date").filter(pl.col("to_ord")>0).max().alias("lo"),
        pl.col("event_date").filter(pl.col("to_cart")>0).max().alias("lc"),
        (pl.col("event_date").dt.weekday()>=6).cast(pl.Float64).mean().alias("wk")])
    out = pl.DataFrame({"user_id":users})
    for g in cols: out = out.join(g, on="user_id", how="left")
    out = out.join(last, on="user_id", how="left")
    X = []
    for w in (7,14,28,42):
        for p in ("da","g","o","c","s","k"):
            X.append(np.log1p(out[f"{p}{w}"].fill_null(0).to_numpy().astype(np.float64)))
    for c in ("le","lo","lc"):
        v = np.array([(a-x).days if x else 99 for x in out[c].to_list()], dtype=np.float64)
        X.append(np.log1p(np.clip(v,0,99)))
    X.append(out["wk"].fill_null(2/7).to_numpy().astype(np.float64))
    return np.stack(X,1).astype(np.float32)

def rank_score(y, p):
    g = np.linspace(-0.9,0.9,181)
    return min(np.sqrt(np.mean((y-np.clip(p+d0,0,None))**2)) for d0 in g)

print("features...", flush=True)
F = {a: short_feats(a) for a in ANCH}
Ft = short_feats(TEST); Fo = short_feats(OFF)
Y = {a: target(a) for a in ANCH}
Yo = target(OFF)
P = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
         feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1, num_threads=32)
tr3 = ANCH[:3]; hold = ANCH[3]
m3 = lgb.train(P, lgb.Dataset(np.concatenate([F[a] for a in tr3]), np.concatenate([Y[a] for a in tr3])), num_boost_round=400)
print(f"LGBM holdout {hold}: rank {rank_score(Y[hold], m3.predict(F[hold])):.5f}", flush=True)
mf = lgb.train(P, lgb.Dataset(np.concatenate([F[a] for a in ANCH]), np.concatenate([Y[a] for a in ANCH])), num_boost_round=400)
po = mf.predict(Fo)
print(f"OFF-season (31.12->янв): rank {rank_score(Yo, po):.5f}", flush=True)
pl.DataFrame({"user_id": users, "pred": np.expm1(np.clip(po,0,None))}).write_parquet(M/"mspec2_pa1231_testpred.parquet")
pt = mf.predict(Ft)
pl.DataFrame({"user_id": users, "pred": np.expm1(np.clip(pt,0,None))}).write_parquet(M/"mspec2_lgbm_testpred.parquet")

D = 42; C = 7
def grid(a):
    h = df.filter((pl.col("event_date")<=a)&(pl.col("event_date")>a-timedelta(days=D)))
    A = np.zeros((N, D, C), dtype=np.float16)
    rows = np.searchsorted(users, h["user_id"].to_numpy())
    dd = h.select(((pl.lit(a)-pl.col("event_date")).dt.total_days()).alias("d"))["d"].to_numpy().astype(np.int64)
    pos = D-1-dd
    A[rows, pos, 0] = 1.0
    for j,c in enumerate(["search","cat","to_cart","to_ord","gmv"]):
        A[rows, pos, 1+j] = np.log1p(np.clip(h[c].to_numpy().astype(np.float64),0,None)).astype(np.float16)
    A[rows, pos, 6] = (h["event_date"].dt.weekday()>=6).cast(pl.Float32).to_numpy().astype(np.float16)
    return A
print("grids...", flush=True)
G = {a: grid(a) for a in ANCH}; Gt = grid(TEST)

class Spec(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.gru = nn.GRU(C, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x):
        _, hn = self.gru(x)
        return self.head(hn[-1]).squeeze(1)

def train_gru(anchors, seed, epochs=6):
    torch.manual_seed(seed)
    m = Spec().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        for ai in rng.permutation(len(anchors)):
            aa = anchors[ai]; A = G[aa]; y = Y[aa]
            perm = rng.permutation(N)
            for i in range(0, N, 8192):
                idx = perm[i:i+8192]
                xb = torch.from_numpy(A[idx]).float().to(dev)
                yb = torch.from_numpy(y[idx]).to(dev)
                loss = ((m(xb)-yb)**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m

@torch.no_grad()
def pred(m, A):
    out = []
    for i in range(0, N, 32768):
        out.append(m(torch.from_numpy(A[i:i+32768]).float().to(dev)).cpu().numpy())
    return np.concatenate(out)

preds = []
for s in (42,1,2):
    g_ = train_gru(ANCH, s)
    preds.append(pred(g_, Gt))
    print(f"GRU seed {s} done", flush=True)
pl.DataFrame({"user_id": users, "pred": np.expm1(np.clip(np.mean(preds,0),0,None))}).write_parquet(M/"mspec2_gru_testpred.parquet")
print("MSPEC2_DONE", flush=True)

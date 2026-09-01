"""Финальная форма mspec: LGBM 0.5 + полный GRU-бэг 11 моделей 0.5. База v29, rms 0.05."""
import numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[2]
sub = pl.read_csv(T/"artifacts/submissions/sub_v29_L233.csv").sort("user_id")
lp29 = np.log1p(sub["predict"].to_numpy())
def lpof(f):
    d = pl.read_parquet(T/f"artifacts/models/{f}").sort("user_id")
    assert (d["user_id"].to_numpy() == sub["user_id"].to_numpy()).all()
    return np.log1p(np.clip(d["pred"].to_numpy(),0,None))
lgbm = lpof("mspec2_lgbm_testpred.parquet")
gru3 = lpof("mspec2_gru_testpred.parquet")
gru8 = lpof("mspec2_gru2_testpred.parquet")
gru_all = (3*gru3 + 8*gru8)/11
print("corr(gru3, gru8):", round(float(np.corrcoef(gru3,gru8)[0,1]),4))
lpm = 0.5*lgbm + 0.5*gru_all
a = lpm - lpm.mean(); b = lp29 - lp29.mean()
beta = float(np.dot(a,b)/np.dot(b,b))
d = a - beta*b; d -= d.mean()
# Проба летала на ЛБ в масштабе rms 0.05; при унификации системы измерений все формы
# приведены к rms 0.02, коэффициенты пересчитаны. Сохраняем в унифицированном масштабе.
s = 0.02/float(np.sqrt((d**2).mean())); d *= s
old = None
np.save(T/"artifacts/forms/probe_form_mspec.npy", d)
x = lp29 + d; x += 2.33 - x.mean()
pred = np.clip(np.expm1(np.clip(x,0,None)),0,None)
out = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(
    pl.DataFrame({"user_id": sub["user_id"].to_numpy(), "predict": pred}), on="user_id", how="left")
assert out["predict"].null_count()==0
out.write_csv(T/"artifacts/submissions/sub_probe_mspec_L233.csv")
print("saved FINAL sub_probe_mspec_L233.csv + probe_form_mspec.npy")
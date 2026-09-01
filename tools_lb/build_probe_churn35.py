"""27.08: одна проба «churn» на базе v35 = нормированная сумма четырёх форм одного механизма (отбор панели завышает метки «уходящих»):
gappy + light + fading + young (каждая rms .02, ⟂ измеренному; между собой |corr|<=.38). Знак −: понижаем уходящих.
k=0 → 1.645300, k=.5 → v35 1.645178, k=1 → 1.645057; если все четыре реальны с k~1, k_churn ≈ 1.6–1.8 (проба ~1.64496)."""
import sys, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
parts = {n: np.load(T/f"artifacts/forms/probe_form_{n}.npy") for n in ("gappy","light","fading","young")}
pack = sum(parts.values()); pack -= pack.mean(); print("rms суммы", round(float(pack.std()),4)); f = pack*(0.02/pack.std())
print("corr churn vs parts:", {n: round(float(np.corrcoef(f,v)[0,1]),2) for n,v in parts.items()}, "| corr vs lp", round(float(np.corrcoef(f,lp)[0,1]),4))
if "save" in sys.argv:
    np.save(T/"artifacts/forms/probe_form_churn.npy", f); x = lp + f; x += 2.33 - x.mean()
    o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
    assert o["predict"].null_count()==0; o.write_csv(S/"sub_probe_churn35_L233.csv"); print("saved sub_probe_churn35_L233.csv (base v35)")

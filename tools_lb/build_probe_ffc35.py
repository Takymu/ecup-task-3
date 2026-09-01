"""27.08: перекладка пробы ffc на базу v35 (sub_v35_L233, ЛБ 1.645178087). Форма probe_form_ffc.npy та же (⟂ lp_alstm и всем формам;
v35 лежит в span{lb_base, формы} → ⟂ и к lp_v35). k=0 (форма пустая) → M = M_v35 + C_ii = 1.645300; k=0.5 → ровно v35 1.645178; k=1 → 1.645057. Usage: build_probe_ffc35.py save"""
import sys, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"
sub = pl.read_csv(S/"sub_v35_L233.csv").sort("user_id"); uid = sub["user_id"].to_numpy(); lp = np.log1p(sub["predict"].to_numpy())
f = np.load(T/"artifacts/forms/probe_form_ffc.npy")
print(f"form rms {f.std():.4f} mean {f.mean():+.2e} | corr(f, lp_v35 centered) {np.corrcoef(f, lp-lp.mean())[0,1]:+.2e}")
x = lp + f; x += 2.33 - x.mean(); print(f"level {x.mean():.4f}; k=0: {np.sqrt(1.645178087**2 + f.var()):.6f}  k=1: {np.sqrt(1.645178087**2 - f.var()):.6f}")
if "save" in sys.argv:
    o = pl.read_csv(T/"data/sample_submit.csv").select("user_id").join(pl.DataFrame({"user_id":uid,"predict":np.clip(np.expm1(np.clip(x,0,None)),0,None)}), on="user_id", how="left")
    assert o["predict"].null_count()==0; o.write_csv(S/"sub_probe_ffc35_L233.csv"); print("saved sub_probe_ffc35_L233.csv (base v35)")

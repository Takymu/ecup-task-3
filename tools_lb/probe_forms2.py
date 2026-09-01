"""Forms for round-2 structure probes: d = log1p(probe) - log1p(v24ind), + C matrix with ctx form."""
import numpy as np, polars as pl, json
from pathlib import Path
T = Path(__file__).resolve().parents[1]; S = T/"artifacts/submissions"

base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv").sort("user_id")
lb = np.log1p(base["predict"].to_numpy())
ctxp = pl.read_csv(S/"sub_probe_ctxup13_L233.csv").sort("user_id")
assert (ctxp["user_id"].to_numpy() == base["user_id"].to_numpy()).all()
forms = {"ctx": np.log1p(ctxp["predict"].to_numpy()) - lb}
np.save(T/"artifacts/forms/probe_form_ctxup13.npy", forms["ctx"])
for name, f in [("ts", "sub_probe_tsup16_L233.csv"), ("d413", "sub_probe_d413up15_L233.csv")]:
    p = pl.read_csv(S/f).sort("user_id")
    assert (p["user_id"].to_numpy() == base["user_id"].to_numpy()).all()
    forms[name] = np.log1p(p["predict"].to_numpy()) - lb

names = list(forms)
F = np.stack([forms[n] for n in names])
C = F @ F.T / F.shape[1]
print("forms:", names)
print("rms:", {n: float(np.sqrt(np.mean(forms[n]**2))) for n in names})
print("C matrix:")
for i, n in enumerate(names): print(" ", n, [f"{C[i,j]:.3e}" for j in range(len(names))])
corr = np.corrcoef(F)
print("corr:", np.round(corr, 3))
np.save(T/"artifacts/forms/probe_forms2.npy", F)
json.dump({"names": names, "C": C.tolist(),
           "M_base": 1.6462083685341875**2,
           "ctx_probe": {"sub": "sub_probe_ctxup13_L233", "lb": 1.6461552952}},
          open(T/"artifacts/lb/lb_probes2.json", "w"), indent=2)
print("saved probe_forms2.npy + lb_probes2.json")

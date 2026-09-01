import json, numpy as np, polars as pl
from pathlib import Path
SUBS = r"artifacts/submissions"; SAMPLE = r"data/sample_submit.csv"
cfg = json.loads(Path(r"artifacts/lb/blend_v15_weights.json").read_text())
acc = users = None; tot = 0
for fam, w in cfg["family_weights"].items():
    ms = cfg["members"][fam]
    for p in ms:
        df = pl.read_parquet(p).sort("user_id"); lp = np.log1p(np.clip(df["pred"].to_numpy(), 0, None))
        ww = w / len(ms); c = ww * (lp - lp.mean()); tot += ww
        users, acc = (df["user_id"], c) if acc is None else (users, acc + c)
        print(f"  {fam:<11} {Path(p).name[:34]:<34} w={ww:.3f} mean_lp={lp.mean():.4f}")
acc /= tot
sample = pl.read_csv(SAMPLE)
for L in (2.30, 2.33, 2.38):
    name = f"sub_v15_famblend_L{int(round(L*100))}"
    pred = np.clip(np.expm1(acc + L), 0, None)
    sub = sample.select("user_id").join(pl.DataFrame({"user_id": users, "predict": pred}), on="user_id", how="left")
    assert sub["predict"].null_count() == 0
    sub.write_csv(f"{SUBS}/{name}.csv"); print(f"saved {name}.csv mean_lp={np.log1p(pred).mean():.4f} p50={np.median(pred):.2f}")

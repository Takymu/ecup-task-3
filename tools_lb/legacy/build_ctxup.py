"""Blend-structure probe: ctx-family weight x1.3 (renormalized) on top of v24ind, comb5 applied.
Also saves the form d_ctx = lp_ctxup - lp_v24 (log diff of the final subs)."""
import json, numpy as np, polars as pl
from pathlib import Path
T = Path(__file__).resolve().parents[2]
cfg = json.loads((T/"artifacts/lb/blend_v24ind_weights.json").read_text())
w = dict(cfg["family_weights"])
isctx = lambda n: ("ctx" in n)
ctx_share = sum(v for k,v in w.items() if isctx(k))
print("ctx share in v24ind:", round(ctx_share,3), {k: round(v,3) for k,v in w.items() if isctx(k)})
w2 = {k: (v*1.3 if isctx(k) else v) for k,v in w.items()}
tot = sum(w2.values()); w2 = {k: v/tot for k,v in w2.items()}
print("new ctx share:", round(sum(v for k,v in w2.items() if isctx(k)),3))
cfg2 = {"rank_blend_val": None, "family_weights": w2, "members": cfg["members"]}
(T/"artifacts/lb/blend_probe_ctxup13_weights.json").write_text(json.dumps(cfg2, indent=2))

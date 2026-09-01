"""Blend val predictions of several models in log-space; grid-search weights.

Usage: python blend.py lgbm_expB two_stage_v2 ...   (stems of *_valpred.parquet in models dir)
"""
from __future__ import annotations

import sys
from itertools import product

import numpy as np
import polars as pl

from common import MODELS_DIR, rmsle

stems = [s for s in sys.argv[1:] if (MODELS_DIR / f"{s}_valpred.parquet").exists()]
missing = set(sys.argv[1:]) - set(stems)
if missing:
    print(f"skipping missing valpreds: {sorted(missing)}")
assert len(stems) >= 2, "need >=2 model stems"

preds = {}
target = None
for s in stems:
    df = pl.read_parquet(MODELS_DIR / f"{s}_valpred.parquet").sort("user_id")
    preds[s] = np.log1p(np.clip(df["pred"].to_numpy(), 0, None))
    target = np.clip(df["target"].to_numpy(), 0, None)

for s in stems:
    print(f"{s}: RMSLE = {rmsle(target, np.expm1(preds[s])):.5f}")

P = np.stack([preds[s] for s in stems])

if len(stems) <= 4:
    best = None
    grid = np.arange(0, 1.01, 0.05)
    for w in product(grid, repeat=len(stems) - 1):
        if sum(w) > 1:
            continue
        ws = np.array(list(w) + [1 - sum(w)])
        s = rmsle(target, np.expm1((ws[:, None] * P).sum(axis=0)))
        if best is None or s < best[0]:
            best = (s, ws)
    print(f"\nbest blend: RMSLE = {best[0]:.5f}  weights = {dict(zip(stems, best[1].round(2)))}")
else:
    # greedy Caruana ensemble with replacement, log-space averaging
    counts = np.zeros(len(stems), dtype=int)
    cur = np.zeros_like(P[0])
    best_s = 1e9
    for _ in range(60):
        cand = [(rmsle(target, np.expm1((cur * counts.sum() + P[i]) / (counts.sum() + 1))), i)
                for i in range(len(stems))]
        s, i = min(cand)
        if s >= best_s - 1e-6:
            break
        counts[i] += 1
        cur = (cur * (counts.sum() - 1) + P[i]) / counts.sum()
        best_s = s
    ws = counts / counts.sum()
    print(f"\ngreedy blend: RMSLE = {best_s:.5f}")
    for s_, w in sorted(zip(stems, ws), key=lambda t: -t[1]):
        if w > 0:
            print(f"  {s_}: {w:.3f}")

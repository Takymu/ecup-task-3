"""BG/NBD (Fader-Hardie-Lee 2005) 'Buy Till You Die' features per anchor.

Transactions = purchase days. Per user at anchor A: x = repeat purchase days, t_x = last-first
purchase (weeks), T = A - first purchase (weeks). Population params (r, alpha, a, b) fitted by
MLE on data <= A only (leak-free). Outputs per user:
  btyd_palive   P(customer still 'alive' at A)
  btyd_e30      E[# purchase days in next 30 days]
  btyd_lam_post posterior mean purchase rate (r+x)/(alpha+T)  [per week]
Users with no purchase yet: nulls (BTYD is undefined before the first purchase).

Writes FEATURES_DIR/v2_btyd/anchor_*.parquet.  Usage: python build_btyd_feats.py [--tag v2]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

from common import FEATURES_DIR, MODELS_DIR, TRAIN_PARQUET


def neg_ll(params, x, tx, T):
    r, alpha, a, b = np.exp(params)
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    A3 = -(r + x) * np.log(alpha + T)
    pos = x > 0
    A4 = np.full_like(A3, -np.inf)
    A4[pos] = np.log(a) - np.log(b + x[pos] - 1) - (r + x[pos]) * np.log(alpha + tx[pos])
    return -np.mean(A1 + A2 + np.logaddexp(A3, A4))


def fit(x, tx, T, seed=0, n_sub=60000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(n_sub, len(x)), replace=False)
    best = None
    for init in ([0.0, 0.0, 0.0, 0.0], [np.log(0.5), np.log(2.0), np.log(0.8), np.log(3.0)]):
        res = minimize(neg_ll, init, args=(x[idx], tx[idx], T[idx]), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(best.x), -best.fun


def quantities(params, x, tx, T, t):
    r, alpha, a, b = params
    pos = x > 0
    ratio = np.zeros_like(T)
    ratio[pos] = a / (b + x[pos] - 1) * ((alpha + T[pos]) / (alpha + tx[pos])) ** (r + x[pos])
    palive = 1.0 / (1.0 + ratio)
    lam_post = (r + x) / (alpha + T)
    if a > 1:
        z = t / (alpha + T + t)
        h = hyp2f1(r + x, b + x, a + b + x - 1, z)
        num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * h)
        e = num / (1.0 + ratio)
    else:  # expectation undefined for a<=1: fall back to alive-weighted posterior rate
        e = palive * lam_post * t
    return palive, e, lam_post


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out-tag", default="v2_btyd")
    args = ap.parse_args()

    t0 = time.time()
    buys = pl.scan_parquet(TRAIN_PARQUET).filter(pl.col("gmv") > 0).select(["event_date", "user_id"]).collect()
    files = sorted((FEATURES_DIR / args.tag).glob("anchor_*.parquet"))
    base_users = pl.read_parquet(files[0], columns=["user_id"]).sort("user_id")
    out_dir = FEATURES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted = {}
    for f in files:
        a = date.fromisoformat(f.stem.removeprefix("anchor_"))
        h = buys.filter(pl.col("event_date") <= a).group_by("user_id").agg([
            pl.col("event_date").min().alias("first"), pl.col("event_date").max().alias("last"),
            pl.len().alias("n")])
        h = h.with_columns([
            (pl.col("n") - 1).cast(pl.Float64).alias("x"),
            ((pl.col("last") - pl.col("first")).dt.total_days() / 7.0).alias("tx"),
            ((pl.lit(a) - pl.col("first")).dt.total_days() / 7.0 + 1e-3).alias("T"),
        ])
        x, tx, T = (h[c].to_numpy().astype(np.float64) for c in ("x", "tx", "T"))
        params, ll = fit(x, tx, T)
        palive, e30, lam = quantities(params, x, tx, T, 30 / 7.0)
        g = pl.DataFrame({"user_id": h["user_id"], "btyd_palive": palive, "btyd_e30": e30,
                          "btyd_lam_post": lam})
        base_users.join(g, on="user_id", how="left").write_parquet(out_dir / f.name)
        fitted[str(a)] = dict(r=params[0], alpha=params[1], a=params[2], b=params[3], ll=ll,
                              n_customers=len(h))
        print(f"  {a}: r={params[0]:.3f} alpha={params[1]:.3f} a={params[2]:.3f} b={params[3]:.3f} "
              f"| mean palive={palive.mean():.3f} e30={e30.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
    (MODELS_DIR / "btyd_params.json").write_text(json.dumps(fitted, indent=2))
    print(f"saved {len(files)} extras files to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

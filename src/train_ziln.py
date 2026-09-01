"""ZILN-style MLP (torch): 3 heads (p, mu, sigma) on tabular anchor features.

Loss = BCE(p, y>0) + [y>0] * GaussianNLL(mu, sigma; log1p y)   (zero-inflated lognormal)
RMSLE prediction = expm1(sigmoid(p) * mu)  — E[log1p(y)] decomposition.

Usage: python train_ziln.py --tag v2 --name ziln_v2 --min-anchor-date 2025-04-01
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta

import numpy as np
import polars as pl

from common import MODELS_DIR, VAL_ANCHOR, rmsle
from train_lgbm import load_anchors

DROP_COLS = ["user_id", "anchor_date", "target", "anchor_month", "anchor_doy"]


def preprocess(df: pl.DataFrame, feat_cols: list[str]) -> np.ndarray:
    X = df.select(feat_cols).to_numpy().astype(np.float32)
    names = np.array(feat_cols)
    never_mask = np.array([c.startswith(("days_since", "buy_gap", "overdue")) for c in feat_cols])
    for j in np.where(never_mask)[0]:
        col = X[:, j]
        col[~np.isfinite(col)] = 999.0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(X) * np.log1p(np.abs(X))


def main() -> None:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--name", default="ziln_v2")
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--min-anchor-date", default="2025-04-01")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    t0 = time.time()
    from datetime import date as _date
    from common import lean_load
    cutoff = VAL_ANCHOR - timedelta(days=args.gap_days)
    data = lean_load(
        args.tag, cutoff=cutoff, min_anchor=_date.fromisoformat(args.min_anchor_date),
        val_anchor=VAL_ANCHOR, drop_cols=tuple(DROP_COLS),
    )
    feat_cols = data["feat_cols"]
    never_idx = [j for j, c in enumerate(feat_cols) if c.startswith(("days_since", "buy_gap", "overdue"))]

    def prep(X: np.ndarray) -> np.ndarray:
        for j in never_idx:
            col = X[:, j]
            col[~np.isfinite(col)] = 999.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.sign(X) * np.log1p(np.abs(X))

    X_tr = prep(data["X_tr"])
    X_va = prep(data["X_va"])
    mu_s, sd_s = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-6
    X_tr = (X_tr - mu_s) / sd_s
    X_va = (X_va - mu_s) / sd_s
    y_tr_raw = data["y_tr_raw"]
    y_va_raw = data["y_va_raw"]
    va_users = data["va_users"]
    lt_tr = np.log1p(y_tr_raw).astype(np.float32)
    print(f"loaded in {time.time()-t0:.0f}s  train={X_tr.shape} val={X_va.shape}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}", flush=True)

    class ZilnMLP(nn.Module):
        def __init__(self, d_in: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(d_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.GELU(),
            )
            self.head = nn.Linear(128, 3)

        def forward(self, x):
            h = self.head(self.body(x))
            return h[:, 0], h[:, 1], nn.functional.softplus(h[:, 2]) + 1e-3

    model = ZilnMLP(X_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(lt_tr)
    pos_t = torch.from_numpy((y_tr_raw > 0).astype(np.float32))
    n = len(Xt)
    Xv = torch.from_numpy(X_va).to(dev)

    best = (1e9, None)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb, pb = Xt[idx].to(dev), yt[idx].to(dev), pos_t[idx].to(dev)
            logit, mu, sig = model(xb)
            loss = bce(logit, pb)
            m = pb > 0
            if m.any():
                nll = 0.5 * torch.log(2 * torch.pi * sig[m] ** 2) + (yb[m] - mu[m]) ** 2 / (2 * sig[m] ** 2)
                loss = loss + nll.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xv), 65536):
                logit, mu, sig = model(Xv[i:i + 65536])
                preds.append((torch.sigmoid(logit) * mu).float().cpu().numpy())
            lp = np.clip(np.concatenate(preds), 0, None)
        score = rmsle(y_va_raw, np.expm1(lp))
        print(f"epoch {ep+1}/{args.epochs}  loss={tot/n:.4f}  val RMSLE={score:.5f}", flush=True)
        if score < best[0]:
            best = (score, lp.copy())
            torch.save(model.state_dict(), MODELS_DIR / f"{args.name}.pt")

    print(f"\n[ZILN] best val RMSLE = {best[0]:.5f}", flush=True)
    pl.DataFrame({"user_id": va_users, "pred": np.expm1(best[1]), "target": y_va_raw}).write_parquet(
        MODELS_DIR / f"{args.name}_valpred.parquet"
    )
    np.savez(MODELS_DIR / f"{args.name}_scaler.npz", mu=mu_s, sd=sd_s)
    (MODELS_DIR / f"{args.name}.meta.json").write_text(
        json.dumps(dict(name=args.name, val_rmsle=best[0], feat_cols=feat_cols), indent=2)
    )


if __name__ == "__main__":
    main()

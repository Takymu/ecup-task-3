"""User-fold OOF stacking of the tab MLP: NN prediction as a GBM feature.

5 folds BY USER (no cross-anchor leakage): each fold model trains on 4/5 of users
(stride-7 anchors <= cutoff), predicts the held-out 1/5 at EVERY train/val anchor.
The test anchor gets predictions from a full-data model. Output extras tag:
FEATURES_DIR/{out_tag}/anchor_*.parquet (user_id, tab_lp) — joinable via lean_load extras.

Usage: python train_tab_stack.py [--folds 5 --n-anchors 16 --epochs 8]
Then:  python train_lgbm.py --tag v2 --boosting dart ... --extras v2_tabstack
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from common import FEATURES_DIR, TEST_ANCHOR, VAL_ANCHOR
from train_seq import SeqZiln

DROP = ("user_id", "anchor_date", "target", "anchor_month", "anchor_doy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out-tag", default="v2_tabstack")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-anchors", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    files = sorted((FEATURES_DIR / args.tag).glob("anchor_*.parquet"))
    dates = [date.fromisoformat(f.stem.removeprefix("anchor_")) for f in files]
    cutoff = VAL_ANCHOR - timedelta(days=30)
    train_dates = sorted([d for d in dates if d <= cutoff])[-args.n_anchors:]
    pred_dates = [d for d in dates if d <= VAL_ANCHOR]

    feat_cols = None
    users = None
    X = {}   # date -> sign-log features (float32, raw scale before standardization)
    Y = {}   # date -> target raw (nan where null)
    alive = {}  # date -> bool mask: user existed at anchor
    for f, d in zip(files, dates):
        if d not in set(train_dates) | set(pred_dates) | {TEST_ANCHOR}:
            continue
        df = pl.read_parquet(f).sort("user_id")
        if feat_cols is None:
            feat_cols = [c for c in df.columns if c not in DROP]
            users = df["user_id"].to_numpy()
        arr = df.select(feat_cols).to_numpy().astype(np.float32)
        X[d] = np.sign(arr) * np.log1p(np.abs(arr))
        Y[d] = df["target"].to_numpy().astype(np.float64) if d != TEST_ANCHOR else None
        alive[d] = df["days_active_total"].is_not_null().to_numpy()
        print(f"  loaded {d} ({time.time()-t0:.0f}s)", flush=True)
    F = len(feat_cols)
    n_users = len(users)

    rng = np.random.default_rng(args.seed)
    fold_id = rng.permutation(n_users) % args.folds

    def build_train(mask_users: np.ndarray):
        Xs, ps, ls = [], [], []
        for d in train_dates:
            m = mask_users & alive[d] & ~np.isnan(Y[d])
            Xs.append(X[d][m])
            y = np.clip(Y[d][m], 0, None)
            ps.append((y > 0).astype(np.float32))
            ls.append(np.log1p(y).astype(np.float32))
        return np.concatenate(Xs), np.concatenate(ps), np.concatenate(ls)

    def fit(X_tr, p_tr, l_tr):
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-6
        Xn = np.nan_to_num((X_tr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        model = SeqZiln(hidden=128, layers=1, cell="tab", feat_dim=F).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        bce = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(Xn)
        pt = torch.from_numpy(p_tr)
        yt = torch.from_numpy(l_tr)
        n = len(Xt)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, args.batch):
                idx = perm[i:i + args.batch]
                fb, pb, yb = Xt[idx].to(dev), pt[idx].to(dev), yt[idx].to(dev)
                logit, mu_p, sig = model(None, fb)
                loss = bce(logit, pb)
                m = pb > 0
                if m.any():
                    nll = 0.5 * torch.log(2 * torch.pi * sig[m] ** 2) + (yb[m] - mu_p[m]) ** 2 / (2 * sig[m] ** 2)
                    loss = loss + nll.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        return model, mu, sd

    @torch.no_grad()
    def predict(model, mu, sd, Xa: np.ndarray) -> np.ndarray:
        model.eval()
        Xn = np.nan_to_num((Xa - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        out = []
        for i in range(0, len(Xn), 65536):
            fb = torch.from_numpy(Xn[i:i + 65536]).to(dev)
            logit, mu_p, _ = model(None, fb)
            out.append((torch.sigmoid(logit) * mu_p).float().cpu().numpy())
        return np.clip(np.concatenate(out), 0, None)

    oof = {d: np.full(n_users, np.nan, dtype=np.float32) for d in pred_dates}
    for f in range(args.folds):
        tr_mask = fold_id != f
        X_tr, p_tr, l_tr = build_train(tr_mask)
        model, mu, sd = fit(X_tr, p_tr, l_tr)
        del X_tr, p_tr, l_tr
        ho = np.where(~tr_mask)[0]
        for d in pred_dates:
            oof[d][ho] = predict(model, mu, sd, X[d][ho])
        print(f"fold {f}: done ({time.time()-t0:.0f}s)", flush=True)

    # full model for the test anchor
    X_tr, p_tr, l_tr = build_train(np.ones(n_users, dtype=bool))
    model, mu, sd = fit(X_tr, p_tr, l_tr)
    del X_tr, p_tr, l_tr
    test_lp = predict(model, mu, sd, X[TEST_ANCHOR])
    print(f"full model + test pred done ({time.time()-t0:.0f}s)", flush=True)

    out_dir = FEATURES_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in pred_dates:
        assert not np.isnan(oof[d]).any()
        pl.DataFrame({"user_id": users, "tab_lp": oof[d]}).write_parquet(
            out_dir / f"anchor_{d}.parquet")
    pl.DataFrame({"user_id": users, "tab_lp": test_lp}).write_parquet(
        out_dir / f"anchor_{TEST_ANCHOR}.parquet")
    print(f"saved {len(pred_dates)+1} extras files to {out_dir}  "
          f"(val-anchor mean_lp={oof[VAL_ANCHOR].mean():.4f})", flush=True)


if __name__ == "__main__":
    main()

"""Test-anchor prediction (and optional +30d refit) for any train_seq model.

Architecture (cell/hidden/layers/pool/unit/seq_len/fusion) is read from {name}.meta.json,
so this handles GRU h128, h256x2, daily tokenization, tab MLP etc. uniformly.

Usage:
  python predict_seq.py --name l_gru_h256x2_a16 --refit   # +30d refit -> *_full, predict test
  python predict_seq.py --name srv_seq                    # predict with validated weights
Output: artifacts/models/{stem}_testpred.parquet (user_id, pred in raw GMV scale)
"""
from __future__ import annotations
import os

import argparse
import json
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from common import FEATURES_DIR, MODELS_DIR, TRAIN_PARQUET, TEST_ANCHOR
from train_seq import CH, WEEKS, SeqZiln, cached_tensor, ctx_width, load_feats, targets_at, ziln_loss, ziln_loss_w, fresh_anchor_dates, main_loss, hist_drop_, ziln_loss_w2, daily_cumsum, step_targets, step_loss, stack_free, obs_channel, finish_tensor, model_seq_len, rec_drop_, drop_recent_np, bins_loss, bins_readout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="srv_seq")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--epochs", type=int, default=0, help="refit epochs; 0 = same as training (8)")
    ap.add_argument("--avg-last", type=int, default=0,
                    help="refit: average test predictions of the last K epochs (0 = final epoch only); stem gets _avgK")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--anchor-offset-days", type=int, default=-1,
                    help="override meta anchor_offset_days for the refit grid (-1 = use meta)")
    ap.add_argument("--n-anchors", type=int, default=0, help="override meta n_anchors for refit (0 = use meta)")
    ap.add_argument("--stem-suffix", default="", help="extra suffix for output stem (e.g. _dec)")
    ap.add_argument("--pred-anchor", default="", help="ISO date: predict at this anchor instead of TEST_ANCHOR (no refit)")
    ap.add_argument("--dump-heads", action="store_true", help="also save sigmoid(logit) and mu of the main head (for logit-shift forms)")
    ap.add_argument("--hist-len", type=int, default=0, help="inference with only the last L weekly tokens (older zeroed; TTA by memory length)")
    ap.add_argument("--drop-recent", type=int, default=0, help="TTA view: hide the last D days of the input (rec-drop models)")
    args = ap.parse_args()

    meta_path = MODELS_DIR / f"{args.name}.meta.json"
    meta = json.loads(meta_path.read_text())
    cell = meta.get("cell", "gru")
    hidden = meta.get("hidden", 128)
    layers = meta.get("layers", 1)
    pool = meta.get("pool", "mean")
    unit = meta.get("unit", "week")
    seq_len = meta.get("seq_len", WEEKS)
    n_anchors = meta.get("n_anchors", 8)
    anchor_stride = meta.get("anchor_stride", 2)
    step_days = int(meta.get("anchor_step_days", 0) or 0) or 7 * anchor_stride
    lr = meta.get("lr", 2e-3)
    fusion_tag = meta.get("fusion_tag", "") or ""
    feat_cols = meta.get("feat_cols")
    n_ch = meta.get("ch", 8)
    tail_days = int(meta.get("tail_days", 0) or 0)
    ctx = bool(meta.get("ctx", False))
    ctx_set = meta.get("ctx_set") or "v1"
    ctx = ctx_set if ctx else False
    if ctx == "v1f":
        import train_seq as _ts
        _ts.FCTX_PROXY_FROM = TEST_ANCHOR  # refit anchors (<= 14.01) use the true window, the test anchor the proxy
    n_ch_model = n_ch + (1 if tail_days else 0) + (ctx_width(ctx) if ctx else 0)
    loss_kind = meta.get("loss", "ziln"); mix_w = float(meta.get("mix_w", 1.0))
    user_emb = int(meta.get("user_emb", 0) or 0)
    anchor_offset_days = int(meta.get("anchor_offset_days", 0) or 0)
    if args.anchor_offset_days >= 0:
        anchor_offset_days = args.anchor_offset_days
    if args.n_anchors > 0:
        n_anchors = args.n_anchors
    target_kind = meta.get("target", "gmv")
    aux_h = list(meta.get("aux_horizons", []) or [])
    aux_specs = [(h, "gmv", 0) for h in aux_h] + [(30, kd, 0) for kd in str(meta.get("aux_count", "") or "").split(",") if kd] + \
                [(30, "gmv", int(o)) for o in str(meta.get("aux_lead", "") or "").split(",") if o] + \
                [(int(hw.split(":")[0]), "gmv", int(hw.split(":")[1])) for hw in str(meta.get("aux_win", "") or "").split(",") if hw]
    pmask_from = date.fromisoformat(meta["pmask_from"]) if meta.get("pmask_from") else None
    obs_mask = bool(meta.get("obs_mask", False)); rec_mask = int(meta.get("rec_mask", 0) or 0); afe = bool(meta.get("afe", False))
    back_readouts = [int(k) for k in str(meta.get("back_readouts", "") or "").split(",") if k]; back_hidden = int(meta.get("back_hidden", 128) or 128)
    import train_seq as _ts0
    _ts0.SIGMA_FIXED = bool(meta.get("sigma_fixed", False))
    gmv_noise = float(meta.get("gmv_noise", 0) or 0)
    step_sup = float(meta.get("step_sup", 0) or 0)
    aux_w = float(meta.get("aux_weight", 0.3))
    unit_days = 7 if unit == "week" else 1
    store_dtype = np.float16 if unit == "day" else np.float32
    epochs = args.epochs or 8
    print(f"[{args.name}] cell={cell} h{hidden}x{layers} pool={pool} {unit}x{seq_len} "
          f"anchors={n_anchors} fusion={fusion_tag or '-'}", flush=True)

    t0 = time.time()
    df = pl.scan_parquet(TRAIN_PARQUET).select(
        ["event_date", "user_id", "gmv", "searches", "to_cart", "to_ord", "search", "cat"]
    ).collect()
    users = df["user_id"].unique().sort().to_numpy()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = len(feat_cols) if (fusion_tag and feat_cols) else 0
    n_users = len(users)
    short_tokens = int(meta.get("short_tokens", 0) or 0); hist_drop = float(meta.get("hist_drop", 0) or 0); hist_min = int(meta.get("hist_min", 6) or 6)
    n_ch_model += 1 if obs_mask else 0
    rank_ch = [int(c) for c in str(meta.get("rank_ch", "") or "").split(",") if c]; cart_ch = bool(meta.get("cart_ch", False))
    coarse = int(meta.get("coarse", 0) or 0); coarse_fine = int(meta.get("coarse_fine", 16) or 16)
    rec_drop = float(meta.get("rec_drop", 0) or 0); rec_drop_max = int(meta.get("rec_drop_max", 28) or 28)
    n_bins = int(meta.get("bins", 0) or 0); bins_w = float(meta.get("bins_w", 1.0) or 1.0)
    bins_edges = torch.tensor(meta["bins_edges"], device=dev) if n_bins else None; bins_centers = torch.tensor(meta["bins_centers"], device=dev) if n_bins else None
    n_ch_model += (2 if cart_ch else 0) + len(rank_ch) + (1 if coarse else 0)
    seq_len_m = model_seq_len(seq_len, tail_days, coarse, coarse_fine)
    fin = dict(obs_mask=obs_mask, rank_ch=rank_ch, cart=cart_ch, coarse=coarse, fine=coarse_fine, n_sum=min(n_ch, 11))
    model = SeqZiln(hidden, layers, cell, seq_len_m, feat_dim, pool, n_ch_model, n_aux=len(aux_specs),
                    n_users=n_users, user_emb=user_emb, short_tokens=short_tokens, obs_pool=obs_mask,
                    back_readouts=back_readouts, back_hidden=back_hidden, n_bins=n_bins).to(dev)
    init_from = meta.get("init_from") or ""
    if args.refit and init_from:  # warm start of the refit from the same pretrained weights as the validated run
        sd0 = torch.load(MODELS_DIR / f"{init_from}.pt", map_location=dev); own = model.state_dict()
        ok = {k: v for k, v in sd0.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(ok, strict=False); print(f"refit init-from {init_from}: {len(ok)}/{len(own)} tensors", flush=True)
    uid_all = torch.arange(n_users)
    stem = args.name + args.stem_suffix
    PRED_ANCHOR = date.fromisoformat(args.pred_anchor) if args.pred_anchor else TEST_ANCHOR
    mu = sd = None

    if args.refit:
        torch.manual_seed(meta.get("seed", 42))
        np.random.seed(meta.get("seed", 42))
        cutoff = TEST_ANCHOR - timedelta(days=30)
        if fusion_tag:
            fa = sorted(date.fromisoformat(f.stem.removeprefix("anchor_"))
                        for f in (FEATURES_DIR / fusion_tag).glob("anchor_*.parquet"))
            anchors = sorted([d for d in fa if d <= cutoff])[-n_anchors:]
        else:
            anchors = []
            a = cutoff - timedelta(days=anchor_offset_days)
            while len(anchors) < n_anchors:
                anchors.append(a)
                a -= timedelta(days=step_days)
            anchors = anchors[::-1]
        fresh_step = int(meta.get("fresh_step", 0) or 0)
        fresh = fresh_anchor_dates(cutoff, TEST_ANCHOR, fresh_step, aux_h)
        print("refit anchors:", anchors, flush=True)
        if fresh:
            print("fresh anchors (aux-only):", fresh, flush=True)

        Xs, ys, Fs = [], [], []
        ys_aux = [[] for _ in aux_specs]
        w_main, w_aux, w_bce, a_idx = [], [[] for _ in aux_specs], [], []
        for ai, a in enumerate(anchors + fresh):
            xa = cached_tensor(df, users, a, seq_len, unit_days, store_dtype, n_ch, tail_days, ctx).astype(np.float16)
            xa = finish_tensor(xa, df, users, a, seq_len, tail_days, **fin)
            Xs.append(xa)
            is_fresh = a in fresh
            ys.append(np.zeros(len(users)) if is_fresh else targets_at(df, users, a, kind=target_kind))
            w_main.append(np.full(len(users), 0.0 if is_fresh else 1.0, dtype=np.float32))
            wb = np.full(len(users), 0.0 if (is_fresh or (pmask_from and a >= pmask_from)) else 1.0, dtype=np.float32)
            if rec_mask:
                last = pl.DataFrame({"user_id": users}).join(df.filter(pl.col("event_date") <= a).group_by("user_id").agg(pl.col("event_date").max().alias("l")), on="user_id", how="left")["l"]
                rec = np.where(last.is_not_null().to_numpy(), (np.datetime64(a) - last.to_numpy().astype("datetime64[D]")).astype("timedelta64[D]").astype(np.int64), 10**6)
                wb = wb * (rec <= rec_mask).astype(np.float32)
            w_bce.append(wb); a_idx.append(np.full(len(users), ai, dtype=np.int64))
            for k, (h, kd, off) in enumerate(aux_specs):
                ok = a + timedelta(days=h + off) <= TEST_ANCHOR
                ys_aux[k].append(targets_at(df, users, a, h, kind=kd, offset=off) if ok else np.zeros(len(users)))
                w_aux[k].append(np.full(len(users), 1.0 if ok else 0.0, dtype=np.float32))
            if fusion_tag:
                fa_arr, feat_cols = load_feats(fusion_tag, a, users, feat_cols)
                Fs.append(fa_arr)
            print(f"  built {a} ({time.time()-t0:.0f}s)", flush=True)
        X_tr = stack_free(Xs)
        y_raw = np.clip(np.concatenate(ys), 0, None)
        y_aux_raw = [np.clip(np.concatenate(v), 0, None) for v in ys_aux]
        del Xs, ys, ys_aux
        Ft = None
        if fusion_tag:
            F_tr = np.concatenate(Fs)
            del Fs
            mu = np.nanmean(F_tr, axis=0)
            sd = np.nanstd(F_tr, axis=0) + 1e-6
            F_tr = np.nan_to_num((F_tr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
            Ft = torch.from_numpy(F_tr)
        lt_tr = np.log1p(y_raw).astype(np.float32)
        print(f"refit tensor {X_tr.shape}", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        bce = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(X_tr)
        if str(dev) == "cuda" and not os.environ.get("SEQ_CPU_TENSOR"):
            nbytes = Xt.numel() * Xt.element_size(); free = torch.cuda.mem_get_info()[0]
            if nbytes + 12e9 < free:
                Xt = Xt.to(dev); del X_tr; print(f"refit tensor on GPU ({nbytes/1e9:.1f} GB, free was {free/1e9:.1f} GB)", flush=True)
            else:
                print(f"refit tensor stays on CPU ({nbytes/1e9:.1f} GB vs free {free/1e9:.1f} GB)", flush=True)
        yt = torch.from_numpy(lt_tr)
        pt = torch.from_numpy((y_raw > 0).astype(np.float32))
        yt_aux = [torch.from_numpy(np.log1p(v).astype(np.float32)) for v in y_aux_raw]
        pt_aux = [torch.from_numpy((v > 0).astype(np.float32)) for v in y_aux_raw]
        w_main = torch.from_numpy(np.concatenate(w_main)); w_aux = [torch.from_numpy(np.concatenate(v)) for v in w_aux]; w_bce = torch.from_numpy(np.concatenate(w_bce))
        weighted = bool(fresh) or pmask_from is not None or bool(rec_mask)
        at_tr = torch.from_numpy(np.concatenate(a_idx)); fe = torch.zeros(int(at_tr.max()) + 1, 2, device=dev, requires_grad=True) if afe else None
        yt_step = None
        if step_sup > 0:
            n_w_steps = seq_len - tail_days; Cd = daily_cumsum(df, users)
            yt_step = torch.from_numpy(np.concatenate([step_targets(Cd, a, n_w_steps, tail_days) for a in anchors + fresh])); del Cd
            print(f"step-sup refit: targets {tuple(yt_step.shape)}", flush=True)
        n = len(Xt)
        avg_acc = None; n_avg = 0
        if args.avg_last:
            X_te_early = finish_tensor(cached_tensor(df, users, TEST_ANCHOR, seq_len, unit_days, store_dtype, n_ch, tail_days, ctx), df, users, TEST_ANCHOR, seq_len, tail_days, **fin)
            F_te_early = None
            if fusion_tag:
                F_te_early, _ = load_feats(fusion_tag, TEST_ANCHOR, users, feat_cols)
                F_te_early = torch.from_numpy(np.nan_to_num((F_te_early - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0))
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, args.batch):
                idx = perm[i:i + args.batch]
                xb, yb, pb = Xt[idx.to(Xt.device)].to(dev).float(), yt[idx].to(dev), pt[idx].to(dev)
                if hist_drop > 0:
                    hist_drop_(xb, seq_len_m - tail_days, hist_drop, hist_min)
                if rec_drop > 0:
                    rec_drop_(xb, seq_len_m - tail_days, tail_days, rec_drop, rec_drop_max)
                if gmv_noise > 0:
                    xb[:, :, 0] += gmv_noise * torch.randn_like(xb[:, :, 0]) * (xb[:, :, 0] > 0).float()
                if ctx == "v1f" and float(meta.get("fctx_noise", 0) or 0) > 0:
                    xb[:, :, -2:] += float(meta["fctx_noise"]) * torch.randn(len(idx), 1, 2, device=dev)
                fb = Ft[idx].to(dev) if Ft is not None else None
                ub = (idx % n_users).to(dev) if user_emb else None
                (logit, mu_p, sig), aux = model(xb, fb, return_aux=True, uid=ub)
                if fe is not None:
                    ab = at_tr[idx].to(dev); logit = logit + fe[ab, 0]; mu_p = mu_p + fe[ab, 1]
                if weighted:
                    loss = ziln_loss_w2(logit, mu_p, sig, yb, pb, w_bce[idx].to(dev), w_main[idx].to(dev))
                    for k, (al, am, asg) in enumerate(aux):
                        loss = loss + aux_w * ziln_loss_w(al, am, asg, yt_aux[k][idx].to(dev),
                                                          pt_aux[k][idx].to(dev), w_aux[k][idx].to(dev))
                else:
                    loss = main_loss(logit, mu_p, sig, yb, pb, bce, loss_kind, mix_w)
                    for k, (al, am, asg) in enumerate(aux):
                        loss = loss + aux_w * ziln_loss(al, am, asg, yt_aux[k][idx].to(dev),
                                                        pt_aux[k][idx].to(dev), bce)
                if yt_step is not None:
                    loss = loss + step_loss(model.step_outputs(n_w_steps), yt_step[idx].to(dev).float(), step_sup)
                if n_bins:
                    loss = loss + bins_w * bins_loss(model._bins_logits, yb, bins_edges, w_main[idx].to(dev) if weighted else None)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            print(f"refit epoch {ep+1}/{epochs} done ({time.time()-t0:.0f}s)", flush=True)
            if args.avg_last and ep + 1 > epochs - args.avg_last:
                model.eval()
                ebs = args.batch if unit == "day" else 32768
                with torch.no_grad():
                    acc = []
                    for i in range(0, len(X_te_early), ebs):
                        fvb = F_te_early[i:i + ebs].to(dev) if F_te_early is not None else None
                        ueb = uid_all[i:i + ebs].to(dev) if user_emb else None
                        lg, mp, _ = model(torch.from_numpy(X_te_early[i:i + ebs]).to(dev).float(), fvb, uid=ueb)
                        acc.append((bins_readout(model._bins_logits, bins_centers) if n_bins else torch.sigmoid(lg) * mp).float().cpu().numpy())
                lp_ep = np.concatenate(acc)
                avg_acc = lp_ep if avg_acc is None else avg_acc + lp_ep; n_avg += 1
                print(f"  epoch {ep+1} test mean_lp={lp_ep.mean():.4f} (accumulated {n_avg})", flush=True)
        del Xt, yt, pt
        try: del X_tr
        except NameError: pass
        stem = f"{args.name}_full" + (f"_avg{args.avg_last}" if args.avg_last else "") + args.stem_suffix
        torch.save(model.state_dict(), MODELS_DIR / f"{stem}.pt")
    else:
        model.load_state_dict(torch.load(MODELS_DIR / f"{args.name}.pt", map_location=dev))
        if fusion_tag:
            fn = np.load(MODELS_DIR / f"{args.name}_fnorm.npz")
            mu, sd = fn["mu"], fn["sd"]

    X_te = cached_tensor(df, users, PRED_ANCHOR, seq_len, unit_days, store_dtype, n_ch, tail_days, ctx)
    X_te = finish_tensor(X_te, df, users, PRED_ANCHOR, seq_len, tail_days, **fin)
    if args.drop_recent:
        X_te = drop_recent_np(X_te, seq_len_m - tail_days, tail_days, args.drop_recent)
        print(f"drop-recent {args.drop_recent}: hid tokens ending within {args.drop_recent} days of the anchor", flush=True)
    if args.hist_len:
        n_w = seq_len_m - tail_days; X_te = X_te.copy(); X_te[:, :max(n_w - args.hist_len, 0)] = 0
        print(f"hist-len {args.hist_len}: zeroed first {max(n_w - args.hist_len, 0)} weekly tokens", flush=True)
    Fv = None
    if fusion_tag:
        F_te, _ = load_feats(fusion_tag, TEST_ANCHOR, users, feat_cols)
        F_te = np.nan_to_num((F_te - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        Fv = torch.from_numpy(F_te)
    del df
    model.eval()
    eval_bs = args.batch if unit == "day" else 8192
    with torch.no_grad():
        while True:  # OOM-safe inference
            try:
                lps = []
                for i in range(0, len(X_te), eval_bs):
                    fvb = Fv[i:i + eval_bs].to(dev) if Fv is not None else None
                    ueb = uid_all[i:i + eval_bs].to(dev) if user_emb else None
                    logit, mu_p, sig = model(torch.from_numpy(X_te[i:i + eval_bs]).to(dev).float(), fvb, uid=ueb)
                    lps.append((bins_readout(model._bins_logits, bins_centers) if n_bins else torch.sigmoid(logit) * mu_p).float().cpu().numpy())
                    if target_kind == "active":
                        pacts = (pacts if i else []) + [torch.sigmoid(logit).float().cpu().numpy()]
                    if args.dump_heads:
                        heads = (heads if i else []) + [np.stack([torch.sigmoid(logit).float().cpu().numpy(), mu_p.float().cpu().numpy()], 1)]
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); eval_bs = max(eval_bs // 2, 512); print(f"eval OOM -> eval_bs {eval_bs}", flush=True)
    lp = np.clip(np.concatenate(lps), 0, None)
    if args.dump_heads:
        hh = np.concatenate(heads); pl.DataFrame({"user_id": users, "p": hh[:, 0], "mu": hh[:, 1]}).write_parquet(MODELS_DIR / f"{stem}_heads.parquet")
        print(f"saved {stem}_heads.parquet  mean p={hh[:,0].mean():.4f} mean mu={hh[:,1].mean():.3f}", flush=True)
    if args.refit and args.avg_last and n_avg:
        lp = np.clip(avg_acc / n_avg, 0, None)  # mean over the last K epochs (log-space p*mu)
        print(f"averaged {n_avg} epochs", flush=True)
    if target_kind == "active":
        pl.DataFrame({"user_id": users, "p_act": np.concatenate(pacts).astype(np.float64)}).write_parquet(
            MODELS_DIR / f"{stem}_pact_{TEST_ANCHOR}.parquet")
        print(f"saved {stem}_pact_{TEST_ANCHOR}.parquet", flush=True)
    pl.DataFrame({"user_id": users, "pred": np.expm1(lp).astype(np.float64)}).write_parquet(
        MODELS_DIR / f"{stem}_testpred.parquet"
    )
    print(f"saved {stem}_testpred.parquet  mean_lp={lp.mean():.4f}  zeros={(lp < 1e-6).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()

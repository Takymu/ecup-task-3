"""Sequence models with ZILN head — 'tokenized' user behavior.

Weekly (default): 52 weeks x 8 channels. Daily: --unit day --seq-len 413.
Cells: --cell gru|lstm|transformer.

Usage: python train_seq.py --name srv_seq [--epochs 8] [--anchor-stride 2]
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

from common import ARTIFACTS_DIR, FEATURES_DIR, MODELS_DIR, TRAIN_PARQUET, VAL_ANCHOR, rmsle

FEAT_DROP = ("user_id", "anchor_date", "target", "anchor_month", "anchor_doy")


def load_feats(tag: str, anchor: date, users: np.ndarray, cols: list[str] | None = None):
    """Tabular features for one anchor, aligned to `users` (both sorted by user_id)."""
    df = pl.read_parquet(FEATURES_DIR / tag / f"anchor_{anchor}.parquet").sort("user_id")
    assert len(df) == len(users) and (df["user_id"].to_numpy() == users).all()
    if cols is None:
        cols = [c for c in df.columns if c not in FEAT_DROP]
    arr = df.select(cols).to_numpy().astype(np.float32)
    return np.sign(arr) * np.log1p(np.abs(arr)), cols

WEEKS = 52
CH = 8  # gmv, searches, to_cart, to_ord, active, gmv_days, search_days, cat_days

SEQ_CACHE = ARTIFACTS_DIR / "seq_cache"

# RU e-commerce gift/holiday calendar (data window 2025-01..2026-03): NY run-up + gift days + sales
HOLIDAYS: set[date] = set(
    [date(2025, 1, d) for d in range(1, 9)] + [date(2026, 1, d) for d in range(1, 9)]
    + [date(2025, 2, 14), date(2025, 2, 23), date(2025, 3, 8),
       date(2025, 5, 1), date(2025, 5, 9), date(2025, 6, 12),
       date(2025, 11, 4), date(2025, 11, 11), date(2025, 11, 28),
       date(2025, 12, 31),
       date(2026, 2, 14), date(2026, 2, 23), date(2026, 3, 8)]
)


def calendar_channels(anchor: date, seq_len: int, unit_days: int) -> np.ndarray:
    """(seq_len, 2): [holiday days in bucket, pre-holiday days (1-7d before a holiday)]."""
    out = np.zeros((seq_len, 2), dtype=np.float32)
    for b in range(seq_len):
        lo = (seq_len - 1 - b) * unit_days
        for off in range(unit_days):
            d = anchor - timedelta(days=lo + off)
            if d in HOLIDAYS:
                out[b, 0] += 1
            if any(d + timedelta(days=k) in HOLIDAYS for k in range(1, 8)):
                out[b, 1] += 1
    return np.log1p(out)


def ctx_width(ctx) -> int:
    return {"v1": 2, "v2": 8, "v1d": 4, "v1f": 4, "v1m": 8}["v1" if ctx is True else ctx]


# --- "future context" (ctx-set v1f): tells the model how big the platform season AHEAD is.
# r = log(mean daily platform GMV over the 30 target days / mean over the 14 days before the anchor), same for
# active user-days. Train anchors use the TRUE future window (teacher forcing); anchors >= FCTX_PROXY_FROM (val at
# training time, test at prediction time) use the PROXY = the same ratio one year (364 days, weekday-aligned) earlier.
FCTX_PROXY_FROM: date | None = None


def _win_stats(df: pl.DataFrame, d1: date, d2: date):
    w = df.filter(pl.col("event_date").is_between(d1, d2))
    return float(w["gmv"].sum()), float(len(w))


def fctx_ratio(df: pl.DataFrame, anchor: date) -> np.ndarray:
    use_proxy = FCTX_PROXY_FROM is not None and anchor >= FCTX_PROXY_FROM
    a = anchor - timedelta(days=364) if use_proxy else anchor
    g_n, n_n = _win_stats(df, a + timedelta(days=1), a + timedelta(days=30))
    g_p, n_p = _win_stats(df, a - timedelta(days=13), a)
    if n_n == 0 or n_p == 0:
        raise SystemExit(f"fctx: window not covered by data for anchor {anchor} (proxy={use_proxy})")
    r = np.array([np.log((g_n / 30) / (g_p / 14)), np.log((n_n / 30) / (n_p / 14))], dtype=np.float32)
    print(f"  fctx {anchor}: {'PROXY' if use_proxy else 'true '} r_gmv={r[0]:+.3f} r_act={r[1]:+.3f}", flush=True)
    return r


_MACRO = None


def _macro_arrays():
    """Дневные ряды artifacts/macro_daily.csv (ЦБ + MOEX), 01.01.2025–20.03.2026 — покрывают и тестовое окно."""
    global _MACRO
    if _MACRO is None:
        m = pl.read_csv(MODELS_DIR.parent / "macro_daily.csv").sort("date")
        _MACRO = {c: m[c].to_numpy().astype(np.float64) for c in ("usd", "key_rate", "ozon")}
        _MACRO["n"] = m.height
    return _MACRO


def _macro_feats(d1: date, d2: date) -> np.ndarray:
    """(3,): Δlog USD за окно ×10, ставка ЦБ на конец окна /20, Δlog OZON ×10."""
    m = _macro_arrays()
    di = lambda d: min(max((d - DAY0).days, 0), m["n"] - 1)
    i1, i2 = di(d1), di(d2)
    return np.array([10 * np.log(m["usd"][i2] / m["usd"][i1]), m["key_rate"][i2] / 20.0,
                     10 * np.log(m["ozon"][i2] / m["ozon"][i1])], dtype=np.float32)


def macro_token_channels(anchor: date, seq_len: int, tail_days: int, unit_days: int) -> np.ndarray:
    n_w = seq_len - tail_days if tail_days else seq_len
    out = np.zeros((seq_len, 3), dtype=np.float32)
    for t in range(seq_len):
        if tail_days and t >= n_w:
            end = anchor - timedelta(days=tail_days - 1 - (t - n_w)); ln = 1
        else:
            step = 7 if tail_days else unit_days
            end = (anchor - timedelta(days=tail_days)) - timedelta(days=step * (n_w - 1 - t)); ln = step
        out[t] = _macro_feats(end - timedelta(days=ln), end)
    return out


def macro_future(anchor: date) -> np.ndarray:
    return _macro_feats(anchor, anchor + timedelta(days=30))


def platform_channels(df: pl.DataFrame, anchor: date, seq_len: int, unit_days: int, ctx_set: str = "v1") -> np.ndarray:
    """Platform-level context per token, same for every user.
    v1 (seq_len, 2): log1p(active users/day), log1p(GMV/day).
    v2 (seq_len, 8): + log1p(buyers/day), log1p(orders/day), log1p(AOV per buyer), conversion buyers/active,
    and detrended v1 pair (log minus trailing 4-token mean = local platform anomaly)."""
    hist = df.filter((pl.col("event_date") <= anchor) & (pl.col("event_date") > anchor - timedelta(days=unit_days * seq_len))
                     ).with_columns(((pl.lit(anchor) - pl.col("event_date")).dt.total_days() // unit_days).alias("wk"))
    agg = hist.group_by("wk").agg(pl.len().alias("n"), pl.col("gmv").sum().alias("g"),
                                  (pl.col("gmv") > 0).sum().alias("b"), pl.col("to_ord").sum().alias("o"))
    nch = ctx_width(ctx_set)
    out = np.zeros((seq_len, nch), dtype=np.float32)
    if ctx_set == "v1f":  # v1 here; the 2 constant future-context channels are appended in cached_tensor
        nch = 2
    if ctx_set == "v1m":  # v1 here; 3 token-macro + 3 future-macro channels are appended in cached_tensor
        nch = 2
    if ctx_set == "v1d":  # v1 + anchor day-of-year (sin/cos), constant over tokens: tells the model WHEN the target window is
        doy = anchor.timetuple().tm_yday
        out[:, 2] = np.sin(2 * np.pi * doy / 365.25)
        out[:, 3] = np.cos(2 * np.pi * doy / 365.25)
        nch = 2
    wk = agg["wk"].to_numpy().astype(np.int64)
    idx = seq_len - 1 - wk
    n_ = agg["n"].to_numpy() / unit_days
    g_ = agg["g"].to_numpy() / unit_days
    out[idx, 0] = np.log1p(n_)
    out[idx, 1] = np.log1p(g_)
    if nch > 2:
        b_ = agg["b"].to_numpy() / unit_days
        o_ = agg["o"].to_numpy() / unit_days
        out[idx, 2] = np.log1p(b_)
        out[idx, 3] = np.log1p(o_)
        out[idx, 4] = np.log1p(g_ / np.maximum(b_, 1e-6))
        out[idx, 5] = b_ / np.maximum(n_, 1e-6)
        for dst, src in ((6, 0), (7, 1)):
            x = out[:, src]
            cs = np.concatenate([[0.0], np.cumsum(x)])
            t = np.arange(seq_len)
            lo = np.maximum(t - 3, 0)
            out[:, dst] = x - (cs[t + 1] - cs[lo]) / (t + 1 - lo)
    return out


def cached_tensor(df: pl.DataFrame, users: np.ndarray, anchor: date,
                  seq_len: int, unit_days: int, dtype, n_ch: int = CH, tail_days: int = 0, ctx=False) -> np.ndarray:
    SEQ_CACHE.mkdir(exist_ok=True)
    suffix = "" if n_ch == CH else f"_c{n_ch}"  # 8-ch keeps legacy names (cache reuse)
    hyb = f"_t{tail_days}" if tail_days else ""
    f = SEQ_CACHE / f"u{unit_days}_l{seq_len}{suffix}{hyb}_{anchor}.npy"
    if f.exists():
        arr = np.load(f)
    else:
        if tail_days:
            arr = build_hybrid_tensor(df, users, anchor, seq_len, tail_days, dtype, n_ch)
        else:
            arr = build_anchor_tensor(df, users, anchor, seq_len, unit_days, dtype, n_ch)
        np.save(f, arr)
    if ctx:  # platform context channels appended (computed on the fly, not cached)
        cs = "v1" if ctx is True else ctx
        if tail_days:
            n_w = seq_len - tail_days
            pc = np.concatenate([platform_channels(df, anchor - timedelta(days=tail_days), n_w, 7, cs),
                                 platform_channels(df, anchor, tail_days, 1, cs)], axis=0)
        else:
            pc = platform_channels(df, anchor, seq_len, unit_days, cs)
        if cs == "v1f":
            pc = np.concatenate([pc[:, :2], np.broadcast_to(fctx_ratio(df, anchor), (seq_len, 2))], axis=1)
        if cs == "v1m":  # macro moves per token + TRUE target-window macro (real-world data covers the test window)
            pc = np.concatenate([pc[:, :2], macro_token_channels(anchor, seq_len, tail_days, unit_days),
                                 np.broadcast_to(macro_future(anchor), (seq_len, 3)).copy()], axis=1)
        arr = np.concatenate([arr, np.broadcast_to(pc.astype(arr.dtype), (arr.shape[0], seq_len, ctx_width(cs)))], axis=2)
    return arr


def build_anchor_tensor(df: pl.DataFrame, users: np.ndarray, anchor: date,
                        seq_len: int = WEEKS, unit_days: int = 7,
                        dtype=np.float32, n_ch: int = CH) -> np.ndarray:
    hist = df.filter(
        (pl.col("event_date") <= anchor)
        & (pl.col("event_date") > anchor - timedelta(days=unit_days * seq_len))
    ).with_columns(
        ((pl.lit(anchor) - pl.col("event_date")).dt.total_days() // unit_days).alias("wk"),
        pl.col("event_date").dt.weekday().alias("wd"),  # Mon=1 .. Sun=7
    )
    aggs = [
        pl.col("gmv").sum().alias("c0"),
        pl.col("searches").sum().alias("c1"),
        pl.col("to_cart").sum().alias("c2"),
        pl.col("to_ord").sum().alias("c3"),
        pl.len().alias("c4"),
        (pl.col("gmv") > 0).sum().alias("c5"),
        pl.col("search").sum().alias("c6"),
        pl.col("cat").sum().alias("c7"),
    ]
    if n_ch >= 9:  # visit-only days (no search, no catalog) — 15% of rows, EDA insight #2
        aggs.append(((pl.col("search") == 0) & (pl.col("cat") == 0)).sum().alias("c8"))
    if n_ch >= 11:  # Fri-Sun activity: weekend-buyer signature
        wknd = pl.col("wd") >= 5
        aggs += [
            pl.col("gmv").filter(wknd).sum().alias("c9"),
            wknd.sum().alias("c10"),
        ]
    agg = hist.group_by("user_id", "wk").agg(aggs)
    uidx = {u: i for i, u in enumerate(users)}
    arr = np.zeros((len(users), seq_len, n_ch), dtype=dtype)
    ui = agg["user_id"].to_numpy()
    wk = agg["wk"].to_numpy().astype(np.int64)
    vals = agg.select([f"c{i}" for i in range(min(n_ch, 11))]).to_numpy().astype(np.float32)
    vals = np.nan_to_num(vals, nan=0.0)  # filtered aggs yield null for empty sets
    rows = np.fromiter((uidx[u] for u in ui), dtype=np.int64, count=len(ui))
    arr[rows, seq_len - 1 - wk, :min(n_ch, 11)] = np.log1p(vals[:, :min(n_ch, 11)])
    if n_ch >= 13:  # calendar channels, identical for every user (positional covariates)
        arr[:, :, 11:13] = calendar_channels(anchor, seq_len, unit_days).astype(dtype)
    return arr


def build_hybrid_tensor(df: pl.DataFrame, users: np.ndarray, anchor: date, seq_len: int,
                        tail_days: int, dtype=np.float32, n_ch: int = CH) -> np.ndarray:
    """Hybrid tokenization: (seq_len - tail_days) weekly tokens for the period BEFORE the tail,
    then tail_days daily tokens (most recent day last) + one flag channel (1 = daily token).
    Fixes the weekly GRU's blur of the last days (it under-predicts 'active today' users by ~+0.09)."""
    n_w = seq_len - tail_days
    w = build_anchor_tensor(df, users, anchor - timedelta(days=tail_days), n_w, 7, dtype, n_ch)
    d = build_anchor_tensor(df, users, anchor, tail_days, 1, dtype, n_ch)
    arr = np.concatenate([w, d], axis=1)
    flag = np.zeros((len(users), seq_len, 1), dtype=dtype)
    flag[:, n_w:, 0] = 1.0
    return np.concatenate([arr, flag], axis=2)


def targets_at(df: pl.DataFrame, users: np.ndarray, anchor: date, horizon: int = 30,
               kind: str = "gmv", offset: int = 0) -> np.ndarray:
    """kind='gmv': 30d GMV (main task); 'active': active days; 'orddays'/'nord'; 'carts' (sum to_cart); 'searches' (sum searches).
    offset shifts the window: (anchor+offset, anchor+offset+horizon]."""
    KINDS = ("gmv", "active", "orddays", "nord", "carts", "searches")
    assert kind in KINDS, f"unknown target kind {kind}"
    w = df.filter(pl.col("event_date").is_between(anchor + timedelta(days=1 + offset), anchor + timedelta(days=horizon + offset)))
    if kind == "carts":
        t = w.group_by("user_id").agg(pl.col("to_cart").sum().cast(pl.Float64).alias("t"))
    elif kind == "searches":
        t = w.group_by("user_id").agg(pl.col("searches").sum().cast(pl.Float64).alias("t"))
    elif kind == "active":
        t = w.group_by("user_id").agg(pl.len().cast(pl.Float64).alias("t"))
    elif kind == "orddays":  # days with an order in the window (label without the amount lottery)
        t = w.group_by("user_id").agg((pl.col("to_ord") > 0).sum().cast(pl.Float64).alias("t"))
    elif kind == "nord":  # number of orders in the window
        t = w.group_by("user_id").agg(pl.col("to_ord").sum().cast(pl.Float64).alias("t"))
    else:
        t = w.group_by("user_id").agg(pl.col("gmv").sum().alias("t"))
    m = dict(zip(t["user_id"].to_numpy(), t["t"].to_numpy()))
    return np.array([m.get(u, 0.0) for u in users], dtype=np.float64)


class SeqZiln(nn.Module):
    def __init__(self, hidden=128, layers=1, cell="gru", seq_len=WEEKS, feat_dim=0,
                 pool="mean", n_ch=CH, n_aux=0, n_users=0, user_emb=0, short_tokens=0, obs_pool=False,
                 back_readouts=(), back_hidden=128, n_bins=0):
        super().__init__()
        self.n_bins = n_bins  # non-parametric head: softmax over log1p-target bins appended after the ZILN blocks
        self.cell = cell
        self.short_tokens = short_tokens
        self.obs_pool = obs_pool  # last input channel = observability mask -> mean-pool over observed tokens only
        self.back_readouts = tuple(back_readouts)  # multi-scale: extra GRU over the last K tokens (sm10-geometry inside one model)
        self.pool = pool
        self.uemb = nn.Embedding(n_users, user_emb) if user_emb else None  # per-user persistent trait vector
        self.n_aux = n_aux  # auxiliary ZILN heads for shorter horizons (multi-task regularizer)
        drop = 0.1 if layers > 1 else 0.0
        if cell == "tab":
            pass  # tabular-only: no sequence tower at all
        elif cell in ("gru", "gru_attn"):
            self.rnn = nn.GRU(n_ch, hidden, num_layers=layers, batch_first=True, dropout=drop)
        elif cell == "lstm":
            self.rnn = nn.LSTM(n_ch, hidden, num_layers=layers, batch_first=True, dropout=drop)
        elif cell == "tcn":  # dilated 1D conv stack (non-causal, whole sequence is past)
            chs = [n_ch, hidden // 2, hidden, hidden]
            blocks = []
            for i in range(3):
                blocks += [nn.Conv1d(chs[i], chs[i + 1], kernel_size=5, padding=2 * (2 ** i), dilation=2 ** i), nn.GELU()]
            self.tcn = nn.Sequential(*blocks)
        else:  # transformer
            self.proj = nn.Linear(n_ch, hidden)
            self.pos = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=max(2, hidden // 32), dim_feedforward=hidden * 4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            self.tfm = nn.TransformerEncoder(enc, num_layers=layers)
        if cell == "gru_attn":  # one self-attention layer over GRU outputs
            enc = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=max(2, hidden // 32), dim_feedforward=hidden * 2,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            self.attn = nn.TransformerEncoder(enc, num_layers=1)
        if pool == "attn":  # additive attention pooling instead of mean
            self.att_w = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fmlp = None
        head_in = 0 if cell == "tab" else hidden * 2
        if short_tokens:
            self.rnn_s = nn.GRU(n_ch, hidden, num_layers=layers, batch_first=True, dropout=drop)
            head_in += hidden * 2
        if self.back_readouts:  # reverse GRU: output after k reversed tokens = representation of exactly the last k tokens
            self.rnn_b = nn.GRU(n_ch, back_hidden, num_layers=1, batch_first=True)
            head_in += back_hidden * len(self.back_readouts)
        if feat_dim:
            self.fmlp = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.2),
                                      nn.Linear(256, 128), nn.GELU())
            head_in += 128
        if user_emb:
            head_in += user_emb
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.GELU(),
                                  nn.Linear(hidden, 3 * (1 + n_aux) + n_bins))

    def step_outputs(self, n_w: int):
        """Causal readouts at each of the first n_w (weekly) tokens through the shared head: z_t = [out_t, cummean_t]."""
        out = self._out[:, :n_w]
        cm = out.cumsum(1) / torch.arange(1, n_w + 1, device=out.device, dtype=out.dtype).view(1, -1, 1)
        o = self.head(torch.cat([out, cm], dim=2))[..., :3]
        return o[..., 0], o[..., 1], nn.functional.softplus(o[..., 2]) + 1e-3

    @staticmethod
    def _split(o):
        return o[:, 0], o[:, 1], nn.functional.softplus(o[:, 2]) + 1e-3

    def forward(self, x, f=None, return_aux=False, uid=None):
        """Returns (logit, mu, sigma) of the 30d head; with return_aux also a list of aux triples."""
        o = self._trunk_out(x, f, uid)
        self._bins_logits = o[:, -self.n_bins:] if self.n_bins else None
        main = self._split(o[:, :3])
        if not return_aux:
            return main
        aux = [self._split(o[:, 3 * (k + 1):3 * (k + 2)]) for k in range(self.n_aux)]
        return main, aux

    def _trunk_out(self, x, f=None, uid=None):
        if self.cell == "tab":
            z = self.fmlp(f)
            if self.uemb is not None:
                z = torch.cat([z, self.uemb(uid)], dim=1)
            return self.head(z)
        if self.cell == "transformer":
            z = self.tfm(self.proj(x) + self.pos)
            z = torch.cat([z[:, -1], z.mean(dim=1)], dim=1)
        elif self.cell == "tcn":
            o = self.tcn(x.transpose(1, 2))  # (B, hidden, L)
            z = torch.cat([o[:, :, -1], o.mean(dim=2)], dim=1)
        else:
            out, h = self.rnn(x)
            if self.cell == "lstm":
                h = h[0]
            if self.cell == "gru_attn":
                out = self.attn(out)
            if self.pool == "attn":
                w = torch.softmax(self.att_w(out), dim=1)
                pooled = (w * out).sum(dim=1)
            elif self.obs_pool:
                m = x[:, :, -1:].to(out.dtype)
                pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                pooled = out.mean(dim=1)
            z = torch.cat([h[-1], pooled], dim=1)
            if self.back_readouts:
                ob, _ = self.rnn_b(torch.flip(x, [1]))
                z = torch.cat([z] + [ob[:, k - 1] for k in self.back_readouts], dim=1)
            self._out = out  # kept for step_outputs (step supervision)
            if self.short_tokens:
                out_s, h_s = self.rnn_s(x[:, -self.short_tokens:])
                z = torch.cat([z, h_s[-1], out_s.mean(dim=1)], dim=1)
        if self.fmlp is not None:
            z = torch.cat([z, self.fmlp(f)], dim=1)
        if self.uemb is not None:
            z = torch.cat([z, self.uemb(uid)], dim=1)
        return self.head(z)


DAY0 = date(2025, 1, 1)
SIGMA_FIXED = False  # --sigma-fixed: plain squared error for the positive part (no learned sigma reweighting)


def obs_channel(df: pl.DataFrame, users: np.ndarray, anchor: date, seq_len: int, tail_days: int) -> np.ndarray:
    """(N, seq_len, 1) float16: 1 where the token's window is inside the observed span [max(DAY0, first_seen(user)), anchor].
    first_seen is computed from events <= anchor (no leak). Structural zeros (pre-2025 weeks, pre-signup weeks) become distinguishable
    from 'observed but inactive', and the pooling can be normalised by the observed length."""
    fs = df.filter(pl.col("event_date") <= anchor).group_by("user_id").agg(pl.col("event_date").min().alias("fs"))
    x = pl.DataFrame({"user_id": users}).join(fs, on="user_id", how="left")
    seen = x["fs"].is_not_null().to_numpy()
    first = np.where(seen, (x["fs"].to_numpy().astype("datetime64[D]") - np.datetime64(DAY0)).astype("timedelta64[D]").astype(np.int64), 10**6)
    first = np.maximum(first, 0)  # days since DAY0
    n_w = seq_len - tail_days
    ends = np.array([(anchor - DAY0).days - tail_days - 7 * (n_w - 1 - t) for t in range(n_w)] +
                    [(anchor - DAY0).days - (tail_days - 1 - j) for j in range(tail_days)], dtype=np.int64)  # token end day index
    return (ends[None, :] >= first[:, None]).astype(np.float16)[:, :, None]


def cart_channels(df: pl.DataFrame, users: np.ndarray, anchor: date, seq_len: int, tail_days: int) -> np.ndarray:
    """(N, seq_len, 2) float32 log1p day-counts per token: days with a cart but no order (abandoned basket) and days with
    searches but no cart (browsing without intent). Hybrid layout as build_hybrid_tensor; cached like the main tensor."""
    SEQ_CACHE.mkdir(exist_ok=True)
    f = SEQ_CACHE / f"xcart_l{seq_len}_t{tail_days}_{anchor}.npy"
    if f.exists():
        return np.load(f)
    uidx = {u: i for i, u in enumerate(users)}

    def part(a: date, L: int, unit: int) -> np.ndarray:
        hist = df.filter((pl.col("event_date") <= a) & (pl.col("event_date") > a - timedelta(days=unit * L))).with_columns(
            ((pl.lit(a) - pl.col("event_date")).dt.total_days() // unit).alias("wk"))
        agg = hist.group_by("user_id", "wk").agg(((pl.col("to_cart") > 0) & (pl.col("to_ord") == 0)).sum().alias("k0"),
                                                 ((pl.col("searches") > 0) & (pl.col("to_cart") == 0)).sum().alias("k1"))
        out = np.zeros((len(users), L, 2), dtype=np.float32)
        rows = np.fromiter((uidx[u] for u in agg["user_id"].to_numpy()), dtype=np.int64, count=len(agg))
        wk = agg["wk"].to_numpy().astype(np.int64)
        out[rows, L - 1 - wk] = np.log1p(agg.select("k0", "k1").to_numpy().astype(np.float32))
        return out

    n_w = seq_len - tail_days
    arr = np.concatenate([part(anchor - timedelta(days=tail_days), n_w, 7), part(anchor, tail_days, 1)], axis=1) if tail_days else part(anchor, seq_len, 7)
    np.save(f, arr)
    return arr


def rank_channels(arr: np.ndarray, idx) -> np.ndarray:
    """Cross-sectional quantile ranks per token for channels idx: positive values -> rank/n_pos in (0, 1], zeros stay 0.
    The user's own channel loses the platform level/scale drift between anchors (ctx keeps the level as a separate input)."""
    out = np.zeros(arr.shape[:2] + (len(idx),), dtype=np.float32)
    for j, c in enumerate(idx):
        for t in range(arr.shape[1]):
            v = arr[:, t, c].astype(np.float32)
            m = v > 0
            n = int(m.sum())
            if n:
                o = np.argsort(v[m], kind="stable")
                r = np.empty(n, dtype=np.float32)
                r[o] = np.arange(1, n + 1, dtype=np.float32) / n
                out[m, t, j] = r
    return out


def coarsen(arr: np.ndarray, n_w: int, fine: int, group: int, n_sum: int) -> np.ndarray:
    """Multi-resolution tokens: weekly tokens older than the last `fine` weeks are merged in groups of `group` weeks
    (log1p-sum channels [:n_sum] -> log1p of the summed counts, other channels -> mean) + flag channel (1 = coarse token)."""
    old, rest = arr[:, :n_w - fine], arr[:, n_w - fine:]
    k = old.shape[1] // group
    old = old[:, old.shape[1] - k * group:]  # a remainder of the oldest weeks is dropped
    g = old.reshape(old.shape[0], k, group, old.shape[2]).astype(np.float32)
    c = np.concatenate([np.log1p(np.expm1(g[..., :n_sum]).sum(2)), g[..., n_sum:].mean(2)], axis=2)
    new = np.concatenate([c.astype(arr.dtype), rest], axis=1)
    flag = np.zeros(new.shape[:2] + (1,), dtype=arr.dtype)
    flag[:, :k] = 1
    return np.concatenate([new, flag], axis=2)


def model_seq_len(seq_len: int, tail_days: int, coarse: int, fine: int) -> int:
    n_w = seq_len - tail_days
    return ((n_w - fine) // coarse + fine + tail_days) if coarse else seq_len


def finish_tensor(arr, df, users, anchor, seq_len, tail_days, obs_mask=False, rank_ch=(), cart=False, coarse=0, fine=16, n_sum=8):
    """Every post-cache channel step in one place so train / val / refit / test tensors are built identically."""
    if obs_mask:
        arr = np.concatenate([arr, obs_channel(df, users, anchor, seq_len, tail_days).astype(arr.dtype)], axis=2)
    if cart:
        arr = np.concatenate([arr, cart_channels(df, users, anchor, seq_len, tail_days).astype(arr.dtype)], axis=2)
    if rank_ch:
        arr = np.concatenate([arr, rank_channels(arr, rank_ch).astype(arr.dtype)], axis=2)
    if coarse:
        arr = coarsen(arr, seq_len - tail_days, fine, coarse, n_sum)
    return arr


def daily_cumsum(df: pl.DataFrame, users: np.ndarray) -> np.ndarray:
    """C[u, k] = sum of the user's GMV over days with index < k (day index = days since DAY0); shape (N, n_days+1)."""
    d = df.group_by("user_id", "event_date").agg(pl.col("gmv").sum().alias("g"))
    n_days = int((d["event_date"].max() - DAY0).days) + 1
    uidx = {u: i for i, u in enumerate(users)}
    D = np.zeros((len(users), n_days), dtype=np.float32)
    rows = np.fromiter((uidx[u] for u in d["user_id"].to_numpy()), dtype=np.int64, count=d.height)
    cols = (d["event_date"] - DAY0).dt.total_days().to_numpy().astype(np.int64)
    np.add.at(D, (rows, cols), d["g"].to_numpy().astype(np.float32))
    C = np.zeros((len(users), n_days + 1), dtype=np.float64); np.cumsum(D, axis=1, out=C[:, 1:]); return C.astype(np.float32)


def step_targets(C: np.ndarray, anchor: date, n_w: int, tail_days: int, horizon: int = 30, min_hist_days: int = 31) -> np.ndarray:
    """Step supervision ('every week is an anchor'): for weekly token t (0..n_w-1, oldest first) ending at
    e_t = anchor - tail_days - 7*(n_w-1-t), target = log1p(GMV over (e_t, e_t+horizon]); -1 where the token has no history
    (e_t < DAY0 + min_hist_days) or its window is not covered by C. Returns (N, n_w) float16."""
    out = np.full((C.shape[0], n_w), -1.0, dtype=np.float16); n_days = C.shape[1] - 1
    for t in range(n_w):
        e = anchor - timedelta(days=tail_days + 7 * (n_w - 1 - t)); i = (e - DAY0).days
        if i < min_hist_days or i + 1 + horizon > n_days: continue
        out[:, t] = np.log1p(np.clip(C[:, i + 1 + horizon] - C[:, i + 1], 0, None)).astype(np.float16)
    return out


def step_loss(so, ys, lam: float):
    """so = (logit, mu, sig) each (B, n_w); ys (B, n_w) log1p targets with -1 = masked. Weighted ZILN over all tokens."""
    logit, mu, sig = so; w = (ys >= 0).float(); y = ys.clamp(min=0); pb = (y > 0).float()
    return lam * ziln_loss_w2(logit.reshape(-1).float(), mu.reshape(-1).float(), sig.reshape(-1).float(), y.reshape(-1), pb.reshape(-1), w.reshape(-1), w.reshape(-1))


def stack_free(Xs: list) -> np.ndarray:
    """np.concatenate without the 2x peak: preallocate, copy anchor by anchor, release each source array."""
    n = sum(len(x) for x in Xs); out = np.empty((n,) + Xs[0].shape[1:], dtype=Xs[0].dtype); o = 0
    for i in range(len(Xs)):
        out[o:o + len(Xs[i])] = Xs[i]; o += len(Xs[i]); Xs[i] = None
    return out


def hist_drop_(xb, n_w: int, p: float, hist_min: int):
    """In-place augmentation: with prob p per sample keep only the last L weekly tokens (L ~ U[hist_min, n_w]),
    zeroing everything older (all channels, incl. ctx). Daily tail (positions >= n_w) untouched. The model thus learns
    from every memory length at once and still sees the full history at inference."""
    B = xb.shape[0]
    L = torch.randint(hist_min, n_w + 1, (B,), device=xb.device)
    L[torch.rand(B, device=xb.device) >= p] = n_w
    t = torch.arange(n_w, device=xb.device)[None, :]
    xb[:, :n_w] = xb[:, :n_w] * (t >= (n_w - L)[:, None]).to(xb.dtype)[:, :, None]


def token_offsets(n_w: int, tail: int, device=None):
    """Days between each token's most recent day and the anchor (hybrid layout: n_w weekly tokens oldest-first, then tail daily tokens)."""
    offs = [tail + 7 * (n_w - 1 - t) for t in range(n_w)] + [tail - 1 - j for j in range(tail)]
    return torch.tensor(offs, device=device)


def rec_drop_(xb, n_w: int, tail: int, p: float, dmax: int):
    """In-place augmentation: with prob p per sample hide the last D days (D ~ U[1, dmax]) — every token whose window ends
    inside the hidden span is zeroed (all channels). The model learns to predict the same 30d target from an input that
    stops D days earlier, i.e. without conditioning on the recent activity that the panel selection forces on the test."""
    B = xb.shape[0]
    D = torch.randint(1, dmax + 1, (B,), device=xb.device)
    D[torch.rand(B, device=xb.device) >= p] = 0
    offs = token_offsets(n_w, tail, xb.device)[None, :]
    xb *= (offs >= D[:, None]).to(xb.dtype)[:, :, None]


def drop_recent_np(X: np.ndarray, n_w: int, tail: int, D: int) -> np.ndarray:
    """Test-time view: zero every token whose most recent day is within D days of the anchor."""
    offs = token_offsets(n_w, tail).numpy()
    X = X.copy(); X[:, offs < D] = 0
    return X


def bins_setup(lt: np.ndarray, n_bins: int):
    """Bin 0 = zero target; bins 1..n_bins-1 = quantile bins of the positive log1p targets. Returns (edges, centers) as lists."""
    pos = lt[lt > 0]
    edges = np.quantile(pos, np.linspace(0, 1, n_bins)[1:-1]).tolist()
    idx = bins_index(torch.from_numpy(lt), torch.tensor(edges)).numpy()
    centers = [float(lt[idx == b].mean()) if (idx == b).any() else 0.0 for b in range(n_bins)]
    return edges, centers


def bins_index(y, edges):
    return torch.where(y > 0, 1 + torch.bucketize(y, edges.to(y.device)), torch.zeros_like(y, dtype=torch.long))


def bins_loss(logits, y, edges, w=None, smooth=0.1):
    """Cross-entropy against a smoothed one-hot (0.1 shared with the neighbouring bins)."""
    n = logits.shape[1]; idx = bins_index(y, edges)
    t = torch.zeros_like(logits).scatter_(1, idx[:, None], 1.0 - smooth)
    t.scatter_add_(1, (idx - 1).clamp(min=0)[:, None], torch.full_like(idx[:, None], smooth / 2, dtype=logits.dtype))
    t.scatter_add_(1, (idx + 1).clamp(max=n - 1)[:, None], torch.full_like(idx[:, None], smooth / 2, dtype=logits.dtype))
    ce = -(t * torch.log_softmax(logits.float(), dim=1)).sum(1)
    return (ce * w).sum() / w.sum().clamp(min=1.0) if w is not None else ce.mean()


def bins_readout(logits, centers):
    return (torch.softmax(logits.float(), dim=1) * centers.to(logits.device)[None, :]).sum(1)


def ziln_loss(logit, mu, sig, yb, pb, bce):
    loss = bce(logit, pb)
    m = pb > 0
    if SIGMA_FIXED:
        sig = torch.ones_like(sig)
    if m.any():
        nll = 0.5 * torch.log(2 * torch.pi * sig[m] ** 2) + (yb[m] - mu[m]) ** 2 / (2 * sig[m] ** 2)
        loss = loss + nll.mean()
    return loss


def ziln_loss_w2(logit, mu, sig, yb, pb, wb, wn):
    """ziln_loss with separate per-sample weights for the BCE part (wb) and the lognormal NLL part (wn).
    --pmask-from: wb=0 on anchors inside the panel's forced-return zone (P(buy|dormant) contaminated), wn kept."""
    bce = nn.functional.binary_cross_entropy_with_logits(logit, pb, reduction="none")
    loss = (wb * bce).sum() / wb.sum().clamp(min=1.0)
    m = (pb > 0) & (wn > 0)
    if SIGMA_FIXED:
        sig = torch.ones_like(sig)
    if m.any():
        nll = 0.5 * torch.log(2 * torch.pi * sig[m] ** 2) + (yb[m] - mu[m]) ** 2 / (2 * sig[m] ** 2)
        loss = loss + (wn[m] * nll).sum() / wn[m].sum()
    return loss


def ziln_loss_w(logit, mu, sig, yb, pb, w):
    """ziln_loss with per-sample weights w (0/1 masks for anchors whose horizon is not observed yet).
    Reduces to ziln_loss when w == 1."""
    bce = nn.functional.binary_cross_entropy_with_logits(logit, pb, reduction="none")
    loss = (w * bce).sum() / w.sum().clamp(min=1.0)
    m = (pb > 0) & (w > 0)
    if m.any():
        nll = 0.5 * torch.log(2 * torch.pi * sig[m] ** 2) + (yb[m] - mu[m]) ** 2 / (2 * sig[m] ** 2)
        loss = loss + (w[m] * nll).sum() / w[m].sum()
    return loss


def fresh_anchor_dates(cutoff: date, target_anchor: date, step: int, aux_h: list[int]) -> list[date]:
    """'Fresh' anchors after the 30d cutoff whose only supervision is the shortest aux horizon:
    a = cutoff+step, cutoff+2*step, ... while a + min(aux_h) <= target_anchor."""
    if not step or not aux_h:
        return []
    out = []; a = cutoff + timedelta(days=step)
    while a + timedelta(days=min(aux_h)) <= target_anchor:
        out.append(a); a += timedelta(days=step)
    return out


def main_loss(logit, mu, sig, yb, pb, bce, kind="ziln", mix_w=1.0):
    """kind: ziln (BCE + Gaussian NLL on buyers) | mse (MSE of p*mu vs log1p y — the metric itself)
    | mix (ziln + mix_w * mse)."""
    if kind == "ziln":
        return ziln_loss(logit, mu, sig, yb, pb, bce)
    mse = ((torch.sigmoid(logit) * mu - yb) ** 2).mean()
    if kind == "mse":
        return mse
    return ziln_loss(logit, mu, sig, yb, pb, bce) + mix_w * mse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="srv_seq")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--anchor-stride", type=int, default=2, help="use every Nth dense anchor for train")
    ap.add_argument("--n-anchors", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--cell", choices=["gru", "lstm", "transformer", "gru_attn", "tab", "tcn"],
                    default="gru")
    ap.add_argument("--pool", choices=["mean", "attn"], default="mean")
    ap.add_argument("--unit", choices=["week", "day"], default="week")
    ap.add_argument("--seq-len", type=int, default=0, help="0 = 52 for week / 413 for day")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast forward pass")
    ap.add_argument("--fusion-tag", default="", help="feature tag to fuse into the head (e.g. v2)")
    ap.add_argument("--channels", type=int, choices=[8, 9, 11, 13], default=8,
                    help="8 base | +visit-only days (9) | +2 weekend (11) | +2 holiday calendar (13)")
    ap.add_argument("--val-anchor", default=str(VAL_ANCHOR),
                    help="ISO date of the validation anchor (train cutoff = val - 30d)")
    ap.add_argument("--aux", default="", help="comma-separated aux horizons in days, e.g. 7,14")
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--target", choices=["gmv", "active"], default="gmv",
                    help="active: predict #active days -> logit head = P(active in 30d); saves *_pact valpred")
    ap.add_argument("--tail-days", type=int, default=0,
                    help="hybrid tokenization: append N daily tokens after the weekly ones (+flag channel)")
    ap.add_argument("--loss", choices=["ziln", "mse", "mix"], default="ziln")
    ap.add_argument("--mix-w", type=float, default=1.0, help="weight of the MSE term for --loss mix")
    ap.add_argument("--user-emb", type=int, default=0, help="dim of a learnable per-user embedding fed to the head (0=off)")
    ap.add_argument("--anchor-offset-days", type=int, default=0, help="shift the train anchor grid back by N days (anchor-diverse bagging)")
    ap.add_argument("--ctx", action="store_true", help="append platform-context channels (active users, GMV per token)")
    ap.add_argument("--ctx-set", default="v1", help="v1 = 2 channels | v2 = 8 channels (buyers, orders, AOV, conversion, detrended)")
    ap.add_argument("--warmup", type=float, default=0.0, help="linear LR warmup over this fraction of the first epoch")
    ap.add_argument("--aux-win", default="", help="aux ZILN heads on GMV sub-windows 'h:off,..' e.g. 10:10,10:20 = (a+10,a+20],(a+20,a+30]")
    ap.add_argument("--bins", type=int, default=0, help="non-parametric head: softmax over N log1p-target bins, prediction = E[log1p y]")
    ap.add_argument("--bins-w", type=float, default=1.0)
    ap.add_argument("--rec-drop", type=float, default=0.0, help="prob of hiding the last D days of input (D ~ U[1, --rec-drop-max]); see rec_drop_")
    ap.add_argument("--rec-drop-max", type=int, default=28)
    ap.add_argument("--rank-ch", default="", help="cross-sectional rank channels for these input channel indices, e.g. 0,1,3,4")
    ap.add_argument("--cart-ch", action="store_true", help="+2 channels: cart-without-order days, search-without-cart days")
    ap.add_argument("--coarse", type=int, default=0, help="merge weekly tokens older than --coarse-fine weeks in groups of N (28d tokens for N=4)")
    ap.add_argument("--coarse-fine", type=int, default=16)
    ap.add_argument("--obs-mask", action="store_true", help="append observability channel (1 = token inside observed span) + pool over observed tokens")
    ap.add_argument("--rec-mask", type=int, default=0, help="BCE of the main head off for rows whose last event is > D days before the anchor (0 = off)")
    ap.add_argument("--afe", action="store_true", help="anchor fixed effects on logit/mu during training (rank, not level)")
    ap.add_argument("--aux-lead", default="", help="comma list of offsets: aux ZILN heads on GMV over (a+off, a+off+30]; e.g. 30")
    ap.add_argument("--sigma-fixed", action="store_true", help="ZILN positive part = plain squared error (sigma=1)")
    ap.add_argument("--back-readouts", default="", help="reverse-GRU readouts at these token counts, e.g. 18,22,24,30,42,66")
    ap.add_argument("--back-hidden", type=int, default=128)
    ap.add_argument("--step-sup", type=float, default=0.0, help="step supervision weight: ZILN on every weekly token with its own 30d target (0 = off)")
    ap.add_argument("--gmv-noise", type=float, default=0.0, help="anti-fingerprint: gaussian jitter (sd) on log-gmv channel of buying tokens")
    ap.add_argument("--aux-count", default="", help="extra aux ZILN heads on 30d COUNT targets: comma list of orddays,nord")
    ap.add_argument("--pmask-from", default="", help="ISO date: anchors >= it train the mu/sigma heads only (BCE of the main head masked)")
    ap.add_argument("--short-tokens", type=int, default=0, help="multi-scale: second GRU over the last K tokens (e.g. 24 = 10 weeks + 14 days)")
    ap.add_argument("--hist-drop", type=float, default=0.0, help="history-length augmentation prob (see hist_drop_)")
    ap.add_argument("--hist-min", type=int, default=6, help="min weekly tokens kept under --hist-drop")
    ap.add_argument("--init-from", default="", help="warm start: load models/<stem>.pt (strict=False; e.g. self-supervised pretrain)")
    ap.add_argument("--horizon", type=int, default=30, help="main-target horizon in days (30 = task; 7 = pretraining on next-week GMV)")
    ap.add_argument("--min-anchor", default="2025-05-01", help="earliest train anchor (ISO); e.g. 2025-02-26 to include the gift season")
    ap.add_argument("--fctx-proxy-from", default="",
                    help="ctx-set v1f: anchors >= this ISO date use the year-ago proxy instead of the true future window "
                         "(default = val anchor, i.e. honest validation; 2099-01-01 = teacher-force everything)")
    ap.add_argument("--fctx-noise", type=float, default=0.03,
                    help="ctx-set v1f: gaussian jitter added to the 2 future-context channels per batch (anti anchor-ID memorization)")
    ap.add_argument("--fresh-step", type=int, default=0,
                    help="add 'fresh' anchors after the 30d cutoff every N days (main head masked, only aux "
                         "horizons that fit before the val anchor are supervised); 0 = off")
    ap.add_argument("--anchor-step-days", type=int, default=0,
                    help="days between train anchors (0 = 7*anchor_stride); e.g. 3 for dense anchors")
    args = ap.parse_args()
    val_anchor = date.fromisoformat(args.val_anchor)
    aux_h = [int(h) for h in args.aux.split(",") if h]
    aux_specs = [(h, "gmv", 0) for h in aux_h] + [(30, kd, 0) for kd in args.aux_count.split(",") if kd] + \
                [(30, "gmv", int(o)) for o in args.aux_lead.split(",") if o] + \
                [(int(hw.split(":")[0]), "gmv", int(hw.split(":")[1])) for hw in args.aux_win.split(",") if hw]
    pmask_from = date.fromisoformat(args.pmask_from) if args.pmask_from else None
    back_readouts = [int(k) for k in args.back_readouts.split(",") if k]
    global SIGMA_FIXED
    SIGMA_FIXED = bool(args.sigma_fixed)

    if args.cell == "tab" and not args.fusion_tag:
        raise SystemExit("--cell tab requires --fusion-tag")
    unit_days = 7 if args.unit == "week" else 1
    seq_len = args.seq_len or (WEEKS if args.unit == "week" else 413)
    if args.tail_days:
        seq_len += args.tail_days  # weekly tokens + daily tail
    ctx_arg = args.ctx_set if args.ctx else False
    if ctx_arg == "v1f":
        global FCTX_PROXY_FROM
        FCTX_PROXY_FROM = date.fromisoformat(args.fctx_proxy_from) if args.fctx_proxy_from else val_anchor
        print(f"fctx proxy from {FCTX_PROXY_FROM}", flush=True)
    rank_ch = [int(c) for c in args.rank_ch.split(",") if c]
    n_ch_model = args.channels + (1 if args.tail_days else 0) + (ctx_width(ctx_arg) if args.ctx else 0) + (1 if args.obs_mask else 0) \
        + (2 if args.cart_ch else 0) + len(rank_ch) + (1 if args.coarse else 0)
    seq_len_m = model_seq_len(seq_len, args.tail_days, args.coarse, args.coarse_fine)  # tokens seen by the model (coarse merges old weeks)
    if args.coarse and args.step_sup > 0:
        raise SystemExit("--coarse is incompatible with --step-sup")
    if args.rec_drop > 0 and (args.coarse or not args.tail_days):
        raise SystemExit("--rec-drop needs the hybrid layout (--tail-days) and no --coarse")
    n_sum = min(args.channels, 11)
    fin = dict(obs_mask=args.obs_mask, rank_ch=rank_ch, cart=args.cart_ch, coarse=args.coarse, fine=args.coarse_fine, n_sum=n_sum)
    # daily tensors are ~1.65GB/anchor even in fp16 — store fp16, cast per-batch on GPU
    store_dtype = np.float16 if args.unit == "day" else np.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    df = pl.scan_parquet(TRAIN_PARQUET).select(
        ["event_date", "user_id", "gmv", "searches", "to_cart", "to_ord", "search", "cat"]
    ).collect()
    users = df["user_id"].unique().sort().to_numpy()

    cutoff = val_anchor - timedelta(days=30)
    min_anchor = date.fromisoformat(args.min_anchor)
    if args.fusion_tag:
        # fusion needs tabular features per anchor — use the feature grid dates
        fa = sorted(date.fromisoformat(f.stem.removeprefix("anchor_"))
                    for f in (FEATURES_DIR / args.fusion_tag).glob("anchor_*.parquet"))
        train_anchors = sorted([d for d in fa if min_anchor <= d <= cutoff])[-args.n_anchors:]
    else:
        train_anchors = []
        a = cutoff - timedelta(days=args.anchor_offset_days)
        while len(train_anchors) < args.n_anchors and a >= min_anchor:
            train_anchors.append(a)
            a -= timedelta(days=args.anchor_step_days or 7 * args.anchor_stride)
        train_anchors = train_anchors[::-1]
    fresh = fresh_anchor_dates(cutoff, val_anchor, args.fresh_step, aux_h)
    if fresh and args.fusion_tag:
        raise SystemExit("--fresh-step is not supported with --fusion-tag")
    print(f"train anchors: {train_anchors}", flush=True)
    if fresh:
        print(f"fresh anchors (aux-only): {fresh}", flush=True)

    Xs, ys, Fs = [], [], []
    ys_aux = [[] for _ in aux_specs]
    w_main, w_aux, w_bce, a_idx = [], [[] for _ in aux_specs], [], []
    feat_cols = None
    for ai, a in enumerate(train_anchors + fresh):
        xa = cached_tensor(df, users, a, seq_len, unit_days, store_dtype, args.channels, args.tail_days, ctx_arg).astype(np.float16)  # fp16 in RAM (cgroup limits)
        xa = finish_tensor(xa, df, users, a, seq_len, args.tail_days, **fin)
        Xs.append(xa)
        is_fresh = a in fresh
        ys.append(np.zeros(len(users)) if is_fresh else targets_at(df, users, a, args.horizon, kind=args.target))
        w_main.append(np.full(len(users), 0.0 if is_fresh else 1.0, dtype=np.float32))
        wb = np.full(len(users), 0.0 if (is_fresh or (pmask_from and a >= pmask_from)) else 1.0, dtype=np.float32)
        if args.rec_mask:
            last = pl.DataFrame({"user_id": users}).join(df.filter(pl.col("event_date") <= a).group_by("user_id").agg(pl.col("event_date").max().alias("l")), on="user_id", how="left")["l"]
            rec = np.where(last.is_not_null().to_numpy(), (np.datetime64(a) - last.to_numpy().astype("datetime64[D]")).astype("timedelta64[D]").astype(np.int64), 10**6)
            wb = wb * (rec <= args.rec_mask).astype(np.float32)
        w_bce.append(wb); a_idx.append(np.full(len(users), ai, dtype=np.int64))
        for k, (h, kd, off) in enumerate(aux_specs):
            ok = a + timedelta(days=h + off) <= val_anchor
            ys_aux[k].append(targets_at(df, users, a, h, kind=kd, offset=off) if ok else np.zeros(len(users)))
            w_aux[k].append(np.full(len(users), 1.0 if ok else 0.0, dtype=np.float32))
        if args.fusion_tag:
            fa_arr, feat_cols = load_feats(args.fusion_tag, a, users, feat_cols)
            Fs.append(fa_arr)
        print(f"  built {a} ({time.time()-t0:.0f}s)", flush=True)
    w_main = torch.from_numpy(np.concatenate(w_main)); w_aux = [torch.from_numpy(np.concatenate(v)) for v in w_aux]; w_bce = torch.from_numpy(np.concatenate(w_bce))
    weighted = bool(fresh) or pmask_from is not None or bool(args.rec_mask)
    at_tr = torch.from_numpy(np.concatenate(a_idx)); n_anch_total = int(at_tr.max()) + 1
    yt_step = None
    if args.step_sup > 0:
        if args.fusion_tag or args.user_emb or args.short_tokens or args.cell not in ("gru", "lstm"):
            raise SystemExit("--step-sup needs a plain gru/lstm trunk (head_in = 2*hidden)")
        n_w_steps = seq_len - args.tail_days; Cd = daily_cumsum(df, users)
        yt_step = torch.from_numpy(np.concatenate([step_targets(Cd, a, n_w_steps, args.tail_days, args.horizon) for a in train_anchors + fresh]))
        del Cd; print(f"step-sup: targets {tuple(yt_step.shape)}, covered {(yt_step >= 0).float().mean():.3f}", flush=True)
    X_va = cached_tensor(df, users, val_anchor, seq_len, unit_days, store_dtype, args.channels, args.tail_days, ctx_arg)
    X_va = finish_tensor(X_va, df, users, val_anchor, seq_len, args.tail_days, **fin)
    assert X_va.shape[1] == seq_len_m and X_va.shape[2] == n_ch_model, (X_va.shape, seq_len_m, n_ch_model)
    y_va_raw = np.clip(targets_at(df, users, val_anchor, kind=args.target), 0, None)
    X_tr = stack_free(Xs); y_tr_raw = np.clip(np.concatenate(ys), 0, None)
    y_aux_raw = [np.clip(np.concatenate(v), 0, None) for v in ys_aux]
    del Xs, ys, ys_aux
    F_tr = F_va = None
    feat_dim = 0
    if args.fusion_tag:
        F_va, feat_cols = load_feats(args.fusion_tag, val_anchor, users, feat_cols)
        F_tr = np.concatenate(Fs)
        del Fs
        mu = np.nanmean(F_tr, axis=0)
        sd = np.nanstd(F_tr, axis=0) + 1e-6
        F_tr = np.nan_to_num((F_tr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        F_va = np.nan_to_num((F_va - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        feat_dim = F_tr.shape[1]
        np.savez(MODELS_DIR / f"{args.name}_fnorm.npz", mu=mu, sd=sd)  # needed at test time
        print(f"fusion features: {feat_dim} cols", flush=True)
    del df
    lt_tr = np.log1p(y_tr_raw).astype(np.float32)
    print(f"train tensor {X_tr.shape}, val {X_va.shape}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)

    n_users = len(users)
    model = SeqZiln(args.hidden, args.layers, args.cell, seq_len_m, feat_dim, args.pool,
                    n_ch_model, n_aux=len(aux_specs), n_users=n_users, user_emb=args.user_emb, short_tokens=args.short_tokens,
                    obs_pool=args.obs_mask, back_readouts=back_readouts, back_hidden=args.back_hidden, n_bins=args.bins).to(dev)
    bins_edges = bins_centers = None
    if args.bins:
        e_, c_ = bins_setup(lt_tr, args.bins); bins_edges, bins_centers = torch.tensor(e_, device=dev), torch.tensor(c_, device=dev)
        print(f"bins: {args.bins} (edges {np.round(e_[:3], 2)}..{np.round(e_[-2:], 2)})", flush=True)
    fe = torch.zeros(n_anch_total, 2, device=dev, requires_grad=True) if args.afe else None
    if args.init_from:
        sd = torch.load(MODELS_DIR / f"{args.init_from}.pt", map_location=dev)
        own = model.state_dict(); ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(ok, strict=False); print(f"init-from {args.init_from}: loaded {len(ok)}/{len(own)} tensors", flush=True)
    uid_all = torch.arange(n_users)
    opt = torch.optim.AdamW([{"params": model.parameters()}] + ([{"params": [fe], "weight_decay": 0.0}] if fe is not None else []),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_tr); yt = torch.from_numpy(lt_tr)
    if str(dev) == "cuda" and not os.environ.get("SEQ_CPU_TENSOR"):
        nbytes = Xt.numel() * Xt.element_size(); free = torch.cuda.mem_get_info()[0]
        if nbytes + 12e9 < free:  # keep the whole fp16 train tensor on the GPU (batch gather on GPU); leave >=12 GB for model/eval workspace
            Xt = Xt.to(dev); del X_tr; print(f"train tensor on GPU ({nbytes/1e9:.1f} GB, free was {free/1e9:.1f} GB)", flush=True)
        else:
            print(f"train tensor stays on CPU ({nbytes/1e9:.1f} GB vs free {free/1e9:.1f} GB)", flush=True)
    pt = torch.from_numpy((y_tr_raw > 0).astype(np.float32))
    yt_aux = [torch.from_numpy(np.log1p(v).astype(np.float32)) for v in y_aux_raw]
    pt_aux = [torch.from_numpy((v > 0).astype(np.float32)) for v in y_aux_raw]
    Xv = torch.from_numpy(X_va)
    Ft = torch.from_numpy(F_tr) if feat_dim else None
    Fv = torch.from_numpy(F_va) if feat_dim else None
    n = len(Xt)
    best = (1e9, None)
    epoch_ranks = []; best_rank = float('inf'); best_epoch = 0; epoch_lps = []
    steps_per_epoch = (n + args.batch - 1) // args.batch
    warm_steps = int(args.warmup * steps_per_epoch)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for bi, i in enumerate(range(0, n, args.batch)):
            if ep == 0 and warm_steps and bi < warm_steps:
                for g in opt.param_groups:
                    g["lr"] = args.lr * (bi + 1) / warm_steps
            idx = perm[i:i + args.batch]
            xb, yb, pb = Xt[idx.to(Xt.device)].to(dev).float(), yt[idx].to(dev), pt[idx].to(dev)
            if ctx_arg == "v1f" and args.fctx_noise > 0:
                xb[:, :, -2:] += args.fctx_noise * torch.randn(len(idx), 1, 2, device=dev)
            if args.hist_drop > 0:
                hist_drop_(xb, seq_len_m - args.tail_days, args.hist_drop, args.hist_min)
            if args.rec_drop > 0:
                rec_drop_(xb, seq_len_m - args.tail_days, args.tail_days, args.rec_drop, args.rec_drop_max)
            if args.gmv_noise > 0:
                xb[:, :, 0] += args.gmv_noise * torch.randn_like(xb[:, :, 0]) * (xb[:, :, 0] > 0).float()
            fb = Ft[idx].to(dev) if feat_dim else None
            ub = (idx % n_users).to(dev) if args.user_emb else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                (logit, mu, sig), aux = model(xb, fb, return_aux=True, uid=ub)
                if fe is not None:  # anchor fixed effects: absorb per-anchor level/selection regime (train only)
                    ab = at_tr[idx].to(dev); logit = logit + fe[ab, 0]; mu = mu + fe[ab, 1]
                if weighted:
                    loss = ziln_loss_w2(logit.float(), mu.float(), sig.float(), yb, pb, w_bce[idx].to(dev), w_main[idx].to(dev))
                    for k, (al, am, asg) in enumerate(aux):
                        loss = loss + args.aux_weight * ziln_loss_w(
                            al.float(), am.float(), asg.float(),
                            yt_aux[k][idx].to(dev), pt_aux[k][idx].to(dev), w_aux[k][idx].to(dev))
                else:
                    loss = main_loss(logit.float(), mu.float(), sig.float(), yb, pb, bce, args.loss, args.mix_w)
                    for k, (al, am, asg) in enumerate(aux):
                        loss = loss + args.aux_weight * ziln_loss(
                            al.float(), am.float(), asg.float(),
                            yt_aux[k][idx].to(dev), pt_aux[k][idx].to(dev), bce)
                if yt_step is not None:
                    loss = loss + step_loss(model.step_outputs(n_w_steps), yt_step[idx].to(dev).float(), args.step_sup)
                if args.bins:
                    loss = loss + args.bins_w * bins_loss(model._bins_logits, yb, bins_edges, w_main[idx].to(dev) if weighted else None)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        # cudnn RNN eval workspace ~ batch*seq_len*hidden*layers — long sequences need small chunks
        eval_bs = args.batch if (args.unit == "day" or Xt.is_cuda) else 32768  # smaller cudnn eval workspace when the train tensor lives on the GPU
        with torch.no_grad():
            while True:  # OOM-safe validation: halve the chunk and retry
                try:
                    lps = []
                    for i in range(0, len(Xv), eval_bs):
                        fvb = Fv[i:i + eval_bs].to(dev) if feat_dim else None
                        uvb = uid_all[i:i + eval_bs].to(dev) if args.user_emb else None
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                            logit, mu, sig = model(Xv[i:i + eval_bs].to(dev).float(), fvb, uid=uvb)
                        lps.append((bins_readout(model._bins_logits, bins_centers) if args.bins else torch.sigmoid(logit.float()) * mu.float()).cpu().numpy())
                        if args.target == "active":
                            pacts = pacts + [torch.sigmoid(logit.float()).cpu().numpy()] if i else [torch.sigmoid(logit.float()).cpu().numpy()]
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); eval_bs = max(eval_bs // 2, 512); print(f"eval OOM -> eval_bs {eval_bs}", flush=True)
            lp = np.clip(np.concatenate(lps), 0, None)
        score = rmsle(y_va_raw, np.expm1(lp))
        # rank-score (opt additive shift in log space) per epoch: raw RMSLE mostly tracks level noise
        _ly = np.log1p(y_va_raw)
        rank_ep = min(float(np.sqrt(np.mean((_ly - np.clip(lp + d, 0, None)) ** 2))) for d in np.linspace(-0.6, 0.4, 51))
        epoch_ranks.append(rank_ep); epoch_lps.append(lp.copy())
        if args.target == "active":
            from sklearn.metrics import roc_auc_score
            pact = np.concatenate(pacts); auc = roc_auc_score(y_va_raw > 0, pact)
            print(f"epoch {ep+1}: AUC(P_active) = {auc:.5f}", flush=True)
            if score < best[0]:
                pl.DataFrame({"user_id": users, "p_act": pact}).write_parquet(MODELS_DIR / f"{args.name}_pact_{val_anchor}.parquet")
        print(f"epoch {ep+1}: val RMSLE = {score:.5f}  rank = {rank_ep:.5f}", flush=True)
        # select the epoch by rank-score (level is calibrated separately on the LB anyway)
        if rank_ep < best_rank:
            best_rank = rank_ep; best_epoch = ep + 1
            best = (score, lp.copy())
            torch.save(model.state_dict(), MODELS_DIR / f"{args.name}.pt")

    if best[1] is None:
        print(f"[SEQ-{args.cell.upper()}] {args.name}: ALL EPOCHS NaN — aborting", flush=True)
        return
    ly = np.log1p(y_va_raw)
    shifts = [float(np.sqrt(np.mean((ly - np.clip(best[1] + d, 0, None)) ** 2)))
              for d in np.linspace(-0.4, 0.2, 61)]
    # epoch-averaging diagnostics: is the last epoch / an average of late epochs as good as the best epoch?
    def _rank(v): return min(float(np.sqrt(np.mean((ly - np.clip(v + d, 0, None)) ** 2))) for d in np.linspace(-0.6, 0.4, 51))
    n_ep = len(epoch_lps); avg_ranks = {}
    if n_ep >= 2:
        for k in (2, 3, 4):
            if n_ep >= k: avg_ranks[f"avg_last{k}"] = _rank(np.mean(epoch_lps[-k:], axis=0))
        avg_ranks["avg_ep2plus"] = _rank(np.mean(epoch_lps[1:], axis=0))
        avg_ranks["last"] = epoch_ranks[-1]
        print("epoch ranks: " + " ".join(f"{r:.5f}" for r in epoch_ranks) + f" | best ep {best_epoch} | "
              + " ".join(f"{k}={v:.5f}" for k, v in avg_ranks.items()), flush=True)
    print(f"\n[SEQ-{args.cell.upper()}] {args.name}: best val RMSLE = {best[0]:.5f}  "
          f"rank-score(opt-shift) = {min(shifts):.5f}", flush=True)
    pl.DataFrame({"user_id": users, "pred": np.expm1(best[1]), "target": y_va_raw}).write_parquet(
        MODELS_DIR / f"{args.name}_valpred.parquet"
    )
    (MODELS_DIR / f"{args.name}.meta.json").write_text(
        json.dumps(dict(name=args.name, val_rmsle=best[0], rank_score=min(shifts),
                        cell=args.cell, unit=args.unit, seq_len=seq_len, ch=args.channels,
                        hidden=args.hidden, layers=args.layers, n_anchors=args.n_anchors,
                        anchor_stride=args.anchor_stride, lr=args.lr, seed=args.seed,
                        pool=args.pool, fusion_tag=args.fusion_tag, feat_cols=feat_cols,
                        val_anchor=str(val_anchor), aux_horizons=aux_h,
                        anchor_step_days=args.anchor_step_days, target=args.target,
                        aux_weight=args.aux_weight, tail_days=args.tail_days, fresh_step=args.fresh_step, fctx_noise=args.fctx_noise, init_from=args.init_from, horizon=args.horizon, short_tokens=args.short_tokens, hist_drop=args.hist_drop, hist_min=args.hist_min, gmv_noise=args.gmv_noise, aux_count=args.aux_count, pmask_from=args.pmask_from, step_sup=args.step_sup,
                        obs_mask=args.obs_mask, rec_mask=args.rec_mask, afe=args.afe, aux_lead=args.aux_lead, sigma_fixed=args.sigma_fixed,
                        back_readouts=args.back_readouts, back_hidden=args.back_hidden,
                        rank_ch=args.rank_ch, cart_ch=args.cart_ch, coarse=args.coarse, coarse_fine=args.coarse_fine,
                        rec_drop=args.rec_drop, rec_drop_max=args.rec_drop_max, aux_win=args.aux_win,
                        bins=args.bins, bins_w=args.bins_w, bins_edges=(bins_edges.tolist() if args.bins else None), bins_centers=(bins_centers.tolist() if args.bins else None),
                        loss=args.loss, mix_w=args.mix_w, user_emb=args.user_emb, ctx=bool(args.ctx), ctx_set=args.ctx_set if args.ctx else None,
                        anchor_offset_days=args.anchor_offset_days, warmup=args.warmup,
                        epoch_ranks=epoch_ranks, best_epoch=best_epoch, avg_ranks=avg_ranks), indent=2)
    )


if __name__ == "__main__":
    main()

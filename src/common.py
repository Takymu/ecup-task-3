"""Shared paths, metric, and anchor-date logic for the LTV pipeline."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TASK_DIR / "data"
TRAIN_PARQUET = DATA_DIR / "train.parquet"
SAMPLE_SUBMIT = DATA_DIR / "sample_submit.csv"

ARTIFACTS_DIR = TASK_DIR / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"

for d in (FEATURES_DIR, MODELS_DIR, SUBMISSIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATA_START = date(2025, 1, 1)
DATA_END = date(2026, 2, 13)  # last day of history; predict [2026-02-14, 2026-03-15]
HORIZON_DAYS = 30

# anchor whose future window is fully inside train data and closest to test
VAL_ANCHOR = DATA_END - timedelta(days=HORIZON_DAYS)  # 2026-01-14
TEST_ANCHOR = DATA_END  # 2026-02-13


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def train_anchors(n: int, stride_days: int = 14, last: date = VAL_ANCHOR) -> list[date]:
    """n anchors ending at `last`, spaced `stride_days` apart, oldest first."""
    return [last - timedelta(days=stride_days * i) for i in range(n)][::-1]


def lean_load(
    tag: str,
    cutoff: date | None = None,
    min_anchor: date | None = None,
    val_anchor: date | None = None,
    drop_cols: tuple[str, ...] = ("user_id", "anchor_date", "target", "anchor_month", "anchor_doy"),
    extras_tag: str | None = None,
):
    """Memory-lean anchor loading: per-file float32 accumulation, no full-frame copies.

    Returns dict with X_tr, y_tr_raw (clipped), age_days_tr, X_va, y_va_raw, va_users, feat_cols.
    Train rows: anchor <= cutoff (if set), anchor >= min_anchor (if set), user existed at anchor.
    """
    import numpy as _np
    import polars as _pl

    files = sorted((FEATURES_DIR / tag).glob("anchor_*.parquet"))
    feat_cols = None
    Xs, ys, ages = [], [], []
    X_va = y_va_raw = va_users = None
    ref_anchor = val_anchor or VAL_ANCHOR

    for f in files:
        a = date.fromisoformat(f.stem.removeprefix("anchor_"))
        is_val = val_anchor is not None and a == val_anchor
        is_train = (cutoff is None or a <= cutoff) and (min_anchor is None or a >= min_anchor)
        if not (is_val or is_train):
            continue
        part = _pl.read_parquet(f)
        if extras_tag is not None:
            ef = FEATURES_DIR / extras_tag / f.name
            if ef.exists():
                part = part.join(_pl.read_parquet(ef), on="user_id", how="left")
        if not is_val:
            part = part.filter(_pl.col("days_active_total").is_not_null())
            if part["target"].null_count() > 0:  # test anchor file — never train on it
                continue
        if feat_cols is None:
            feat_cols = [c for c in part.columns if c not in drop_cols]
        X = part.select(feat_cols).to_numpy().astype(_np.float32)
        y = _np.clip(part["target"].to_numpy(), 0, None).astype(_np.float64)
        if is_val:
            X_va, y_va_raw, va_users = X, y, part["user_id"].to_numpy()
        if is_train and not is_val:
            Xs.append(X)
            ys.append(y)
            ages.append(_np.full(len(y), (ref_anchor - a).days, dtype=_np.float32))
        del part

    X_tr = _np.vstack(Xs) if Xs else None
    del Xs
    return dict(
        X_tr=X_tr,
        y_tr_raw=_np.concatenate(ys) if ys else None,
        age_days_tr=_np.concatenate(ages) if ages else None,
        X_va=X_va, y_va_raw=y_va_raw, va_users=va_users, feat_cols=feat_cols,
    )

"""sub_v15_recshift_L233 — сегментная поправка по давности активности поверх sub_v15_famblend_L233.

Механизм (записан в рабочем журнале при постройке): тестовая выборка отобрана по активности
в окне отбора, поэтому «дремлющие» пользователи в обучении всегда возвращаются, а в тесте — нет,
и модели завышают их GMV. Поправка — сдвиг log1p-предсказания по корзинам паузы активности
(0 / 1–3 / 4–7 / 8–14 / 15+ дней) на [-.017, -.037, -.064, -.09, -.12], центрированный,
чтобы средний уровень сабмита не менялся.

Исходный скрипт постройки был одноразовым и не сохранился; этот файл восстанавливает правило
по записи в журнале и воспроизводит исторический файл с точностью до машинного нуля
(max |Δlog1p| ~ 8e-16 при сверке с отправленным).
"""
import numpy as np, polars as pl
from pathlib import Path

T = Path(__file__).resolve().parents[2]
S = T / "artifacts/submissions"

v15 = pl.read_csv(S / "sub_v15_famblend_L233.csv").sort("user_id")
lp = np.log1p(v15["predict"].to_numpy())
f = pl.read_parquet(T / "artifacts/features/v2/anchor_2026-02-13.parquet").sort("user_id")
assert (f["user_id"].to_numpy() == v15["user_id"].to_numpy()).all()
r = f["days_since_last_event"].to_numpy()

SHIFT = np.array([-.017, -.037, -.064, -.09, -.12])
bucket = np.select([r < 1, r <= 3, r <= 7, r <= 14], [0, 1, 2, 3], default=4)
d = SHIFT[bucket]
d = d - d.mean()                      # средний уровень сохраняется

pred = np.clip(np.expm1(np.clip(lp + d, 0, None)), 0, None)
out = v15.select("user_id").with_columns(pl.Series("predict", pred))
out.write_csv(S / "sub_v15_recshift_L233.csv")
print(f"saved sub_v15_recshift_L233.csv  mean_lp={np.log1p(pred).mean():.4f}")

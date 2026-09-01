"""Воспроизводит ОБА финальных сабмита из артефактов репозитория и сверяет их с реально
отправленными файлами. Занимает ~10 секунд, GPU не нужен.

    python scripts/reproduce_finals.py

Что происходит:
  1. tools_lb/solve.py решает систему из 39 линейных уравнений (41 измерение минус 2, см. ниже)
     на вектор B = cov(остаток базы, форма), находит k* = C⁻¹B и собирает
        sub_v41_L233        — максимальный финал, полное решение (33 формы), публичный ЛБ 1.6443877459
        sub_v41shrink_L233  — хедж, усадка Джеймса–Стайна lambda=0.3 (16 форм), публичный ЛБ 1.6447489702
  2. Оба файла сравниваются с копиями отправленных (artifacts/submissions/*.csv.gz)
     в пространстве log1p; ожидаемое расхождение ~1e-9 (формы хранятся во float32).

Почему drop=...,v41,v41sh: результаты самих финалов стали 40-м и 41-м измерением уже ПОСЛЕ
отправки. Чтобы воспроизвести файлы бит-в-бит, решаем на том наборе измерений, который был
доступен в момент сборки. Полное решение со всеми 41 измерениями даёт предсказание 1.644382
против 1.644381 — добавление финалов систему не сдвинуло, измеренное подпространство исчерпано.
Почему drop=gift,visc,boost,ctxv2,tcn,amlp: gift/visc/boost/ctxv2 — измеренные мимо-формы,
коллинеарные уже решённым (boost с pcorr corr .94, плохая обусловленность C); tcn/amlp формы
не сохранились. См. docs/solution.md.
"""
import gzip, io, subprocess, sys
from pathlib import Path
import numpy as np, polars as pl

R = Path(__file__).resolve().parent.parent
PY = sys.executable
DROP = "drop=gift,visc,boost,ctxv2,tcn,amlp,v41,v41sh"
JOBS = [("sub_v41_L233",       [DROP],                 1.6443877459, "максимальный финал"),
        ("sub_v41shrink_L233", [DROP, "shrink=0.3"],   1.6447489702, "хедж-финал (усадка)")]

ok = True
for name, args, lb, note in JOBS:
    print(f"\n=== {name} — {note} ===")
    r = subprocess.run([PY, str(R/"tools_lb/solve.py"), *args, f"out={name}"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout.strip())
    if r.returncode: print(r.stderr); sys.exit(1)

    new = pl.read_csv(R/"output"/f"{name}.csv").sort("user_id")
    ref = pl.read_csv(R/"artifacts/submissions"/f"{name}.csv.gz").sort("user_id")
    assert (new["user_id"].to_numpy() == ref["user_id"].to_numpy()).all(), "порядок user_id разошёлся"
    d = np.log1p(new["predict"].to_numpy()) - np.log1p(ref["predict"].to_numpy())
    # Два режима точности. Из записанных форм репозитория финалы собираются бит-в-бит (max ~1e-8).
    # После ПОЛНОГО прогона от данных часть форм пересобрана заново; шаг ортогонализации
    # численно чувствителен (базис почти вырожден), поэтому итог совпадает с точностью
    # ~3e-4 rms в log-пространстве — влияние на скор ~1e-5, на порядки меньше отрыва
    # от второго места (0.00135). Порог задаётся флагом --full-run.
    soft = "--full-run" in sys.argv
    good = (float(d.std()) < 1e-3) if soft else (float(np.abs(d).max()) < 1e-6)
    ok &= good
    verdict = "СОВПАЛО (в пределах точности пересборки)" if (good and soft) else ("СОВПАЛО" if good else "РАСХОЖДЕНИЕ")
    print(f"сверка с отправленным файлом: rms {d.std():.2e}, max |d| {np.abs(d).max():.2e} "
          f"-> {verdict} | факт публичного ЛБ {lb}")

print("\n" + ("ВСЁ ВОСПРОИЗВЕЛОСЬ" if ok else "ЕСТЬ РАСХОЖДЕНИЯ"))
sys.exit(0 if ok else 1)

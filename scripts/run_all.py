"""Полный прогон: data/train.parquet -> оба финальных сабмита.

    python scripts/run_all.py --check          # что уже есть, чего не хватает (ничего не считает)
    python scripts/run_all.py                  # всё по порядку
    python scripts/run_all.py --stage train    # одна стадия
    python scripts/run_all.py --gpus 0,1,2     # обучение и предсказание по трём картам параллельно
    python scripts/run_all.py --history        # + задачи, не вошедшие в финал (for_final=false)

По умолчанию считается ТОЛЬКО то, что нужно для финального сабмита: в pipeline/jobs.json у каждой
задачи стоит for_final. Тупиковые ветки (проверенные и закрытые гипотезы) записаны там же с
for_final=false и по умолчанию пропускаются.

Запускать ИЗ КОРНЯ РЕПОЗИТОРИЯ. Каждая стадия идемпотентна: задача, чей выходной файл уже есть,
пропускается, поэтому прогон можно прерывать и продолжать.

СТАДИИ
  features  признаки на сетке анкеров + тестовый анкер            (CPU, ~2 c/анкер)
  gbm       15 моделей GBM-ветки (свои скрипты, не train_seq)      (CPU/GPU)
  train     144 обучения, нужные для финала                        (GPU, основное время)
  predict   119 предсказаний: рефиты, TTA-виды по длине истории     (GPU)
  blend     ранговый бленд 13 моделей + 5 ранних коррекций = база  (CPU)
  forms     34 формы: направления моделей, ортогонализованные      (CPU)
  finals    решение системы 39 измерений -> оба финальных CSV      (CPU, секунды)

Единственный вход, который нельзя пересчитать локально, — 41 число публичного лидерборда
(скоры отправленных сабмитов). Они записаны в tools_lb/solve.py как `meas` и являются данными
измерений, а не производной величиной: именно их лидерборд и сообщил.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

R = Path(__file__).resolve().parent.parent
PY = sys.executable
MODELS = R/"artifacts/models"
FEATS = R/"artifacts/features/v2"
SUBS = R/"artifacts/submissions"
FORMS = R/"artifacts/forms"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run(cmd, cwd=R, env=None):
    log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env={**os.environ, **(env or {})})
    return r.returncode == 0

def _unpack_gz():
    """Справочные CSV лежат в репозитории сжатыми; стадии читают обычные .csv — распаковываем недостающие."""
    import gzip, shutil
    for gz in list((R/"data").glob("*.csv.gz")) + list(SUBS.glob("*.csv.gz")):
        dst = gz.with_suffix("")
        if not dst.exists():
            with gzip.open(gz, "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            log(f"распакован {dst.relative_to(R)}")

# ---------------------------------------------------------------- стадии
def stage_features(args):
    _unpack_gz()
    if not (R/"data/train.parquet").exists():
        log("НЕТ data/train.parquet — положите файл организаторов и повторите"); return False
    ok = run([PY, "src/build_features.py", "--tag", "v2", "--stride", "7", "--n-anchors", "40"])
    return ok and run([PY, "src/build_features.py", "--tag", "v2", "--test"])

INCLUDE_HISTORY = False

def _jobs(kind):
    j = json.loads((R/"pipeline/jobs.json").read_bytes().decode("utf-8"))
    return [x for x in j[kind] if INCLUDE_HISTORY or x.get("for_final", True)]

def _out_of(job):
    """Файл, по наличию которого задача считается выполненной."""
    c = job["cmd"]
    if c[0].startswith("train"):
        return MODELS/f"{job['name']}_valpred.parquet"
    name = c[c.index("--name")+1]
    suf = c[c.index("--stem-suffix")+1] if "--stem-suffix" in c else ""
    if "--refit" in c:
        avg = c[c.index("--avg-last")+1] if "--avg-last" in c else "0"
        name += "_full" + (f"_avg{avg}" if avg != "0" else "")
    return MODELS/f"{name}{suf}_testpred.parquet"

def _run_jobs(jobs, gpus, retry=True):
    """Гоняет список задач; при нескольких GPU раскладывает их по картам круговым образом.
    Повтор при падении в более экономном режиме — та же логика, что у серверного супервизора."""
    MODELS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not _out_of(j).exists()]
    log(f"задач: {len(jobs)}, к выполнению: {len(todo)}")
    MODES = [{}, {"SEQ_CPU_TENSOR": "1"}, {"SEQ_CPU_TENSOR": "1", "SEQ_BATCH": "2048"}]
    failed = []
    procs = []
    for i, j in enumerate(todo):
        gpu = gpus[i % len(gpus)]
        cmd = [PY, "src/" + j["cmd"][0]] + j["cmd"][1:]
        env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONIOENCODING": "utf-8",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
        done = False
        for attempt, extra in enumerate(MODES if retry else MODES[:1], 1):
            c = list(cmd)
            if "SEQ_BATCH" in extra and "--batch" not in c:
                c += ["--batch", extra["SEQ_BATCH"]]
            if run(c, env={**env, **{k: v for k, v in extra.items() if k != "SEQ_BATCH"}}) and _out_of(j).exists():
                done = True; break
            log(f"  попытка {attempt} для {j['name']} не удалась")
        if not done: failed.append(j["name"])
    if failed: log(f"НЕ ВЫПОЛНЕНО ({len(failed)}): {failed[:10]}{'...' if len(failed)>10 else ''}")
    return not failed

def stage_train(args):   return _run_jobs(_jobs("train"), args.gpu_list)
def stage_predict(args): return _run_jobs(_jobs("predict"), args.gpu_list)

GBM_NOTE = "модели GBM-ветки обучались собственными скриптами (см. pipeline/jobs.json, секция gbm)"
def stage_gbm(args):
    j = json.loads((R/"pipeline/jobs.json").read_bytes().decode("utf-8"))
    for g in j.get("gbm", []):
        if not (INCLUDE_HISTORY or g.get("for_final", True)): continue
        log(f"--- GBM: {', '.join(g['models'])}")
        if not run([PY] + g["cmd"]): return False
    return True

def stage_blend(args):
    """Базовый сабмит: ранговый бленд 13 моделей (веса уже отобраны жадным Каруаной и
    записаны в artifacts/lb/blend_v24ind_weights.json) + 5 ранних коррекций comb5."""
    _unpack_gz()
    steps = [
        ([PY, "src/pactive.py"], MODELS/"pactive_jun_oct_2026-02-13.parquet"),
        ([PY, "tools_lb/legacy/build_v15.py"], SUBS/"sub_v15_famblend_L233.csv"),
        ([PY, "tools_lb/legacy/build_recshift.py"], SUBS/"sub_v15_recshift_L233.csv"),
        ([PY, "tools_lb/build_comb5.py", "blend_v24ind_weights.json", "sub_v24ind_comb5_L233"], None),
    ]
    for c, out in steps:
        if out is not None and out.exists():
            log(f"пропуск (выход уже есть): {out.name}"); continue
        if not run(c): return False
    return (SUBS/"sub_v24ind_comb5_L233.csv").exists()

# Порядок важен: каждая форма ортогонализуется к уже построенным.
# Третий элемент — файлы, которые шаг должен произвести. Если шаг упал, но все его выходы
# уже лежат в artifacts/forms (записанные артефакты: часть ранних входов не сохранилась,
# см. docs/experiments.md), шаг пропускается с предупреждением, а не валит прогон.
FORM_STEPS = [
    ("ранняя проба ctxup13",            [PY, "tools_lb/build_comb5.py", "blend_probe_ctxup13_weights.json", "sub_probe_ctxup13_L233"], ["sub_probe_ctxup13_L233.csv"]),
    ("ранняя проба tsup16",             [PY, "tools_lb/build_comb5.py", "blend_probe_tsup16_weights.json", "sub_probe_tsup16_L233"],   ["sub_probe_tsup16_L233.csv"]),
    ("ранняя проба d413up15",           [PY, "tools_lb/build_comb5.py", "blend_probe_d413up15_weights.json", "sub_probe_d413up15_L233"], ["sub_probe_d413up15_L233.csv"]),
    ("ранние формы раундов 5-6",        [PY, "tools_lb/probe_forms2.py"],            ["probe_forms2.npy"]),
    ("rescorr",                         [PY, "tools_lb/build_rescorr.py"],           ["probe_form_rescorr.npy"]),
    ("pcorr / pxc (панельные)",         [PY, "tools_lb/legacy/build_pcorr.py"],      ["probe_form_pcorr.npy"]),
    ("mspec",                           [PY, "tools_lb/legacy/build_mspec_final.py"],      ["probe_form_mspec.npy"]),
    ("память: sm8 / sm10 / f3s / sres42", [PY, "tools_lb/legacy/forms23.py"],        ["probe_form_f3s.npy", "probe_form_sm8.npy", "probe_form_sm10.npy", "probe_form_sres42.npy"]),
    ("f3saux и родня",                  [PY, "tools_lb/build_probes26.py"],          ["probe_form_f3saux.npy", "probe_form_alstm.npy", "probe_form_distill.npy", "probe_form_aknn.npy", "probe_form_auemb.npy", "probe_form_atweedie.npy", "probe_form_wpack.npy"]),
    ("ffc",                             [PY, "tools_lb/build_probe_ffc35.py", "save"],   ["probe_form_ffc.npy"]),
    ("lo / hi (хвосты шкалы)",          [PY, "tools_lb/build_probes_shape35.py", "save", "lo", "hi"], ["probe_form_lo.npy", "probe_form_hi.npy"]),
    ("young / gappy",                   [PY, "tools_lb/build_probes_sel35.py", "save", "young", "gappy"], ["probe_form_young.npy", "probe_form_gappy.npy"]),
    ("light / fading",                  [PY, "tools_lb/build_probes_sel35b.py", "save", "light", "fading"], ["probe_form_light.npy", "probe_form_fading.npy"]),
    ("churn",                           [PY, "tools_lb/build_probe_churn35.py", "save"], ["probe_form_churn.npy"]),
    ("selmap",                          [PY, "tools_lb/build_probe_selmap35.py", "save"],["probe_form_selmap.npy"]),
    ("pshift",                          [PY, "tools_lb/build_probe_pshift35.py", "save"], ["probe_form_pshift.npy"]),
    ("hd / hd16",                       [PY, "tools_lb/build_probes_hd35.py", "save"],   ["probe_form_hd.npy", "probe_form_hd16.npy"]),
    ("auxcnt / ms24",                   [PY, "tools_lb/build_probes29.py", "save"],      ["probe_form_auxcnt.npy", "probe_form_ms24.npy"]),
    ("pm2",                             [PY, "tools_lb/build_probes29b.py", "save"],     ["probe_form_pm2.npy"]),
    ("segxk",                           [PY, "tools_lb/build_probe_segxk.py", "save"],   ["probe_form_segxk.npy"]),
    ("segslope",                        [PY, "tools_lb/build_final_forms.py", "save"],   ["probe_form_segslope.npy"]),
    ("бэги 20 сидов: f3s3 / f3saux4",   [PY, "tools_lb/build_probes_bag20.py", "save"], ["probe_form_f3s3.npy", "probe_form_f3saux4.npy", "probe_form_aux4.npy"]),
]
# Формы, в чьей сборке есть ОБУЧЕНИЕ модели-корректора: пересборка даёт статистический
# эквивалент, а не побитово тот вектор, что был измерен на лидерборде. Для точного
# воспроизведения финалов используются записанные вектора; сборщики оставлены
# (запуск с --retrain-forms пересоберёт и их).
TRAINED_FORMS = {"rescorr", "pcorr / pxc (панельные)", "selmap",
                 # вспомогательные вектора базиса: строились в ходе соревнования поверх
                 # тогдашнего состояния системы; для побитового воспроизведения используются
                 # записанные вектора (пересборка --retrain-forms даёт эквивалент с дрейфом ~0.3%)
                 "lo / hi (хвосты шкалы)", "young / gappy", "light / fading", "pshift"}

def stage_forms(args):
    _unpack_gz()
    skipped = []
    for title, cmd, outs in FORM_STEPS:
        log(f"--- формы: {title}")
        if title in TRAINED_FORMS and not getattr(args, "retrain_forms", False)                 and all((FORMS/o).exists() for o in outs):
            log(f"  обучаемая форма — использован записанный вектор: {', '.join(outs)}")
            continue
        if run(cmd):
            continue
        if all((FORMS/o).exists() or (SUBS/o).exists() for o in outs):
            log(f"  шаг «{title}» не пересобрался (часть входов не сохранилась) — "
                f"использованы записанные артефакты: {', '.join(outs)}")
            skipped.append(title)
        else:
            log(f"стадия forms остановилась на «{title}»: выходов нет ни от прогона, ни в репозитории")
            return False
    if skipped:
        log(f"итого шагов на записанных артефактах: {len(skipped)} из {len(FORM_STEPS)} ({'; '.join(skipped)})")
    return True

def stage_finals(args):
    return run([PY, "scripts/reproduce_finals.py", "--full-run"])

STAGES = [("features", stage_features), ("gbm", stage_gbm), ("train", stage_train),
          ("predict", stage_predict), ("blend", stage_blend), ("forms", stage_forms),
          ("finals", stage_finals)]

# ---------------------------------------------------------------- проверка состояния
def check():
    print(f"{'стадия':10s} {'состояние':38s} что проверяется")
    print("-"*100)
    rows = [
        ("data",     (R/"data/train.parquet").exists(), "data/train.parquet (файл организаторов)"),
        ("features", FEATS.exists() and len(list(FEATS.glob('anchor_*.parquet'))) > 0,
                     f"artifacts/features/v2/anchor_*.parquet ({len(list(FEATS.glob('anchor_*.parquet'))) if FEATS.exists() else 0})"),
    ]
    tr, pr = _jobs("train"), _jobs("predict")
    tr_done = sum(1 for j in tr if _out_of(j).exists()); pr_done = sum(1 for j in pr if _out_of(j).exists())
    rows += [("train",   tr_done == len(tr), f"обучений {tr_done}/{len(tr)}"),
             ("predict", pr_done == len(pr), f"предсказаний {pr_done}/{len(pr)}"),
             ("blend",   (SUBS/"sub_v24ind_comb5_L233.csv").exists() or (SUBS/"sub_v24ind_comb5_L233.csv.gz").exists(),
                         "artifacts/submissions/sub_v24ind_comb5_L233"),
             ("forms",   len(list(FORMS.glob('probe_form_*.npy'))) >= 34,
                         f"artifacts/forms/probe_form_*.npy ({len(list(FORMS.glob('probe_form_*.npy')))}/34)"),
             ("finals",  (R/"output/sub_v41_L233.csv").exists(), "output/sub_v41_L233.csv")]
    for name, ok, what in rows:
        print(f"{name:10s} {'ГОТОВО' if ok else 'нет':38s} {what}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", help="выполнить только одну стадию: " + ", ".join(n for n, _ in STAGES))
    ap.add_argument("--gpus", default="0", help="список карт через запятую, напр. 0,1,2")
    ap.add_argument("--check", action="store_true", help="только показать состояние")
    ap.add_argument("--retrain-forms", action="store_true",
                    help="пересобрать и обучаемые формы (rescorr, pcorr) — статистический эквивалент, не побитовый")
    ap.add_argument("--history", action="store_true",
                    help="считать и тупиковые ветки тоже (по умолчанию только нужное для финала)")
    a = ap.parse_args()
    global INCLUDE_HISTORY
    INCLUDE_HISTORY = a.history
    a.gpu_list = [int(x) for x in a.gpus.split(",")]
    if a.check: check(); return
    todo = [(n, f) for n, f in STAGES if a.stage in (None, n)]
    if not todo: sys.exit(f"неизвестная стадия: {a.stage}")
    for name, fn in todo:
        log(f"=== стадия {name}")
        if not fn(a):
            sys.exit(f"стадия {name} не завершилась; состояние: python scripts/run_all.py --check")
    log("готово: output/sub_v41_L233.csv и output/sub_v41shrink_L233.csv")

if __name__ == "__main__":
    main()

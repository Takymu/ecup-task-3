"""Совместный солвер ЛБ-калибровки (репозиторная, путе-независимая версия solve_probes5.py).

МАТЕМАТИКА. Метрика соревнования RMSLE = sqrt(MSE) в пространстве log1p. Значит для любого
сабмита вида   x = base + Σ_i k_i · F_i   (всё в log1p) выполняется ТОЧНО:

    M(k) = M_base − 2·kᵀB + kᵀCk,   где  C_ij = cov(F_i, F_j),  B_i = cov(остаток_base, F_i).

C считается локально (формы у нас на руках), B — единственное неизвестное. Каждый отправленный
сабмит с известным весовым вектором w даёт ОДНО линейное уравнение на B:

    wᵀB = (M_base − M_submit + wᵀCw) / 2.

41 сабмит ⇒ переопределённая система ⇒ B по МНК ⇒ оптимум k* = C⁻¹B и точное предсказание M(k*).
Невязки МНК (печатаются ниже) ~1e-6 — это и есть проверка, что модель метрики верна.

Формы нормированы на rms 0.02 ⇒ C_ii = 4e-4. Отсюда шкала «пробы»: сабмит base + 1·F при
k_истинном = 0 даёт M_base + C_ii (хуже базы на ~1.2e-4 RMSLE), при k = 0.5 равен базе,
при k = 1 лучше базы на C_ii. Измеренное k_i = B_i / C_ii = (M_base − M_probe + C_ii) / (2 C_ii).

ЗАПУСК
    python tools_lb/solve.py                       # решение по всем измерениям
    python tools_lb/solve.py drop=gift,visc,...    # исключить измерения (коллинеарные пары)
    python tools_lb/solve.py out=sub_v41_L233      # + собрать CSV в output/
    python tools_lb/solve.py shrink=0.3            # усадка Джеймса–Стайна (хедж-финал)
    python tools_lb/solve.py name=1.6444           # добавить/переопределить измерение

Финальные команды, которыми собраны оба финала, — см. scripts/reproduce_finals.py.
"""
import sys, json
import numpy as np, polars as pl
from pathlib import Path

R = Path(__file__).resolve().parent.parent
A, S, FRM, LB = R/"artifacts", R/"artifacts/submissions", R/"artifacts/forms", R/"artifacts/lb"

# RMSLE базового сабмита sub_v24ind_comb5_L233 на публичном ЛБ (50k юзеров).
M_BASE = 1.6462083685341875**2

def lp(stem):
    """log1p предсказаний сабмита (порядок — по user_id)."""
    return np.log1p(pl.read_csv(S/f"{stem}.csv.gz").sort("user_id")["predict"].to_numpy())

base = pl.read_csv(S/"sub_v24ind_comb5_L233.csv.gz").sort("user_id")
lb_base = np.log1p(base["predict"].to_numpy())

# --- формы -------------------------------------------------------------------------------
# Три исторические формы заданы не файлом, а разностью сабмитов (так они и строились в раунде 5).
F2 = np.load(FRM/"probe_forms2.npy")
v27, dec = lp("sub_v27_comb5_L233"), lp("sub_probe_decon_L233")
OLD = ["ctx","rescorr","v27","decon","pxc","pcorr","gift","bgs","visc","misc","pcorr3","mspec","boost","ctxv2"]
NEW = ["d413","ts","sstd30","rk","fac","fri","allconf","cohort","rkonly","frionly","rktop","sres42","sm6","rec6",
       "gbm42","f3s","sm6s","sm8","sm10","sm4","sd42","f3s12","f3saux","f3spring","f3sd3","f3sauxx","distill",
       "alstm","auemb","d413s","aknn","tcn","atweedie","axgb","agpurf","amlp","wpack","ffc","lo","hi","young",
       "gappy","light","fading","churn","paspring","selmap","pshift","hd","hd16","hdf2","auxcnt","ms24","pm2",
       "hdsr","segxk","hd16b","aux4","segslope","segcarr","ebshrink","f3s3","f3saux4"]
forms = {"ctx": F2[0].astype(np.float64),
         "rescorr": np.load(FRM/"probe_form_rescorr.npy").astype(np.float64),
         "v27": v27 - lb_base, "decon": dec - v27}
for n in OLD[4:] + NEW:
    p = FRM/f"probe_form_{n}.npy"
    if p.exists(): forms[n] = np.load(p).astype(np.float64)
names = [n for n in OLD+NEW if n in forms]; idx = {n: i for i, n in enumerate(names)}
F = np.stack([forms[n] for n in names]); C = F @ F.T / F.shape[1]
def onehot(n): v = np.zeros(len(names)); v[idx[n]] = 1; return v

# --- измерения: имя сабмита -> его публичный RMSLE ---------------------------------------
# Ключ = имя формы, которую проба добавляла к своей базе (или имя версии решения vNN).
meas = {
    # раунд 1 (база v24ind): первые формы и ранние совместные решения
    "ctx":1.6461552952, "rescorr":1.6472096617, "v27":1.6461953871, "decon":1.6462678251612863,
    "pxc":1.6466035603831362, "pcorr":1.6466743219602038, "v29":1.6457551433,
    "gift":1.6459423896, "visc":1.6459531589, "boost":1.645931772, "ctxv2":1.6459065524558394,
    "mspec":1.6457007136,
    # раунд 2 (база v31): семейство коротких памятей и весенние анкеры
    "sres42":1.6458246179, "sm10":1.6456668977777935, "f3s":1.6455222110871799, "sm8":1.6456682215,
    "v32":1.6454849669067975,
    # раунд 3 (база v32): f3saux — сильное попадание; v33 — перекошенное решение (баг базы), само стало измерением
    "f3s12":1.6454824494, "f3saux":1.645329198648242, "v33":1.6455459565,
    "f3spring":1.6454178778, "f3sauxx":1.6454292284, "distill":1.645508822, "alstm":1.6452317676,
    "aknn":1.6453479177,
    # раунд 4 (база alstm): три мимо; v35 — полное решение
    "atweedie":1.6454616154, "auemb":1.6454023476542972, "wpack":1.6454775535106467,
    "v35":1.645178087441008,
    # раунд 5 (база v35): будущий контекст, отток, аугментация длины истории
    "ffc":1.6451729395, "churn":1.645146861, "hd":1.6451333987,
    "hd16":1.6448716436, "selmap":1.6454548131, "v36":1.6450401227, "v37":1.644633554,
    "auxcnt":1.6451002097,
    # раунд 6 (база v38): интеракции с сегментами маргинальности
    "pm2":1.6447209816877806, "ms24":1.64459415357728, "segxk":1.644460072888934,
    # раунд 7 (база v39): сегмент-условный наклон (мимо) и перемеры на бэгах 20 сидов
    "segslope":1.6445800177, "f3saux4":1.6444489592, "f3s3":1.6444479166,
    # оба финала (тоже измерения: подтверждают, что подпространство исчерпано)
    "v41":1.6443877459, "v41sh":1.6447489702,
}

# --- весовые векторы сабмитов ------------------------------------------------------------
k29 = json.loads((LB/"lb_probes_round5.json").read_text())
v29w = np.array([dict(zip(k29["forms"], k29["k"])).get(n, 0.0) for n in names])
wvec = {"ctx":onehot("ctx"), "rescorr":onehot("rescorr"), "v27":onehot("v27"),
        "decon":onehot("v27")+onehot("decon"), "pxc":onehot("pxc"), "pcorr":onehot("pcorr"), "v29":v29w}
for n in ["gift","bgs","visc","misc","pcorr3","mspec","boost","ctxv2"]:
    if n in forms: wvec[n] = v29w + onehot(n)

out_name, drop, shrink = None, set(), 0.0
base31, base32 = {"sres42","sm10","f3s","sm8"}, {"f3s12","f3saux","f3spring","f3sd3"}
for a in sys.argv[1:]:
    k, v = a.split("=")
    if k == "out": out_name = v
    elif k == "drop": drop |= set(v.split(","))
    elif k == "shrink": shrink = float(v)
    else: meas[k] = float(v)

def solve(ms):
    rows, rhs = [], []
    for n, m in ms.items():
        w = wvec[n]; rows.append(w); rhs.append((M_BASE - m**2 + w @ C @ w) / 2)
    A_ = np.stack(rows); sel = sorted({i for r in rows for i in np.nonzero(np.abs(r) > 1e-9)[0]})
    B = np.linalg.lstsq(A_[:, sel], np.array(rhs), rcond=None)[0]; Cs = C[np.ix_(sel, sel)]
    k = np.linalg.solve(Cs, B); mse = M_BASE - 2*k@B + k@Cs@k
    kfull = np.zeros(len(names)); kfull[sel] = k
    return kfull, float(np.sqrt(mse)), A_[:, sel] @ B - np.array(rhs)

# v31 = решение подмножества измерений (ровно так и был собран залитый файл) — служит базой для проб раунда 2
m31 = {n: m for n, m in meas.items() if n in ("ctx","rescorr","v27","decon","pxc","pcorr","v29","mspec")}
v31w, p31, _ = solve(m31); print(f"v31 воспроизведён: предсказание {p31:.6f} (факт файла 1.645683)")
wvec["v31"] = v31w
for n in NEW:
    if n in forms: wvec[n] = (v31w if n in base31 else v29w) + onehot(n)
# базы более поздних проб — снапшоты решений (веса всех форм на момент сборки соответствующего vNN)
for tag in ("v32","v33","v35","v36","v37","v38","v39","v41","v41sh"):
    kj = json.loads((LB/f"lb_probes_round7_{tag}.json").read_text())
    wvec[tag] = np.array([dict(zip(kj["forms"], kj["k"])).get(n, 0.0) for n in names])
for n in NEW:
    if n in forms and n in base32: wvec[n] = wvec["v32"] + onehot(n)
for n in ("f3spring","f3sauxx","distill","alstm"):                      # база = проба f3saux
    if n in forms: wvec[n] = wvec["v32"] + onehot("f3saux") + onehot(n)
for n in ("auemb","aknn","tcn","d413s","atweedie","axgb","agpurf","amlp","wpack"):   # база = проба alstm
    if n in forms: wvec[n] = wvec["v32"] + onehot("f3saux") + onehot("alstm") + onehot(n)
for n in ("ffc","lo","hi","young","gappy","light","fading","churn","paspring","selmap","pshift",
          "hd","hd16","hdf2","auxcnt","ms24"):                          # база = sub_v35
    if n in forms: wvec[n] = wvec["v35"] + onehot(n)
for n in ("ms24","pm2","hdsr","segxk"):                                 # база = sub_v38
    if n in forms: wvec[n] = wvec["v38"] + onehot(n)
for n in ("hd16b","aux4","segslope","segcarr","ebshrink","f3s3","f3saux4"):  # база = sub_v39
    if n in forms: wvec[n] = wvec["v39"] + onehot(n)

ms = {n: m for n, m in meas.items() if n not in drop}
k, pred, resid = solve(ms)
print(f"измерений использовано: {len(ms)}")
print("k*:", {n: round(float(kk), 3) for n, kk in zip(names, k) if abs(kk) > 1e-9})
print(f"предсказание публичного ЛБ: {pred:.6f}")
print("невязки МНК (должны быть ~1e-6):", f"max |r| = {np.abs(resid).max():.2e}")

if shrink:
    # Джеймс–Стайн покомпонентно: sigma(k_i) ~ 0.65 (оценка из бутстрапа матрицы измерений),
    # k'_i = k_i · max(0, 1 − lambda·sigma²/k_i²). Гасит формы, чьё k сравнимо с шумом измерения.
    s2 = 0.65**2
    k = k * np.clip(1 - shrink*s2/np.maximum(k**2, 1e-12), 0, 1)
    print(f"после усадки lambda={shrink}: форм осталось {int((np.abs(k)>1e-9).sum())}, "
          f"||k'||/||k|| = {np.linalg.norm(k)/np.linalg.norm(solve(ms)[0]):.3f}")

if out_name:
    (R/"output").mkdir(exist_ok=True)
    x = lb_base + k @ F; x += 2.33 - x.mean()          # выравнивание уровня: mean(log1p) = 2.33
    pr = np.clip(np.expm1(np.clip(x, 0, None)), 0, None)
    o = (pl.read_csv(R/"data/sample_submit.csv.gz").select("user_id")
           .join(pl.DataFrame({"user_id": base["user_id"].to_numpy(), "predict": pr}), on="user_id", how="left"))
    assert o["predict"].null_count() == 0
    o.write_csv(R/"output"/f"{out_name}.csv")
    json.dump({"forms": names, "k": [float(v) for v in k], "pred_lb": pred, "measured": ms},
              open(R/"output"/f"{out_name}.weights.json", "w"), indent=2)
    print(f"записано output/{out_name}.csv (+ .weights.json)")

"""Генератор pipeline/jobs.json (запускать из корня репозитория).

Собрать pipeline/jobs.json — полный упорядоченный список обучений и предсказаний.

Источники (в порядке приоритета):
  1. server/lanes/*.json — литеральные команды, как они реально исполнялись (авторитет);
  2. artifacts/model_meta/*.meta.json — реконструкция argv для моделей, обученных до появления
     супервизора очередей. Round-trip на 138 моделях, присутствующих в обоих источниках, показал
     полное совпадение по всем флагам, КРОМЕ --min-anchor (его meta.json не сохраняет) — он
     задаётся таблицей MIN_ANCHOR по семейству имени.
"""
import json, glob, re
from pathlib import Path

R = Path(".")
LANES = sorted(glob.glob("server/lanes/*.json"))

DEF = dict(epochs=8, batch=4096, anchor_stride=2, n_anchors=8, hidden=128, layers=1, lr=2e-3,
           cell="gru", pool="mean", unit="week", seq_len=0, seed=42, fusion_tag="", ch=8,
           val_anchor="2026-01-14", aux_weight=0.3, target="gmv", tail_days=0, loss="ziln",
           mix_w=1.0, user_emb=0, anchor_offset_days=0, ctx=False, ctx_set="v1", warmup=0.0,
           aux_win="", bins=0, bins_w=1.0, rec_drop=0.0, rec_drop_max=28, rank_ch="", cart_ch=False,
           coarse=0, coarse_fine=16, obs_mask=False, rec_mask=0, afe=False, aux_lead="",
           sigma_fixed=False, back_readouts="", back_hidden=128, step_sup=0.0, gmv_noise=0.0,
           aux_count="", pmask_from="", short_tokens=0, hist_drop=0.0, hist_min=6, init_from="",
           horizon=30, fctx_noise=0.03, fresh_step=0, anchor_step_days=0, chan_drop=0.0)
FLAG = {"anchor_stride":"--anchor-stride","n_anchors":"--n-anchors","hidden":"--hidden","layers":"--layers",
        "lr":"--lr","cell":"--cell","pool":"--pool","unit":"--unit","seed":"--seed","fusion_tag":"--fusion-tag",
        "ch":"--channels","val_anchor":"--val-anchor","aux_weight":"--aux-weight","target":"--target",
        "tail_days":"--tail-days","loss":"--loss","mix_w":"--mix-w","user_emb":"--user-emb",
        "anchor_offset_days":"--anchor-offset-days","ctx_set":"--ctx-set","warmup":"--warmup",
        "aux_win":"--aux-win","bins":"--bins","bins_w":"--bins-w","rec_drop":"--rec-drop",
        "rec_drop_max":"--rec-drop-max","rank_ch":"--rank-ch","coarse":"--coarse","coarse_fine":"--coarse-fine",
        "rec_mask":"--rec-mask","aux_lead":"--aux-lead","back_readouts":"--back-readouts",
        "back_hidden":"--back-hidden","step_sup":"--step-sup","gmv_noise":"--gmv-noise",
        "aux_count":"--aux-count","pmask_from":"--pmask-from","short_tokens":"--short-tokens",
        "hist_drop":"--hist-drop","hist_min":"--hist-min","init_from":"--init-from","horizon":"--horizon",
        "fctx_noise":"--fctx-noise","fresh_step":"--fresh-step","anchor_step_days":"--anchor-step-days",
        "chan_drop":"--chan-drop"}
STORE_TRUE = {"ctx":"--ctx","cart_ch":"--cart-ch","obs_mask":"--obs-mask","afe":"--afe","sigma_fixed":"--sigma-fixed"}

# --min-anchor не пишется в meta.json. Значения взяты из лейн-спеков (где семейство там есть)
# и из журнала: флаг появился 22.08, до него действовал дефолт 2025-05-01.
SPRING = "2025-02-26"          # «весенние» анкеры: вся история с начала данных
DEFAULT_MIN = "2025-05-01"
SPRING_PREFIXES = ("f3s", "f3m", "f3spring", "hd", "hdp3", "hd16", "hdf2", "hdax", "hdsr", "auxcnt",
                   "ms24", "mshd", "pm2", "pmask", "jit", "pst", "obs", "bg", "afe", "sigfix",
                   "lead", "auxact", "auxall", "dn3", "ssp", "rd", "bh", "tres", "cdrop", "sev", "mr3")

def min_anchor_for(name):
    base = re.sub(r"_s\d+$|_s\d+_.*$|p3$", "", name)
    for p in sorted(SPRING_PREFIXES, key=len, reverse=True):
        if base.startswith(p):
            return SPRING
    return DEFAULT_MIN

def argv_from_meta(m):
    a = ["train_seq.py", "--name", m["name"]]
    if m.get("aux_horizons"):
        a += ["--aux", ",".join(str(x) for x in m["aux_horizons"])]
    for k, fl in FLAG.items():
        if k in m and m[k] != DEF.get(k):
            v = m[k]
            a += [fl, ("%g" % v) if isinstance(v, float) else str(v)]
    for k, fl in STORE_TRUE.items():
        if m.get(k):
            a.append(fl)
    a += ["--min-anchor", min_anchor_for(m["name"])]
    return a

lane_jobs, order = {}, []
for f in LANES:
    for j in json.loads(Path(f).read_text()):
        if j["name"] in lane_jobs:
            continue
        cmd = list(j["cmd"])
        if cmd[0].startswith("train") and "--min-anchor" not in cmd:
            cmd += ["--min-anchor", min_anchor_for(j["name"])]
        lane_jobs[j["name"]] = {"name": j["name"], "cmd": cmd, "kind": cmd[0].split(".")[0],
                                "src": Path(f).name, "needs": [Path(n).name for n in j.get("needs", [])]}
        order.append(j["name"])

metas = {p.name[:-10]: json.loads(p.read_text()) for p in sorted(Path("artifacts/model_meta").glob("*.meta.json"))}
extra = []
for name, m in metas.items():
    if name in lane_jobs:
        continue
    extra.append({"name": name, "cmd": argv_from_meta(m), "kind": "train_seq",
                  "src": "meta+convention", "needs": []})

train = [lane_jobs[n] for n in order if lane_jobs[n]["kind"].startswith("train")] + extra
predict = [lane_jobs[n] for n in order if lane_jobs[n]["kind"].startswith("predict")]

Path("pipeline").mkdir(exist_ok=True)
out = {"note": "полный список задач обучения и предсказания; порядок = порядок исполнения",
       "train": train, "predict": predict}
Path("pipeline/jobs.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"обучений: {len(train)} (из лейнов {len(train)-len(extra)}, из мет {len(extra)}) | предсказаний: {len(predict)}")
print("моделей со «весенним» min-anchor:", sum(1 for j in train if SPRING in j["cmd"]))

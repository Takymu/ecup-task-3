"""Супервизор GPU-лейна: гонит задачи из JSON-списка, каждая с ожидаемым выходным файлом; падение → повтор в более безопасном
режиме (1: как есть, 2: тензор на CPU, 3: CPU + batch 2048); зависимые шаги пропускаются, если входа нет; всё пишется в лог.
Usage (на сервере, из /root/ecup/src): python server_runner.py <lane_jobs.json> <gpu_index> <log>
Формат задачи: {"name": str, "cmd": [argv без интерпретатора], "out": path, "needs": [paths]}; уже готовые (out есть) пропускаются.
Если задачу с тем же --name уже гонит другой процесс (старая очередь) — ждём его и проверяем out."""
import json, os, subprocess, sys, time, re
PY = "/root/mf/envs/ml/bin/python"; M = "/root/ecup/artifacts/models"
jobs, gpu, logp = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); open(logp, "a").write(line + "\n")
def running(name):
    out = subprocess.run(["pgrep", "-f", f"name {name} "], capture_output=True, text=True).stdout.split()
    return [p for p in out if p != str(os.getpid())]
ENV0 = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONIOENCODING="utf-8", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", POLARS_MAX_THREADS="16")
MODES = [dict(), dict(SEQ_CPU_TENSOR="1"), dict(SEQ_CPU_TENSOR="1", SEQ_BATCH="2048")]
for j in jobs:
    name, out = j["name"], j["out"]
    if os.path.exists(out):
        log(f"skip {name}: out exists"); continue
    while running(name):
        log(f"wait {name}: already running elsewhere"); time.sleep(120)
        if os.path.exists(out): break
    if os.path.exists(out):
        log(f"skip {name}: out appeared"); continue
    missing = [n for n in j.get("needs", []) if not os.path.exists(n)]
    if missing:
        log(f"SKIP {name}: missing inputs {missing}"); continue
    for attempt, extra in enumerate(MODES, 1):
        cmd = [PY] + j["cmd"]
        if "SEQ_BATCH" in extra and "--batch" not in cmd and cmd[1].startswith("train_seq"):
            cmd += ["--batch", extra["SEQ_BATCH"]]
        env = dict(ENV0, **{k: v for k, v in extra.items() if k != "SEQ_BATCH"})
        log(f"run {name} attempt {attempt} mode={extra or 'default'}: {' '.join(cmd[1:])[:160]}")
        t0 = time.time()
        with open(f"/root/ecup/logs/{name}.a{attempt}.log", "w") as lf:
            rc = subprocess.run(cmd, cwd="/root/ecup/src", env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        ok = rc == 0 and os.path.exists(out)
        log(f"{'DONE' if ok else 'FAIL'} {name} attempt {attempt} rc={rc} ({(time.time()-t0)/60:.0f} min)")
        if ok: break
        time.sleep(30)
log("LANE_DONE")

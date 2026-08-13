#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4 via `colab run`] Proxy PAREADO del harness TAAF con modelo pequeño.

Pregunta que responde (guía el duck v4 SIN gastar cuota G4): ¿schema_helpers
cambia el comportamiento del agente de forma medible? Brazo A = floor F
(efficiency+retry_guard+shortcircuit, el config que marcó 1.17); brazo B =
A + schema_helpers (helpers precargados en el sandbox python).

Modelo proxy: Qwen/Qwen3-4B (FP16, cabe en T4; misma familia y mismo modo
thinking que el 27B del duck). Regla AG2: los NIVELES ABSOLUTOS de un 4B no
transfieren; las señales RELATIVAS y de comportamiento sí:
  - ¿el modelo LLAMA los helpers precargados? (adopción)
  - ¿deja de reescribir su propia plomería de grids? (def connected_components)
  - tracebacks del sandbox por juego (tasa de error de exec)
  - tokens y acciones por juego, niveles en juegos fáciles

Uso:  colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_taaf_proxy.py <KAGGLE_API_TOKEN>
Salida: resumen JSON por stdout + /content/taaf_proxy_result.json en la VM.
"""

import asyncio
import json
import os
import pickle
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- setup
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN
os.environ["KAGGLE_USERNAME"] = "juliancamilovilla"

GAMES = ("tu93", "sc25", "cd82", "vc33", "su15", "sb26")  # canario + 2 rápidos
PER_GAME_S = 420          # 7 min/juego/brazo
ARM_BACKSTOP_MIN = 38     # tope duro por brazo
MODEL = "Qwen/Qwen3-4B"

t0 = time.time()
def log(m):
    print(f"[proxy {time.time()-t0:6.0f}s] {m}", flush=True)

log("pip install (vllm, kaggle) ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "kaggle"],
               check=True)

os.makedirs("/content", exist_ok=True)
os.chdir("/content")

log("descargando data de la competencia (wheels + environment_files) ...")
subprocess.run(["kaggle", "competitions", "download", "arc-prize-2026-arc-agi-3",
                "-p", "/content", "-q"], check=True)
subprocess.run("cd /content && for z in *.zip; do unzip -o -q \"$z\"; done",
               shell=True, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--find-links",
                "/content/arc_agi_3_wheels", "arcengine", "arc-agi"], check=True)

log("descargando bundle del fork ...")
subprocess.run(["kaggle", "datasets", "download",
                "thtennant/taaf-kaggle-source-share-fork",
                "-p", "/content/bundle", "--unzip", "-q"], check=True)
BUNDLE = Path("/content/bundle")
for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src", "taaf-grafts"):
    sys.path.insert(0, str(BUNDLE / "src" / repo))

# ---------------------------------------------------------------- vLLM server
log(f"lanzando vLLM con {MODEL} (T4: dtype half, backend xformers) ...")
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"  # T4 (sm75): sin flash-attn
vllm_log = open("/content/vllm.log", "w")
vllm_proc = subprocess.Popen(
    [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
     "--model", MODEL, "--port", "1234", "--dtype", "half",
     "--max-model-len", "16384", "--gpu-memory-utilization", "0.92",
     "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
     "--disable-log-requests"],
    stdout=vllm_log, stderr=subprocess.STDOUT)
deadline = time.time() + 1500
while time.time() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    if vllm_proc.poll() is not None:
        print(Path("/content/vllm.log").read_text()[-4000:])
        raise RuntimeError("vLLM murió durante el arranque")
    time.sleep(10)
else:
    raise RuntimeError("vLLM no respondió en 25 min")
log("vLLM listo")

# ---------------------------------------------------------------- entorno TAAF
# Contrato leído de taaf_setup_env.json del duck real; misma config de sampling
# (temp 0.6, thinking on) para que el proxy comparta régimen con el 27B.
os.environ.update({
    "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1234/v1",
    "OPENAI_BASE_URL": "http://127.0.0.1:1234/v1",
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": MODEL,
    "INFERENCE_ANALYZER_MODEL": MODEL,
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "16384",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "",   # Qwen3-4B es texto puro: sin imagen adjunta
    "VLLM_NO_USAGE_STATS": "1",
    "USE_TF": "0",
})

import arc_agi  # noqa: E402
import taaf.game_api  # noqa: E402


def offline_games(env_dir):
    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    ar = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    return [taaf.game_api.GameAPI(env_name=e.game_id, arcade_spec=spec)
            for e in ar.available_environments
            if e.game_id.split("-")[0] in GAMES]


def run_arm(name, flags):
    log(f"=== brazo {name}: flags={sorted(k for k, v in flags.items() if v)} ===")
    job = Path(f"/content/job_{name}")
    job.mkdir(exist_ok=True)
    os.environ["RECORDINGS_DIR"] = str(job / "rec")
    with open(BUNDLE / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    bm.job_dir = job
    bm.n_passes = 1
    bm.game_weights = None
    bm.solver.max_runtime_s_per_game = PER_GAME_S
    bm.solver.concurrency = 3
    bm.solver.model = MODEL

    from taaf_grafts.composite import install
    install(bm, flags=flags)

    bm.games = offline_games("/content/environment_files")
    log(f"{len(bm.games)} juegos: {[g.env_name for g in bm.games]}")

    from datetime import datetime, timedelta
    soft_end = datetime.now() + timedelta(minutes=ARM_BACKSTOP_MIN)
    asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=None,
                       minimal_diagnostics=False))

    # métricas de comportamiento desde transcripts + summary
    metrics = {"games": {}, "helper_calls": 0, "own_plumbing_defs": 0,
               "tracebacks": 0, "action_calls": 0}
    helpers = ("grid_diff", "connected_components", "action_effect_summary",
               "recent_history")
    tdir = job / "transcripts"
    if tdir.is_dir():
        for t in tdir.glob("*.txt"):
            txt = t.read_text(errors="replace")
            metrics["helper_calls"] += sum(
                len(re.findall(rf"(?<!def ){h}\(", txt)) for h in helpers)
            metrics["own_plumbing_defs"] += sum(
                len(re.findall(rf"def {h}\w*\(", txt)) for h in helpers)
            metrics["tracebacks"] += txt.count("Traceback (most recent call")
            metrics["action_calls"] += len(re.findall(r"\baction\(", txt))
    summ = job / "summary.txt"
    if summ.exists():
        text = summ.read_text()
        metrics["summary"] = text[:1500]
        for m in re.finditer(r"(\w+)-\w+: score=([\d.]+), levels=([\d.]+)/(\d+), "
                             r"actions=(\d+), tokens=(\d+)", text):
            metrics["games"][m.group(1)] = {
                "levels": float(m.group(3)), "actions": int(m.group(5)),
                "tokens": int(m.group(6))}
    log(f"brazo {name}: {json.dumps({k: v for k, v in metrics.items() if k != 'summary'})[:400]}")
    return metrics


FLOOR = {"efficiency": True, "retry_guard": True, "shortcircuit": True}
result = {"model": MODEL, "games": GAMES, "per_game_s": PER_GAME_S}
result["A_floor"] = run_arm("A", dict(FLOOR))
result["B_helpers"] = run_arm("B", dict(FLOOR, schema_helpers=True))

vllm_proc.terminate()
Path("/content/taaf_proxy_result.json").write_text(json.dumps(result, indent=2))
print("\n===== TAAF PROXY RESULT =====")
print(json.dumps(result, indent=2)[:6000])
log("done")

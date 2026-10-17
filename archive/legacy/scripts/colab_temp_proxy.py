#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Proxy #2 PAREADO: temperatura 0.6 vs 0.2 sobre el config v4.

Motivación (medido en el set oculto): el duck con código idéntico dio
{1.17, 1.03, 0.76} — rango 41 niveles — y corre a temp 0.6 + thinking ON.
Pregunta: ¿bajar la temperatura comprime la dispersión sin degradar la
exploración? Si la calidad/diversidad de acciones se mantiene a temp 0.2,
es un lever de varianza gratis para el duck.

Diseño (lecciones del proxy #1: T4 a 14-18 tok/s con thinking ON = timeouts
y 0-4 acciones/juego):
  - AMBOS brazos con thinking OFF (throughput ~3-5x) y schema_helpers ON
    (= config v4 desplegado). La divergencia thinking OFF vs el duck real se
    acepta: la señal buscada es RELATIVA entre temperaturas.
  - 4 juegos x 9 min/brazo (su15/sb26 suben nivel rápido; tu93/cd82 canario).
Métricas por brazo: acciones y niveles por juego, tokens, acciones DISTINTAS
(diversidad de exploración), repeticiones consecutivas, tracebacks, helper calls.

Uso:  colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_temp_proxy.py <KAGGLE_API_TOKEN>
Salida: JSON por stdout + /content/temp_proxy_result.json.
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

TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_API_TOKEN", "")
assert TOKEN, "pasa el KAGGLE_API_TOKEN como argv[1]"
os.environ["KAGGLE_API_TOKEN"] = TOKEN
os.environ["KAGGLE_USERNAME"] = "juliancamilovilla"

# 3 juegos (los que mostraron actividad real con el 4B) con MÁS tiempo cada uno:
# el proxy #2 murió de anemia (11 acciones totales). Concurrency 3 = 1 sola tanda.
GAMES = ("su15", "sb26", "cd82")
PER_GAME_S = 720
ARM_BACKSTOP_MIN = 26
MODEL = "Qwen/Qwen3-4B"

t0 = time.time()
def log(m):
    print(f"[temp-proxy {time.time()-t0:6.0f}s] {m}", flush=True)

# Heartbeat: el kernel client del CLI aborta con "Timeout waiting for output"
# ante silencios largos (bm.run calla ~10 min). Un print por minuto lo evita.
import threading as _th
def _heartbeat():
    while True:
        time.sleep(60)
        print(f"[hb {time.time()-t0:6.0f}s]", flush=True)
_th.Thread(target=_heartbeat, daemon=True).start()


def persist_to_kaggle(payload: dict, note: str):
    """Backstop: si el cliente CLI muere, el resultado sobrevive como dataset."""
    try:
        d = Path("/content/proxy_upload")
        d.mkdir(exist_ok=True)
        (d / "temp_proxy_result.json").write_text(json.dumps(payload, indent=2))
        (d / "dataset-metadata.json").write_text(json.dumps({
            "title": "arc3 temp proxy result",
            "id": "juliancamilovilla/arc3-temp-proxy-result",
            "licenses": [{"name": "CC0-1.0"}]}))
        r = subprocess.run(["kaggle", "datasets", "version", "-p", str(d),
                            "-m", note], capture_output=True, text=True)
        if "successfully" not in (r.stdout or "").lower():
            subprocess.run(["kaggle", "datasets", "create", "-p", str(d)],
                           capture_output=True, text=True)
        log(f"persistido a Kaggle ({note})")
    except Exception as exc:  # noqa: BLE001 — backstop, nunca tumbar el run
        log(f"persist_to_kaggle falló: {type(exc).__name__}: {exc}")

log("pip install (vllm, kaggle) ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "kaggle"],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchaudio"],
               check=False)
os.environ.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                   "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1"})

os.makedirs("/content", exist_ok=True)
os.chdir("/content")

log("descargando data de la competencia ...")
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

log(f"lanzando vLLM con {MODEL} ...")
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
vllm_log = open("/content/vllm.log", "w")
vllm_proc = subprocess.Popen(
    [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
     "--model", MODEL, "--port", "1234", "--dtype", "half",
     "--max-model-len", "16384", "--gpu-memory-utilization", "0.92",
     "--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
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

BASE_ENV = {
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
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "false",  # T4: throughput sobre parity
    "MULTIMODAL_CONTEXT": "",
}
os.environ.update(BASE_ENV)

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


# Contador de fallos del analyzer (timeouts contra vLLM): el harness los emite
# por logging, no van a los transcripts. Sin esto no se distingue "el modelo
# decidió poco" de "las peticiones se cayeron" — el error que invalidó el #2.
import logging as _logging

class _AnalyzerFailCounter(_logging.Handler):
    def __init__(self):
        super().__init__(level=_logging.WARNING)
        self.n = 0
    def emit(self, record):
        if "analyzer request failed" in record.getMessage():
            self.n += 1

_fail_counter = _AnalyzerFailCounter()
_logging.getLogger("inference.agent.tool_agent").addHandler(_fail_counter)


def ensure_vllm_alive():
    """El server puede morir entre brazos (visto: brazo B con 0 tokens y 33
    tracebacks contra un puerto muerto). Verificar y reiniciar si hace falta."""
    global vllm_proc
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=10) as r:
            if r.status == 200:
                return
    except Exception:
        pass
    log("vLLM caído; reiniciando ...")
    try:
        vllm_proc.kill()
    except Exception:
        pass
    vllm_proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--port", "1234", "--dtype", "half",
         "--max-model-len", "16384", "--gpu-memory-utilization", "0.92",
         "--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
        stdout=open("/content/vllm2.log", "w"), stderr=subprocess.STDOUT)
    dl = time.time() + 1200
    while time.time() < dl:
        try:
            with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5) as r:
                if r.status == 200:
                    log("vLLM reiniciado OK")
                    return
        except Exception:
            pass
        if vllm_proc.poll() is not None:
            raise RuntimeError("vLLM no pudo reiniciar")
        time.sleep(10)
    raise RuntimeError("vLLM no respondió tras reinicio")


def run_arm(name, temperature):
    log(f"=== brazo {name}: temp={temperature}, thinking OFF, helpers ON ===")
    ensure_vllm_alive()
    os.environ["LOCAL_ANALYZER_TEMPERATURE"] = str(temperature)
    # BUG confirmado: _LOCAL_ANALYZER_TEMPERATURE es global de módulo congelado
    # al primer import (tool_agent.py:145) — la env var entre brazos NO basta.
    # Mismo seam que el composite usa para context_window: parchear el global.
    import inference.agent.tool_agent as _ta
    _ta._LOCAL_ANALYZER_TEMPERATURE = float(temperature)
    log(f"tool_agent._LOCAL_ANALYZER_TEMPERATURE = {_ta._LOCAL_ANALYZER_TEMPERATURE}")
    _fail_counter.n = 0
    job = Path(f"/content/job_{name}")
    job.mkdir(exist_ok=True)
    os.environ["RECORDINGS_DIR"] = str(job / "rec")
    with open(BUNDLE / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    bm.job_dir = job
    bm.n_passes = 1
    bm.game_weights = None
    bm.solver.max_runtime_s_per_game = PER_GAME_S
    bm.solver.concurrency = 3  # = len(GAMES): una sola tanda, sin colas
    bm.solver.model = MODEL

    from taaf_grafts.composite import install
    install(bm, flags={"efficiency": True, "retry_guard": True,
                       "shortcircuit": True, "schema_helpers": True})

    bm.games = offline_games("/content/environment_files")
    log(f"{len(bm.games)} juegos: {[g.env_name for g in bm.games]}")

    from datetime import datetime, timedelta
    soft_end = datetime.now() + timedelta(minutes=ARM_BACKSTOP_MIN)
    import threading
    box = {}
    def _target():
        try:
            box["v"] = asyncio.run(bm.run(soft_end_time=soft_end,
                                          runtime_environment=None,
                                          minimal_diagnostics=False))
        except BaseException as exc:  # noqa: BLE001
            box["err"] = exc
    th = threading.Thread(target=_target)
    th.start()
    th.join()
    if "err" in box:
        raise box["err"]

    metrics = {"temperature": temperature, "games": {}, "helper_calls": 0,
               "tracebacks": 0, "distinct_actions": 0, "repeat_runs": 0,
               "analyzer_failures": _fail_counter.n}
    # OJO (bug del proxy #1): la nota HELPERS del prompt menciona los 4 helpers
    # con parentesis CADA TURNO ("grid_diff(a,b)...") y los transcripts incluyen
    # el prompt → contar "helper(" a secas infla la adopcion. Filtro: contar solo
    # llamadas con argumentos que NO sean la firma literal de la nota.
    helpers = ("grid_diff", "connected_components", "action_effect_summary",
               "recent_history")
    _note_sigs = ("grid_diff(a,b)", "connected_components(grid, colors=None)",
                  "action_effect_summary(before,after)", "recent_history(n)")
    tdir = job / "transcripts"
    if tdir.is_dir():
        for t in tdir.glob("*.txt"):
            txt = t.read_text(errors="replace")
            for sig in _note_sigs:
                txt = txt.replace(sig, "")
            metrics["helper_calls"] += sum(
                len(re.findall(rf"(?<!def ){h}\(", txt)) for h in helpers)
            metrics["tracebacks"] += txt.count("Traceback (most recent call")
            # diversidad: acciones ejecutadas (UP/DOWN/LEFT/RIGHT/MOUSE...) y
            # rachas repetidas (mismo action 3+ veces seguidas)
            acts = re.findall(r"action\(\[?['\"]?(UP|DOWN|LEFT|RIGHT|SPACE|MOUSE)",
                              txt)
            metrics["distinct_actions"] += len(set(acts))
            metrics["repeat_runs"] += len(re.findall(
                r"(UP|DOWN|LEFT|RIGHT|SPACE)(?:\W+\1){2,}", " ".join(acts)))
    summ = job / "summary.txt"
    if summ.exists():
        text = summ.read_text()
        metrics["summary"] = text[:1200]
        for m in re.finditer(r"(\w+)-\w+: score=([\d.]+), levels=([\d.]+)/(\d+), "
                             r"actions=(\d+), tokens=(\d+)", text):
            metrics["games"][m.group(1)] = {
                "levels": float(m.group(3)), "actions": int(m.group(5)),
                "tokens": int(m.group(6))}
    log(f"brazo {name}: {json.dumps({k: v for k, v in metrics.items() if k != 'summary'})[:400]}")
    return metrics


result = {"model": MODEL, "games": GAMES, "per_game_s": PER_GAME_S,
          "design": "thinking OFF ambos brazos, schema_helpers ON (config v4)"}
result["A_temp06"] = run_arm("A", 0.6)
persist_to_kaggle(result, "arm A done")
result["B_temp02"] = run_arm("B", 0.2)
persist_to_kaggle(result, "arm B done")

vllm_proc.terminate()
Path("/content/temp_proxy_result.json").write_text(json.dumps(result, indent=2))
print("\n===== TEMP PROXY RESULT =====")
print(json.dumps(result, indent=2)[:6000])
log("done")

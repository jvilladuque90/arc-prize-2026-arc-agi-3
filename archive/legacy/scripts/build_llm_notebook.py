"""Genera notebooks/llm.ipynb: NUESTRO agente LLM (Fase 3) en la G4.

A diferencia del duck-harness (clon del ganador), este es nuestro agente:
  - arc3.llm_agent.LLMAgent: VLM con features objetuales de arc3 inyectadas en el prompt
    (nuestro diferenciador) + memoria failed-state + fallback a GraphExplorer.
  - Reusa piezas de infra públicas y probadas en RTX Pro 6000: wheelhouse vLLM y el
    snapshot Qwen3-27B-FP8 (datasets públicos), pero el arranque y el loop son nuestros.

Datasets requeridos (adjuntar al pushear con --dataset):
  driessmit1/arc3-vllm-h100-wheelhouse-v3      (wheels vLLM 0.19 / torch 2.10 / flashinfer)
  driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot (pesos ~27GB FP8)

Cuota G4: en Save & Run corre TAAF_OFFLINE_SOFT_MIN (default 30) min de validación.
Uso:  python scripts/build_llm_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "arc3"
OUT = ROOT / "notebooks" / "llm.ipynb"

SETUP = '''\
# NUESTRO agente LLM en la G4 (RTX Pro 6000). Offline, wheels+modelo de datasets públicos.
import json, os, subprocess, sys, time
from pathlib import Path
from urllib.request import urlopen

TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1","true"}
NOTEBOOK_START = time.time()
os.environ["ONLY_RESET_LEVELS"] = "true"
os.environ["MPLBACKEND"] = "Agg"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    e for e in ["/usr/local/nvidia/lib64", os.environ.get("LIBRARY_PATH","")] if e)
OFFLINE_SOFT_MIN = float(os.environ.get("TAAF_OFFLINE_SOFT_MIN", "30"))

def find_input(marker_names):
    for dp, dns, fns in os.walk("/kaggle/input"):
        names = set(dns) | set(fns)
        for m in marker_names:
            if m in names:
                return Path(dp)
    return None

COMP_ROOT = find_input(["arc_agi_3_wheels"])
WHEELHOUSE = find_input(["vllm"]) or None
# el snapshot del modelo trae config.json + *.safetensors
MODEL_DIR = None
for dp, dns, fns in os.walk("/kaggle/input"):
    if "config.json" in fns and any(f.endswith(".safetensors") for f in fns):
        MODEL_DIR = Path(dp); break
assert COMP_ROOT, "wheelhouse de competencia no encontrado"
print("COMP_ROOT=", COMP_ROOT, "MODEL_DIR=", MODEL_DIR)

# arc-agi (competencia) offline
subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--no-index",
    "--no-warn-conflicts","--disable-pip-version-check",
    f"--find-links={COMP_ROOT/'arc_agi_3_wheels'}","arc-agi"], stdout=subprocess.DEVNULL)
import arc_agi
print("arc_agi OK")
'''

VLLM = '''\
# Instalar vLLM del wheelhouse público y arrancar el server OpenAI-compatible.
VLLM_HOST, VLLM_PORT = "127.0.0.1", 1234
VLLM_BASE = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
SERVED = "arc-qwen"
SERVER_LOG = Path("/kaggle/working/vllm.log")

def find_wheelhouse():
    for dp, dns, fns in os.walk("/kaggle/input"):
        if any(f.startswith("vllm-") and f.endswith(".whl") for f in fns):
            return Path(dp)
    return None

wh = find_wheelhouse()
if wh is not None and MODEL_DIR is not None:
    subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--no-index",
        f"--find-links={wh}","vllm"], stdout=subprocess.DEVNULL)
    # El crash previo fue el autotuner de flashinfer al capturar el grafo. --enforce-eager
    # salta torch.compile + cudagraph (donde se dispara el autotune); FLASH_ATTN evita la
    # ruta flashinfer de atención; desactivamos el sampler flashinfer. max-model-len 16k
    # basta para nuestros prompts cortos y baja presión de KV.
    # El crash es el kernel GEMM FP8 de flashinfer (FlashInferFP8ScaledMMLinearKernel):
    # su autotuner JIT muere. Forzamos el kernel Marlin FP8 (Triton, sin flashinfer) con
    # VLLM_TEST_FORCE_FP8_MARLIN=1. Sumado a FLASH_ATTN + enforce-eager, no queda ninguna
    # ruta que dependa de flashinfer.
    serve_env = os.environ.copy()
    serve_env.update({
        "VLLM_TEST_FORCE_FP8_MARLIN": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_NO_USAGE_STATS": "1",
    })
    cmd = [sys.executable,"-m","vllm.entrypoints.openai.api_server",
           "--model", str(MODEL_DIR), "--served-model-name", SERVED,
           "--host", VLLM_HOST, "--port", str(VLLM_PORT),
           "--tensor-parallel-size","1","--enforce-eager","--enable-prefix-caching",
           "--max-model-len","16384","--gpu-memory-utilization","0.90"]
    print("arrancando vLLM:", " ".join(cmd), flush=True)
    logf = SERVER_LOG.open("w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=serve_env)
    # esperar readiness
    ready = False
    dl = time.monotonic() + 900
    while time.monotonic() < dl:
        try:
            with urlopen(f"{VLLM_BASE}/models", timeout=5) as r:
                if r.status == 200:
                    ready = True; break
        except Exception:
            pass
        if proc.poll() is not None:
            print("vLLM murió:\\n", SERVER_LOG.read_text()[-2000:]); break
        time.sleep(5)
    print("vLLM ready:", ready, flush=True)
else:
    ready = False
    print("sin wheelhouse/modelo -> el agente correrá SOLO con fallback GraphExplorer")
'''

CHAT = '''\
# chat_fn: cliente OpenAI-compatible contra el vLLM local (texto; imagen opcional).
import urllib.request

def chat_fn(system, user_text, image_data_uri):
    content = [{"type":"text","text":user_text}]
    if image_data_uri:
        content.append({"type":"image_url","image_url":{"url":image_data_uri}})
    body = {"model": SERVED, "messages":[
                {"role":"system","content":system},
                {"role":"user","content": content if image_data_uri else user_text}],
            # temperature 0 (greedy) para REDUCIR VARIANZA: decisiones deterministas dado el
            # prompt -> las diferencias entre versiones dejan de estar enmascaradas por el
            # muestreo del LLM (el 0.26 vs 0.25 resultó ser ruido, ver working notes).
            "max_tokens":800,"temperature":0.0,
            # Qwen3 es modelo de razonamiento: sin esto emite <think> largo y el JSON
            # de acciones se trunca. enable_thinking=False -> responde directo.
            "chat_template_kwargs":{"enable_thinking":False}}
    req = urllib.request.Request(f"{VLLM_BASE}/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"]

# SMOKE TEST: una llamada real, imprime respuesta cruda + acciones parseadas (barato,
# evita gastar la corrida entera a ciegas si el formato del modelo no encaja).
if ready:
    import numpy as np
    sys.path.insert(0, "/kaggle/working/src")
    from arc3.llm_prompt import SYSTEM_PROMPT, build_user_text, parse_actions
    g = np.zeros((64,64), dtype=np.int8); g[10:14,20:24]=2; g[40,40]=5
    smoke_user = build_user_text(g, None, [1,2,3,4,6], 0)
    try:
        raw = chat_fn(SYSTEM_PROMPT, smoke_user, None)
        print("SMOKE raw (first 500):", repr(raw[:500]), flush=True)
        print("SMOKE parsed actions:", parse_actions(raw), flush=True)
    except Exception as e:
        print("SMOKE error:", e, flush=True)
'''

RUN = '''\
sys.path.insert(0, "/kaggle/working/src")
from arc_agi.base import Arcade, OperationMode
from arc3.runner import run_games, play_game
from arc3.llm_agent import HybridAgent
from arc3.agent import GraphExplorer
import arc3.runner as runner_mod

# Híbrido: el explorador bankea niveles fáciles (piso ~0.25) y delega en el LLM cuando se
# atasca (techo semántico). Si vLLM no arrancó, explorador puro. Ver docs/STRATEGY.md.
USE_IMAGE = os.environ.get("ARC_USE_IMAGE","0") == "1"
def make_agent(game_id, max_actions):
    if ready:
        return HybridAgent(game_id, chat_fn, max_actions=max_actions, use_image=USE_IMAGE)
    return GraphExplorer(game_id, max_actions=max_actions)
runner_mod._AGENT_FACTORY = make_agent  # play_game lo usa si existe

if TRUE_SUBMISSION:
    base = os.environ.get("ARC_BASE_URL","http://gateway:8001/")
    dl=time.monotonic()+600
    while time.monotonic()<dl:
        try:
            with urlopen(base+"api/games", timeout=10) as r:
                if r.status<500: break
        except Exception: pass
        time.sleep(5)
    arcade = Arcade(arc_api_key=os.environ.get("ARC_API_KEY","test-key-123"),
                    arc_base_url=base, operation_mode=OperationMode.COMPETITION, environments_dir="")
    BUDGET = 8*3600-900 - (time.time()-NOTEBOOK_START); WORKERS=8; MAXG=3600.0
else:
    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=str(COMP_ROOT/"environment_files"))
    # vLLM agrupa (continuous batching) los requests concurrentes: subir workers eleva el
    # throughput del LLM en la MISMA GPU. Clave para que el LLM haga suficientes acciones.
    BUDGET = OFFLINE_SOFT_MIN*60; WORKERS=8; MAXG=600.0

game_ids=[e.game_id for e in arcade.available_environments]
try: card_id = arcade.open_scorecard(tags=["agent","llm"])
except Exception as e: print("scorecard:",e); card_id=None

import pandas as pd
pd.DataFrame([["1_0","1",True,1]],columns=["row_id","game_id","end_of_game","score"]).to_parquet("/kaggle/working/submission.parquet",index=False)

results = run_games(arcade, game_ids, total_budget_s=BUDGET, workers=WORKERS,
                    max_actions=15000, max_game_s=MAXG, card_id=card_id)

# CSV sin los campos voluminosos; diag/memory se imprimen abajo.
slim = [{k: v for k, v in r.items() if k not in ("diag", "memory")} for r in results]
pd.DataFrame(slim).to_csv("/kaggle/working/results.csv", index=False)
print("TOTAL niveles:", sum(r["levels"] for r in results))

# ---- DIAGNOSTICO: que genera el LLM (para iterar via Save & Run sin gastar submission) ----
print("\\n" + "="*70 + "\\nDIAGNOSTICO LLM (primeros juegos con actividad)\\n" + "="*70)
shown = 0
for r in sorted(results, key=lambda x: -x.get("llm_calls", 0)):
    d = r.get("diag")
    if not d or shown >= 3:
        continue
    shown += 1
    print(f"\\n### {r['game_id']}  niveles={r['levels']} acciones={r['actions']} "
          f"llm_calls={r['llm_calls']} llm_fails={r['llm_fails']}")
    print(f"  fallos: exception={d['fail_exception']} parse_empty={d['fail_parse_empty']} "
          f"no_legal={d['fail_no_legal']}")
    for j, s in enumerate(d.get("samples", [])[:3]):
        print(f"  --- sample {j}: PROMPT ---\\n  {s[0][:280]}")
        print(f"  --- RAW REPLY ---\\n  {repr(s[1])[:380]}")
        print(f"  --- PARSED ---  {s[2]}")
    for j, m in enumerate(d.get("reflections", [])[:2]):
        print(f"  --- REFLECTION {j} ---\\n  {m[:380]}")
    if r.get("memory"):
        print(f"  --- MEMORIA FINAL ---\\n  {r['memory'][:500]}")
print("\\n(fin diagnostico)")
'''


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def main() -> int:
    files = {f"src/arc3/{p.name}": p.read_text(encoding="utf-8")
             for p in sorted(SRC.glob("*.py"))}
    writer = ["import os\n", "os.makedirs('/kaggle/working/src/arc3', exist_ok=True)\n",
              f"SOURCES = {json.dumps(files, indent=1)}\n",
              "for path, code in SOURCES.items():\n",
              "    open('/kaggle/working/'+path,'w',encoding='utf-8').write(code)\n",
              "print('src/arc3 reconstruido')\n"]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}},
          "cells": [code_cell(SETUP), code_cell("".join(writer)),
                    code_cell(VLLM), code_cell(CHAT), code_cell(RUN)]}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"generado {OUT} con {len(files)} fuentes embebidas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

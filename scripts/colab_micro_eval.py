#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Evalua ESTRATEGIAS DE PROMPT en el banco micro, con un modelo pequeno.

POR QUE ESTE EXPERIMENTO EXISTE. Nuestro unico instrumento fiable era el envio diario:
un dato por noche, varianza 0.41, 3-4 dias por decision. Cuatro experimentos despues
seguiamos sin norte. Esto mide **cientos de items por minuto** sobre lo que de verdad
nos falta (que el agente entienda la mecanica), con respuestas verificables.

DOS PREGUNTAS, CADA UNA CON SU COMPARACION:

  A. effect_of_action -- "visto esto, que hace ACTION_N?"   mide INFERENCIA
       V0 crudo        : solo los recortes ASCII antes/despues
       V1 +objetos     : ademas la lista de objetos (nuestras features de src/arc3)
     -> responde por fin la tesis de Fase 3: las features objetuales, ayudan?

  B. plan_action -- "el objeto esta en P, la meta en T, que accion acerca?"  mide PLANIFICACION
       V0 sin tabla    : solo posiciones (el modelo no sabe que hace cada accion)
       V2 +tabla       : ademas el modelo de movimiento MEDIDO (la carga del seam C)
     -> responde si inyectar efectos medidos basta para que planifique bien

Modelo: el mas pequeno de la gama (Qwen3-1.7B). Los NIVELES absolutos de un modelo asi
no transfieren, pero estas preguntas no son de nivel: son de comprension mecanica, y ahi
la senal relativa entre variantes si informa el diseno del prompt del modelo grande.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_micro_eval.py
Salida: tabla de aciertos por variante y tipo, en stdout y en /content/micro_eval_result.json
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BENCH_URL = ("https://raw.githubusercontent.com/jvilladuque90/"
             "arc-prize-2026-arc-agi-3/main/micro_bench.jsonl")
MODEL = "Qwen/Qwen3-1.7B"
MAX_ITEMS = int(os.environ.get("MICRO_MAX_ITEMS", "0"))  # 0 = todos

t0 = time.time()
def log(m):
    print(f"[micro {time.time()-t0:6.0f}s] {m}", flush=True)

import threading as _th
def _hb():
    while True:
        time.sleep(60)
        print(f"[hb {time.time()-t0:6.0f}s]", flush=True)
_th.Thread(target=_hb, daemon=True).start()

log("pip install vllm ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchaudio"], check=False)
os.environ.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                   "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1",
                   "VLLM_ATTENTION_BACKEND": "XFORMERS"})

log("descargando el banco desde el repo publico ...")
with urllib.request.urlopen(BENCH_URL, timeout=60) as r:
    raw = r.read().decode("utf-8")
ITEMS = [json.loads(l) for l in raw.splitlines() if l.strip()]
if MAX_ITEMS:
    ITEMS = ITEMS[:MAX_ITEMS]
log(f"{len(ITEMS)} items")

log(f"lanzando vLLM con {MODEL} ...")
vllm_log = open("/content/vllm.log", "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
     "--model", MODEL, "--port", "1234", "--dtype", "half",
     "--max-model-len", "8192", "--gpu-memory-utilization", "0.90"],
    stdout=vllm_log, stderr=subprocess.STDOUT)
dl = time.time() + 1500
while time.time() < dl:
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    if proc.poll() is not None:
        print(open("/content/vllm.log").read()[-3000:])
        raise RuntimeError("vLLM murio al arrancar")
    time.sleep(10)
else:
    raise RuntimeError("vLLM no respondio")
log("vLLM listo")

LEGEND = ("Colores: W=blanco w=gris-claro g=gris G=gris-oscuro c=carbon B=negro M=magenta "
          "P=rosa R=rojo b=azul S=celeste Y=amarillo O=naranja r=rojo-oscuro N=verde p=morado")


def prompt_effect(item, with_objects):
    p = ["Estas analizando un juego de rejilla. Cada ejemplo muestra el tablero ANTES y "
         "DESPUES de ejecutar la misma accion.", LEGEND, ""]
    for i, s in enumerate(item["shots"], 1):
        p += [f"Ejemplo {i} — ANTES:", s["before"], f"Ejemplo {i} — DESPUES:", s["after"], ""]
    if with_objects and item.get("objects"):
        objs = ", ".join(f"color {o['color']} tam {o['size']} centro {o['center']}"
                         for o in item["objects"][:6])
        p += [f"Objetos detectados en el tablero: {objs}", ""]
    p += [f"Que hace la accion {item['action']}?",
          "Responde EXACTAMENTE una de estas formas, sin explicar:",
          "  none                (no cambia nada)",
          "  change              (cambia el tablero sin trasladar un objeto)",
          "  move DR DC          (traslada un objeto DR filas y DC columnas; ej: move -1 0)",
          "Respuesta:"]
    return "\n".join(p)


def prompt_which(item, with_objects):
    obs = "\n".join(f"  {s['action']} desplaza {s['moved']}" for s in item["shots"] if s.get("moved"))
    p = ["Observaciones medidas en este juego (fila, columna):", obs, "",
         f"Que accion mueve el objeto hacia {item['direction']}?",
         "Responde solo el nombre de la accion (ej: ACTION1). Respuesta:"]
    return "\n".join(p)


def prompt_plan(item, with_table):
    p = ["Un objeto debe llegar a una casilla objetivo en una rejilla.",
         f"Posicion actual del objeto (fila, columna): {item['player']}",
         f"Posicion objetivo (fila, columna): {item['target']}", ""]
    if with_table:
        tab = "\n".join(f"  {a}: {v}" for a, v in item["effects_table"].items())
        p += ["Efecto MEDIDO de cada accion (filas, columnas):", tab, ""]
    else:
        p += ["Acciones disponibles: " + ", ".join(item["effects_table"].keys()), ""]
    p += ["Que accion acerca mas el objeto al objetivo?",
          "Responde solo el nombre de la accion (ej: ACTION1). Respuesta:"]
    return "\n".join(p)


def ask(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 24,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def normalize(text, kind):
    t = (text or "").strip().lower().replace("**", "")
    if kind in ("which_action", "plan_action"):
        m = re.search(r"action\s*([1-7])", t)
        return f"ACTION{m.group(1)}" if m else t[:20]
    if "none" in t:
        return "none"
    m = re.search(r"move\s*(-?\d+)\s*[, ]\s*(-?\d+)", t)
    if m:
        return f"move {int(m.group(1))} {int(m.group(2))}"
    if "change" in t:
        return "change"
    return t[:20]


VARIANTS = [
    ("A.V0_crudo", "effect_of_action", lambda it: prompt_effect(it, False)),
    ("A.V1_objetos", "effect_of_action", lambda it: prompt_effect(it, True)),
    ("B.V0_sin_tabla", "plan_action", lambda it: prompt_plan(it, False)),
    ("B.V2_con_tabla", "plan_action", lambda it: prompt_plan(it, True)),
    ("C.lookup", "which_action", lambda it: prompt_which(it, False)),
]

result = {"model": MODEL, "n_items": len(ITEMS), "variants": {}}
for name, kind, builder in VARIANTS:
    subset = [i for i in ITEMS if i["type"] == kind]
    hits = 0
    examples = []
    for it in subset:
        try:
            raw_ans = ask(builder(it))
        except Exception as exc:
            raw_ans = f"<error {type(exc).__name__}>"
        got = normalize(raw_ans, kind)
        ok = got == it["answer"]
        hits += ok
        if len(examples) < 3:
            examples.append({"game": it["game"], "esperado": it["answer"],
                             "obtenido": got, "crudo": (raw_ans or "")[:60]})
    acc = hits / len(subset) if subset else 0.0
    result["variants"][name] = {"n": len(subset), "aciertos": hits,
                                "precision": round(acc, 3), "ejemplos": examples}
    log(f"{name:18} {hits:3}/{len(subset):3} = {acc:.1%}")

proc.terminate()
open("/content/micro_eval_result.json", "w").write(json.dumps(result, indent=2, ensure_ascii=False))
print("\n===== MICRO EVAL RESULT =====")
print(json.dumps(result, indent=2, ensure_ascii=False)[:5000])
log("done")

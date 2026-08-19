#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Evalua ESTRATEGIAS DE PROMPT en el banco micro, con modelos pequenos.

POR QUE ESTE EXPERIMENTO EXISTE. Nuestro unico instrumento fiable era el envio diario:
un dato por noche, varianza 0.41, 3-4 dias por decision. Cuatro experimentos despues
seguiamos sin norte. Esto mide **cientos de items por minuto** sobre lo que de verdad
nos falta (que el agente entienda la mecanica), con respuestas verificables.

DOS PREGUNTAS, CADA UNA CON SU COMPARACION PAREADA (mismos items en ambos brazos):

  A. effect_of_action -- "visto esto, que hace ACTION_N?"   mide INFERENCIA
       V0 crudo        : solo los recortes ASCII antes/despues
       V1 +objetos     : ademas la lista de objetos (nuestras features de src/arc3)
     -> responde por fin la tesis de Fase 3: las features objetuales, ayudan?

  B. plan_action -- "el objeto esta en P, la meta en T, que accion acerca?"  mide PLANIFICACION
       V0 sin tabla    : solo posiciones (el modelo no sabe que hace cada accion)
       V2 +tabla       : ademas el modelo de movimiento MEDIDO (la carga del seam C)
     -> responde si inyectar efectos medidos basta para que planifique bien

TRES SALVAGUARDAS contra celebrar ruido:
  1. LINEA BASE TRIVIAL: la clase mayoritaria de cada tipo. Una variante que no la supere
     no ha aprendido nada, aunque su precision suene alta.
  2. COMPARACION PAREADA: se guarda el acierto por item, asi que el contraste entre brazos
     se hace sobre los DISCORDANTES (items que uno acierta y el otro no), no sobre dos
     porcentajes sueltos. Con n=125 eso es mucho mas sensible.
  3. DOS TAMANOS (0.6B y 1.7B): si el efecto aparece en ambos es estructural del prompt;
     si solo en uno, es ruido de ese modelo.

Los NIVELES absolutos de un modelo asi no transfieren al de 27B; el contraste RELATIVO
entre formatos de prompt si informa el diseno.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_micro_eval.py
Salida: tabla por variante en stdout + JSON completo al final (stdout, no fichero:
        la VM se destruye al terminar).
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
MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
BATCH = 16
MAX_NEW = 16

t0 = time.time()
def log(m):
    print(f"[micro {time.time()-t0:6.0f}s] {m}", flush=True)

import threading as _th
def _hb():
    while True:
        time.sleep(25)
        print(f"[hb {time.time()-t0:6.0f}s]", flush=True)
_th.Thread(target=_hb, daemon=True).start()

# transformers ya viene en Colab; solo aseguramos version con soporte Qwen3.
# NO instalamos vLLM: en la T4 gratis se lleva la RAM del host y mata el kernel.
log("preparando entorno ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=4.51", "accelerate"], check=False)
os.environ.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                   "TRANSFORMERS_NO_TORCHVISION": "1", "TOKENIZERS_PARALLELISM": "false"})

log("descargando el banco desde el repo publico ...")
with urllib.request.urlopen(BENCH_URL, timeout=60) as r:
    raw = r.read().decode("utf-8")
ITEMS = [json.loads(l) for l in raw.splitlines() if l.strip()]
log(f"{len(ITEMS)} items")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

log(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA'}")

LEGEND = ("Colores: W=blanco w=gris-claro g=gris G=gris-oscuro c=carbon B=negro M=magenta "
          "P=rosa R=rojo b=azul S=celeste Y=amarillo O=naranja r=rojo-oscuro N=verde p=morado")


def prompt_effect(item, with_objects):
    p = ["Estas analizando un juego de rejilla. Cada ejemplo muestra el tablero ANTES y "
         "DESPUES de ejecutar la misma accion.", LEGEND, ""]
    for i, s in enumerate(item["shots"], 1):
        p += [f"Ejemplo {i} - ANTES:", s["before"], f"Ejemplo {i} - DESPUES:", s["after"], ""]
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
    obs = "\n".join(f"  {s['action']} desplaza {s['moved']}"
                    for s in item["shots"] if s.get("moved"))
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
    ("A.V0_crudo",     "effect_of_action", lambda it: prompt_effect(it, False)),
    ("A.V1_objetos",   "effect_of_action", lambda it: prompt_effect(it, True)),
    ("B.V0_sin_tabla", "plan_action",      lambda it: prompt_plan(it, False)),
    ("B.V2_con_tabla", "plan_action",      lambda it: prompt_plan(it, True)),
    ("C.lookup",       "which_action",     lambda it: prompt_which(it, False)),
]

# --- salvaguarda 1: linea base trivial (responder siempre la clase mayoritaria)
BASELINES = {}
for kind in ("effect_of_action", "plan_action", "which_action"):
    sub = [i["answer"] for i in ITEMS if i["type"] == kind]
    if sub:
        freq = {}
        for a in sub:
            freq[a] = freq.get(a, 0) + 1
        top = max(freq, key=freq.get)
        BASELINES[kind] = {"clase": top, "precision": round(freq[top] / len(sub), 3),
                           "n": len(sub)}
log(f"linea base trivial (clase mayoritaria): {json.dumps(BASELINES, ensure_ascii=False)}")


def run_model(model_id):
    log(f"cargando {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    log(f"{model_id} listo")

    def generate(prompts):
        outs = []
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i:i + BATCH]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=4096).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True))
        return outs

    res = {}
    for name, kind, builder in VARIANTS:
        subset = [i for i in ITEMS if i["type"] == kind]
        if not subset:
            continue
        raws = generate([builder(it) for it in subset])
        # salvaguarda 2: acierto por item -> permite el contraste pareado despues
        per_item, hits, examples = [], 0, []
        for it, raw_ans in zip(subset, raws):
            got = normalize(raw_ans, kind)
            ok = got == it["answer"]
            hits += ok
            per_item.append(1 if ok else 0)
            if len(examples) < 3:
                examples.append({"game": it["game"], "esperado": it["answer"],
                                 "obtenido": got, "crudo": (raw_ans or "").strip()[:60]})
        acc = hits / len(subset)
        res[name] = {"n": len(subset), "aciertos": hits, "precision": round(acc, 3),
                     "por_item": per_item, "ejemplos": examples}
        base = BASELINES.get(kind, {}).get("precision", 0)
        flag = "" if acc > base else "  <- NO supera la linea base trivial"
        log(f"  {name:16} {hits:3}/{len(subset):3} = {acc:5.1%}  (base {base:.1%}){flag}")

    # salvaguarda 2: discordantes entre los brazos de cada comparacion
    pares = {}
    for etiqueta, a, b in (("A objetos", "A.V0_crudo", "A.V1_objetos"),
                           ("B tabla", "B.V0_sin_tabla", "B.V2_con_tabla")):
        if a in res and b in res:
            va, vb = res[a]["por_item"], res[b]["por_item"]
            solo_a = sum(1 for x, y in zip(va, vb) if x and not y)
            solo_b = sum(1 for x, y in zip(va, vb) if y and not x)
            pares[etiqueta] = {"solo_" + a: solo_a, "solo_" + b: solo_b,
                              "discordantes": solo_a + solo_b}
            log(f"  pareado {etiqueta}: {b} gana {solo_b}, {a} gana {solo_a} "
                f"(de {solo_a + solo_b} discordantes)")

    del model
    torch.cuda.empty_cache()
    return {"variantes": res, "pareado": pares}


result = {"n_items": len(ITEMS), "linea_base": BASELINES, "modelos": {}}
for mid in MODELS:
    try:
        result["modelos"][mid] = run_model(mid)
    except Exception as exc:
        log(f"{mid} FALLO: {type(exc).__name__}: {exc}")
        result["modelos"][mid] = {"error": f"{type(exc).__name__}: {exc}"}

print("\n===== MICRO EVAL RESULT =====", flush=True)
print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
log("done")

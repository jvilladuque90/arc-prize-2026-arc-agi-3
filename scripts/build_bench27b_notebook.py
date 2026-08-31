"""Genera un kernel que corre NUESTRO BANCO contra el 27B de produccion.

LA PREGUNTA QUE NUNCA HICIMOS. Todo el banco se midio con modelos de Colab
(0.6B a 8B). Del 27B que corre en produccion no tenemos **ni un dato**. Y la
aritmetica de throughput hace que la comparacion sea decisiva:

    modelo   tok/s aprox   acciones/juego (con thinking off)
    27B         195              ~234
    8B          658              ~791
    4B         1316            ~1.582

El 4B saca **90.9%** planificando en nuestro banco. Si el 27B saca algo parecido,
estamos pagando ~7x de computo por unos pocos puntos — y 7x mas acciones vale
mucho mas que esos puntos, porque el eje de cantidad es el unico que no hemos
agotado.

Si el 27B saca mucho mas (p.ej. 99%), la respuesta es la contraria y el tamano
esta justificado. En cualquier caso el dato ordena la decision.

DE DONDE SALE LA IDEA. El usuario propuso dos modelos pequenos votando. Esa forma
concreta tiene evidencia en contra (a 1.7B la informacion estructurada PERJUDICA,
pareado 15-5; y la votacion exige errores independientes que dos Qwen3 de la
misma familia no tienen). Pero la intuicion de fondo —cambiar tamano por
throughput— nunca se probo, y esto la prueba.

Reusa el arranque del notebook duck (mismos datasets y setup_commands, que ya
levantan vLLM con el 27B en el puerto 1234) y sustituye la fase de juego por el
banco.

Uso: python scripts/build_bench27b_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "bench27b.ipynb"

CELDA_BANCO = '''
# ---- NUESTRO BANCO contra el 27B servido por vLLM en localhost:1234 ----
# PETICIONES CONCURRENTES. El primer intento las hizo una a una y el kernel murio
# por tiempo: con un 27B y una sola peticion en vuelo, vLLM iba a ~1 tok/s (su log).
# El servidor agrupa por lotes solo si le llegan varias a la vez, que es justo como
# trabaja el harness real (28 juegos concurrentes).
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

RAW = "https://raw.githubusercontent.com/jvilladuque90/arc-prize-2026-arc-agi-3/main"
MODELO = "vrfai/Qwen3.6-27B-FP8"
BASE = "http://127.0.0.1:1234/v1"
CONCURRENCIA = 16
PRESUPUESTO_S = 45 * 60          # tope duro: reporta lo que haya

def bajar(p):
    with urllib.request.urlopen(f"{RAW}/{p}?cb={int(time.time())}", timeout=60) as r:
        return r.read().decode("utf-8")

ITEMS = [json.loads(l) for l in bajar("micro_bench.jsonl").splitlines() if l.strip()]
_mp = {}
exec(compile(bajar("scripts/micro_prompts.py"), "micro_prompts.py", "exec"), _mp)
prompt_plan_words, prompt_effect = _mp["prompt_plan_words"], _mp["prompt_effect"]
normalize, trivial_baselines = _mp["normalize"], _mp["trivial_baselines"]

PLAN = [i for i in ITEMS if i["type"] == "plan_action"]
EFECTO = [i for i in ITEMS if i["type"] == "effect_of_action"]
print(f"banco: {len(PLAN)} planificacion + {len(EFECTO)} efecto", flush=True)

dl = time.monotonic() + 900
while time.monotonic() < dl:
    try:
        with urllib.request.urlopen(f"{BASE}/models", timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    time.sleep(10)
print("vLLM listo", flush=True)

def preguntar(args):
    prompt, pensar, maxtok = args
    cuerpo = json.dumps({
        "model": MODELO,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": maxtok,
        "chat_template_kwargs": {"enable_thinking": bool(pensar)},
    }).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=cuerpo,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        return (d["choices"][0]["message"].get("content") or "",
                d.get("usage", {}).get("completion_tokens", 0))
    except Exception as exc:
        return (f"<error {type(exc).__name__}>", 0)

t0 = time.monotonic()
resultado = {"modelo": MODELO, "variantes": {}}

def medir(nombre, items, constructor, tipo, pensar, maxtok):
    if time.monotonic() - t0 > PRESUPUESTO_S:
        print(f"  {nombre}: saltado (presupuesto agotado)", flush=True)
        return
    tareas = [(constructor(it), pensar, maxtok) for it in items]
    with ThreadPoolExecutor(max_workers=CONCURRENCIA) as ex:
        salidas = list(ex.map(preguntar, tareas))
    aciertos = sum(1 for it, (txt, _) in zip(items, salidas)
                   if normalize(txt, tipo) == it["answer"])
    toks = [n for _, n in salidas]
    resultado["variantes"][nombre] = {
        "n": len(items), "aciertos": aciertos,
        "precision": round(aciertos / len(items), 3),
        "tokens_medios": round(sum(toks) / max(1, len(toks)), 1),
        "ejemplos": [{"esperado": it["answer"], "crudo": txt.strip()[:120]}
                     for it, (txt, _) in list(zip(items, salidas))[:2]]}
    r = resultado["variantes"][nombre]
    print(f"  {nombre:22} {aciertos:3}/{len(items):3} = {r['precision']:6.1%} | "
          f"{r['tokens_medios']:7.1f} tok | {time.monotonic()-t0:.0f}s", flush=True)

# orden por valor: primero lo que compara directo con el 4B (90.9% en planificacion)
medir("B.V3_plan_sin_think", PLAN, prompt_plan_words, "plan_action", False, 64)
medir("B.V3_plan_con_think", PLAN, prompt_plan_words, "plan_action", True, 512)
medir("A.V0_efecto_sin_think", EFECTO, lambda it: prompt_effect(it, False),
      "effect_of_action", False, 64)
medir("A.V0_efecto_con_think", EFECTO, lambda it: prompt_effect(it, False),
      "effect_of_action", True, 512)

print("\\n===== BENCH 27B =====")
print(json.dumps(resultado, indent=2, ensure_ascii=False)[:6000])
'''



def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_duck_notebook import CELLS  # mismas celdas de arranque

    # se conservan las 3 primeras (setup + wheelhouse + bundle/vLLM) y se
    # reemplaza la fase de juego por el banco
    celdas = list(CELLS[:3]) + [CELDA_BANCO]
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": c.splitlines(keepends=True)}
                    for c in celdas],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"generado {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

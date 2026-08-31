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
import json, re, time, urllib.request

RAW = "https://raw.githubusercontent.com/jvilladuque90/arc-prize-2026-arc-agi-3/main"
MODELO = "vrfai/Qwen3.6-27B-FP8"      # kaggle_served_model_name del solver

def bajar(p):
    with urllib.request.urlopen(f"{RAW}/{p}?cb={int(time.time())}", timeout=60) as r:
        return r.read().decode("utf-8")

ITEMS = [json.loads(l) for l in bajar("micro_bench.jsonl").splitlines() if l.strip()]
_mp = {}
exec(compile(bajar("scripts/micro_prompts.py"), "micro_prompts.py", "exec"), _mp)
VARIANTS, normalize = _mp["VARIANTS"], _mp["normalize"]
trivial_baselines, paired_contrast = _mp["trivial_baselines"], _mp["paired_contrast"]
print(f"banco: {len(ITEMS)} items", flush=True)

# esperar a que el servidor este vivo (el setup ya lo arranco)
BASE = "http://127.0.0.1:1234/v1"
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

def preguntar(prompt, thinking, max_tokens=64):
    cuerpo = json.dumps({
        "model": MODELO,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=cuerpo,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return m.get("content") or "", d.get("usage", {}).get("completion_tokens", 0)

BASES = trivial_baselines(ITEMS)
print("base trivial:", json.dumps(BASES, ensure_ascii=False), flush=True)

# Se mide con thinking OFF y ON: ademas de la precision del 27B, da su coste real
# en tokens sobre las MISMAS preguntas donde el 4B saco 90.9%.
resultado = {"modelo": MODELO, "variantes": {}}
for pensar in (False, True):
    for nombre, tipo, constructor in VARIANTS:
        if not nombre.startswith(("A.", "B.V3", "C.")):
            continue                      # los mas informativos, para no eternizar
        sub = [i for i in ITEMS if i["type"] == tipo]
        if not sub:
            continue
        aciertos, toks, ejem = 0, [], []
        for it in sub:
            try:
                txt, nt = preguntar(constructor(it), pensar,
                                    max_tokens=512 if pensar else 64)
            except Exception as exc:
                txt, nt = f"<error {type(exc).__name__}>", 0
            toks.append(nt)
            ok = normalize(txt, tipo) == it["answer"]
            aciertos += ok
            if len(ejem) < 2:
                ejem.append({"esperado": it["answer"], "crudo": txt.strip()[:120]})
        clave = f"{nombre}{'_think' if pensar else ''}"
        resultado["variantes"][clave] = {
            "n": len(sub), "aciertos": aciertos,
            "precision": round(aciertos / len(sub), 3),
            "tokens_medios": round(sum(toks) / max(1, len(toks)), 1),
            "ejemplos": ejem}
        print(f"  {clave:22} {aciertos:3}/{len(sub):3} = {aciertos/len(sub):6.1%} | "
              f"{resultado['variantes'][clave]['tokens_medios']:7.1f} tok", flush=True)

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

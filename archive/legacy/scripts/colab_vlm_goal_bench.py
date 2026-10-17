#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] ¿La VISION arregla la inferencia de meta? (hipotesis de salto grande)

LA FALTA QUE ATACA. La inferencia de meta esta en 31% (base trivial 25%) y el
diagnostico de §8.16 dice que el fallo es de INDEXACION: aplicar una pista
("la meta es la celda de color 5") exige contar filas y columnas en un ASCII de
64x64, que es lo que un LLM de texto peor hace. Un modelo con vision lee
posiciones nativamente. El #2 del milestone (0.86) era un VLM (Gemma-4-31B):
la arquitectura del competidor, no una especulacion nuestra.

DISENO — tres brazos, MISMO modelo (un VL acepta texto solo), mismos 82 items,
pareado por item:

    T  : tablero en ASCII + candidatas como coordenadas   (el regimen actual)
    V1 : tablero como IMAGEN + candidatas como coordenadas (¿la imagen ayuda?)
    V2 : imagen con candidatas MARCADAS con numeros        (¿y sin indexar nada?)

V2 es la forma fuerte de la hipotesis: el modelo elige "marcador 3" y el host
mapea de vuelta a la celda — cero indexacion textual en toda la cadena.

REGLA PRE-REGISTRADA: la hipotesis gana si V2 > T con p<0.05 en el pareado
agrupado Y V2 supera la base trivial (25% en firma, la mayor de los tipos) por
>=15 puntos. Si gana: el salto propuesto es mover produccion a un VLM (research
de cual cabe en la G4 + vLLM 0.19) — presupuesto Kaggle autorizado: 5h.

Uso: powershell -File scripts\\colab_cuenta.ps1 -Cuenta N -Script scripts\\colab_vlm_goal_bench.py
"""

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from math import comb

RAW = "https://raw.githubusercontent.com/jvilladuque90/arc-prize-2026-arc-agi-3/main"
MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
MODEL_FALLBACK = "Qwen/Qwen2.5-VL-3B-Instruct"
CELL = 8            # px por celda -> 512x512
MAX_NEW = 24
PRESUPUESTO_S = int(os.environ.get("VLM_BUDGET", str(80 * 60)))

t0 = time.time()


def log(m):
    print(f"[vlm {time.time()-t0:6.0f}s] {m}", flush=True)


def _hb():
    while True:
        time.sleep(25)
        print(f"[hb {time.time()-t0:6.0f}s]", flush=True)


threading.Thread(target=_hb, daemon=True).start()

log("preparando entorno ...")
# OJO: NO instalar/actualizar pillow — medio-actualizarlo en el runtime vivo de
# Colab rompe sus imports internos (ImportError: _Ink; corrida perdida 2026-09-06).
# El PIL preinstalado basta para rectangulos, elipses y texto.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=4.51", "accelerate"], check=False)
os.environ.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                   "TRANSFORMERS_NO_TORCHVISION": "1", "TOKENIZERS_PARALLELISM": "false"})


def fetch(path):
    with urllib.request.urlopen(f"{RAW}/{path}?cb={int(time.time())}", timeout=60) as r:
        return r.read().decode("utf-8")


ITEMS = [json.loads(l) for l in fetch("goal_bench.jsonl").splitlines() if l.strip()]
_mp = {}
exec(compile(fetch("scripts/micro_prompts.py"), "micro_prompts.py", "exec"), _mp)
grid_ascii, LEGEND = _mp["grid_ascii"], _mp["LEGEND"]
log(f"{len(ITEMS)} items de meta")

# paleta ARC (indices 0-15, mismo orden que GRID_LEGEND de micro_prompts)
PALETA = [(255, 255, 255), (200, 200, 200), (150, 150, 150), (100, 100, 100),
          (60, 60, 60), (0, 0, 0), (255, 0, 255), (255, 150, 200),
          (255, 0, 0), (0, 80, 255), (120, 200, 255), (255, 220, 0),
          (255, 140, 0), (150, 30, 30), (0, 180, 60), (140, 60, 200)]

from PIL import Image, ImageDraw  # noqa: E402


def render(board, marcar=None):
    """board -> PIL Image; marcar = lista de celdas [(r,c),...] con numeros 1..k."""
    h, w = len(board), len(board[0])
    img = Image.new("RGB", (w * CELL, h * CELL))
    dr = ImageDraw.Draw(img)
    for r, fila in enumerate(board):
        for c, v in enumerate(fila):
            dr.rectangle([c * CELL, r * CELL, (c + 1) * CELL - 1, (r + 1) * CELL - 1],
                         fill=PALETA[v % 16])
    if marcar:
        for i, (r, c) in enumerate(marcar, 1):
            x, y = c * CELL + CELL // 2, r * CELL + CELL // 2
            rad = CELL * 2
            dr.ellipse([x - rad, y - rad, x + rad, y + rad],
                       outline=(255, 0, 0), width=3)
            dr.text((x + rad + 2, y - rad), str(i), fill=(255, 0, 0))
    return img


def prompt_texto(it):
    cands = ", ".join(f"[{c[0]}, {c[1]}]" for c in it["candidates"])
    p = ["Estas jugando un juego de rejilla. Este es el tablero:", LEGEND, "",
         grid_ascii(it["board"]), ""]
    if it.get("trail"):
        p += [f"Trayecto observado: {it['trail']}", ""]
    if it.get("firma"):
        p += [it["firma"], ""]
    p += [f"Para completar el nivel hay que llegar a UNA de estas celdas — {it['goal_desc']}.",
          f"Candidatas (fila, columna): {cands}",
          "Cual es la celda objetivo? Responde solo la celda, fila y columna (ej: 20 33). Respuesta:"]
    return "\n".join(p)


def prompt_v1(it):
    cands = ", ".join(f"[{c[0]}, {c[1]}]" for c in it["candidates"])
    p = ["Estas jugando un juego de rejilla. La imagen muestra el tablero "
         f"({len(it['board'])}x{len(it['board'][0])} celdas)."]
    if it.get("trail"):
        p += [f"Trayecto observado: {it['trail']}"]
    if it.get("firma"):
        p += [it["firma"]]
    p += [f"Para completar el nivel hay que llegar a UNA de estas celdas — {it['goal_desc']}.",
          f"Candidatas (fila, columna): {cands}",
          "Cual es la celda objetivo? Responde solo la celda, fila y columna (ej: 20 33). Respuesta:"]
    return "\n".join(p)


def prompt_v2(it):
    k = len(it["candidates"])
    p = ["Estas jugando un juego de rejilla. La imagen muestra el tablero; las "
         f"celdas candidatas estan marcadas con circulos rojos numerados 1 a {k}."]
    if it.get("trail"):
        p += [f"Trayecto observado: {it['trail']}"]
    if it.get("firma"):
        p += [it["firma"]]
    p += [f"Para completar el nivel hay que llegar a UNA de las candidatas — {it['goal_desc']}.",
          f"Cual marcador es el objetivo? Responde solo el numero (1 a {k}). Respuesta:"]
    return "\n".join(p)


def norm_celda(txt):
    ms = re.findall(r"(\d+)\s*[,; ]\s*(\d+)", txt or "")
    return f"{int(ms[-1][0])} {int(ms[-1][1])}" if ms else (txt or "")[:20]


def norm_marcador(txt, it):
    ms = re.findall(r"\d+", txt or "")
    if not ms:
        return "?"
    i = int(ms[-1])
    if 1 <= i <= len(it["candidates"]):
        r, c = it["candidates"][i - 1]
        return f"{r} {c}"
    return "?"


import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

log(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA'}")


def cargar(mid):
    from transformers import AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(mid)
    mdl = AutoModelForImageTextToText.from_pretrained(
        mid, dtype=torch.float16, device_map="cuda")
    mdl.eval()
    return proc, mdl


try:
    log(f"cargando {MODEL} ...")
    proc, model = cargar(MODEL)
    usado = MODEL
except Exception as exc:
    log(f"{MODEL} fallo ({type(exc).__name__}); probando {MODEL_FALLBACK}")
    proc, model = cargar(MODEL_FALLBACK)
    usado = MODEL_FALLBACK
log(f"modelo en uso: {usado}")


def preguntar(prompt, imagen=None):
    contenido = ([{"type": "image", "image": imagen}] if imagen is not None else []) \
        + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": contenido}]
    kwargs = {}
    try:
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[txt], images=[imagen] if imagen is not None else None,
               return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
    out = gen[0][enc["input_ids"].shape[1]:]
    return proc.tokenizer.decode(out, skip_special_tokens=True)


def signo(a, b):
    sa = sum(1 for x, y in zip(a, b) if x and not y)
    sb = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = sa + sb, min(sa, sb)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)) if n else 1.0
    return {"solo_A": sa, "solo_B": sb, "discordantes": n, "p_signo": round(p, 4)}


res = {"modelo": usado, "n": len(ITEMS), "brazos": {}}
SALIDA = "/content/vlm_goal_bench.json"

for nombre, con_img, marcado, constructor, normalizador in (
        ("T_texto", False, False, prompt_texto, lambda o, it: norm_celda(o)),
        ("V1_imagen", True, False, prompt_v1, lambda o, it: norm_celda(o)),
        ("V2_marcadores", True, True, prompt_v2, norm_marcador)):
    vec, ejemplos = [], []
    log(f"brazo {nombre} ...")
    for i, it in enumerate(ITEMS):
        if time.time() - t0 > PRESUPUESTO_S:
            log(f"  presupuesto agotado en {i}/{len(ITEMS)}")
            break
        img = render(it["board"], it["candidates"] if marcado else None) if con_img else None
        out = preguntar(constructor(it), img)
        ok = 1 if normalizador(out, it) == it["answer"] else 0
        vec.append(ok)
        if len(ejemplos) < 2:
            ejemplos.append({"crudo": out.strip()[:80], "esperado": it["answer"]})
    res["brazos"][nombre] = {
        "n": len(vec), "aciertos": sum(vec),
        "precision": round(sum(vec) / max(1, len(vec)), 3),
        "por_item": vec, "ejemplos": ejemplos}
    log(f"  {nombre}: {sum(vec)}/{len(vec)} = {sum(vec)/max(1,len(vec)):.1%}")
    with open(SALIDA, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

if all(k in res["brazos"] for k in ("T_texto", "V1_imagen", "V2_marcadores")):
    n = min(len(res["brazos"][k]["por_item"]) for k in res["brazos"])
    vt, v1, v2 = (res["brazos"][k]["por_item"][:n]
                  for k in ("T_texto", "V1_imagen", "V2_marcadores"))
    res["pareados"] = {"V1_vs_T": signo(vt, v1), "V2_vs_T": signo(vt, v2)}
    s = res["pareados"]["V2_vs_T"]
    pv2 = res["brazos"]["V2_marcadores"]["precision"]
    gana = s["p_signo"] < 0.05 and s["solo_B"] > s["solo_A"] and pv2 >= 0.40
    res["veredicto"] = ("HIPOTESIS GANA: la vision arregla la indexacion -> "
                        "research de VLM para la G4" if gana else
                        "hipotesis NO confirmada en este modelo/tamano")
    for k, v in res["pareados"].items():
        log(f"pareado {k}: {v}")
    log(res["veredicto"])

with open(SALIDA, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)

print("\n===== VLM GOAL BENCH =====", flush=True)
comp = {k: {kk: vv for kk, vv in v.items() if kk != "por_item"}
        for k, v in res["brazos"].items()}
print(json.dumps({"modelo": usado, "brazos": comp,
                  "pareados": res.get("pareados"), "veredicto": res.get("veredicto")},
                 indent=2, ensure_ascii=False)[:5000], flush=True)
log("done")

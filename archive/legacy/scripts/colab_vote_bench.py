#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] ¿Se puede COMPRAR ACIERTO CON TOKENS? (auto-consistencia por voto)

LA PREGUNTA, Y POR QUE ES ESTA Y NO LA OBVIA.

La version ingenua —"muestrear dos veces y votar para recuperar el ruido del
muestreo"— ya esta MUERTA sin gastar GPU. El barrido 2x2 (DESIGN §8.25) midio:

    greedy sin thinking          90.9%   3.0 tokens
    produccion (T=0.6) sin think 89.9%   3.0 tokens     pareado 1-0, p=1.0

Con el pensamiento apagado el modelo emite ~3 tokens ("ACTION3"): la
distribucion esta clavada y el muestreo no introduce ruido. No hay nada que
recuperar votando. Cualquier voto sobre ese regimen mide cero por construccion.

LO QUE SI ESTA SIN PROBAR. Con el pensamiento ENCENDIDO el modelo genera ~420
tokens de razonamiento que SI divergen entre muestras. Ahi el voto no recupera
ruido: agrega caminos de razonamiento distintos. Es auto-consistencia clasica
(CoT + voto mayoritario), y la pregunta real es:

    ¿mayoria de 3 con pensamiento  >  1 muestra sin pensamiento (= v11)?

POR QUE LA EVIDENCIA LICENCIA JUSTO ESTE CAMBIO. Nuestro resultado mas repetido
es que por encima de un piso de ~18 acciones/juego las acciones y los niveles
estan DESACOPLADOS (STRATEGY §53-59: recortar contexto dio +48% acciones y 0.60,
peor). Produccion esta en ~94 acciones. O sea que el eje "mas acciones" esta
cerrado y el eje "mejor accion, aunque cueste mas tokens" es el unico que la
evidencia deja abierto — y nunca lo hemos empujado. Aritmetica de coste:

    v11 (sin think, 1 muestra)   856 tok/accion   ->  ~94 acciones/juego
    3x con think (voto)        ~4.326 tok/accion  ->  ~31 acciones/juego
    3x sin think (voto)        ~2.568 tok/accion  ->  ~52 acciones/juego

31 sigue por ENCIMA del piso de 18. El cambio es pagable; lo que falta es saber
si compra algo.

DISENO. Dos pasadas de generacion, cinco brazos derivados de ellas — asi el
pareado es exacto (los brazos 1x son la muestra 0 de la misma llamada que produce
el voto, que es un sorteo iid valido y no una corrida distinta):

    pasada A: thinking OFF, muestreo de produccion, 3 secuencias, 48 tokens
    pasada B: thinking ON,  muestreo de produccion, 3 secuencias, 512 tokens

    1. 1x_sin_think          = muestra 0 de A      <- LINEA BASE (config de v11)
    2. 3x_sin_think_voto     = mayoria de A        <- control: debe dar ~igual
    3. 1x_con_think          = muestra 0 de B
    4. 3x_con_think_voto     = mayoria de B        <- LA CANDIDATA
    5. adaptativo_con_think  = si las 2 primeras coinciden, esa; si no, mayoria
                               de 3 (coste esperado 2 + P(discrepan))

TAREAS ELEGIDAS POR TECHO, NO POR COSTUMBRE. El brazo estrella del banco
(B.V3_palabras, 90.9%) tiene solo 9 puntos de recorrido: un efecto de 3 puntos
son 3 items de 99 y no se lee. Se usan las dos variantes con margen real:

    B.V2_con_tabla (tabla vectorial)  99 items  66.1% -> 34 puntos de techo
    A.V0_crudo     (efecto de accion) 54 items  70.4% -> 30 puntos de techo

EL DIAGNOSTICO QUE DECIDE ANTES QUE EL ACIERTO: la tasa de ACUERDO entre las 3
muestras. Si con pensamiento las 3 coinciden el ~99% de las veces, el voto esta
muerto por la misma razon que sin pensamiento, y el numero de acierto sobra.

REGLA PRE-REGISTRADA (para no leer el resultado a conveniencia): solo pasa a
produccion si el brazo 4 gana al brazo 1 en el contraste PAREADO agrupado con
p<0.05. Dos porcentajes sueltos no cuentan — es la leccion que ya costo once
defectos de instrumento.

Uso: powershell -File scripts\colab_run.ps1 scripts\colab_vote_bench.py 7000
"""

import collections
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from math import comb

RAW = "https://raw.githubusercontent.com/jvilladuque90/arc-prize-2026-arc-agi-3/main"
MODEL = os.environ.get("VOTE_MODEL", "Qwen/Qwen3-4B")
K = int(os.environ.get("VOTE_K", "3"))            # muestras por item
BATCH = int(os.environ.get("VOTE_BATCH", "2"))    # x K secuencias en vuelo
MAX_THINK = int(os.environ.get("VOTE_MAX_THINK", "512"))
MAX_PLAIN = int(os.environ.get("VOTE_MAX_PLAIN", "48"))
PRESUPUESTO_S = int(os.environ.get("VOTE_BUDGET", str(80 * 60)))

t0 = time.time()


def log(m):
    print(f"[vote {time.time()-t0:6.0f}s] {m}", flush=True)


def _hb():
    while True:
        time.sleep(25)
        print(f"[hb {time.time()-t0:6.0f}s]", flush=True)


threading.Thread(target=_hb, daemon=True).start()

log("preparando entorno ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=4.51", "accelerate"], check=False)
os.environ.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                   "TRANSFORMERS_NO_TORCHVISION": "1", "TOKENIZERS_PARALLELISM": "false"})

log("descargando banco y prompts ...")


def fetch(path):
    with urllib.request.urlopen(f"{RAW}/{path}?cb={int(time.time())}", timeout=60) as r:
        return r.read().decode("utf-8")


ITEMS = [json.loads(l) for l in fetch("micro_bench.jsonl").splitlines() if l.strip()]
_mp = {}
exec(compile(fetch("scripts/micro_prompts.py"), "micro_prompts.py", "exec"), _mp)
prompt_plan, prompt_effect = _mp["prompt_plan"], _mp["prompt_effect"]
normalize = _mp["normalize"]

# (etiqueta, tipo del banco, constructor de prompt) — las dos con techo real
TAREAS = [
    ("B.V2_plan_tabla", "plan_action",
     [i for i in ITEMS if i["type"] == "plan_action"], lambda it: prompt_plan(it, True)),
    ("A.V0_efecto", "effect_of_action",
     [i for i in ITEMS if i["type"] == "effect_of_action"], lambda it: prompt_effect(it, False)),
]
for etq, _k, items, _c in TAREAS:
    log(f"{etq}: {len(items)} items")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

log(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA'}")
log(f"cargando {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
model.eval()

GEN = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)  # el de produccion


def muestrear(items, constructor, pensar, max_new):
    """Devuelve, por item, la lista de K salidas normalizadas + tokens gastados.

    num_return_sequences reparte K muestras del MISMO prefill: mucho mas barato
    que K pasadas, y las muestras siguen siendo iid entre si.
    """
    crudos, n_tokens = [], []
    for i in range(0, len(items), BATCH):
        if time.time() - t0 > PRESUPUESTO_S:
            log(f"  presupuesto agotado en el item {i}/{len(items)} — se reporta lo hecho")
            break
        lote = items[i:i + BATCH]
        textos = [tok.apply_chat_template(
            [{"role": "user", "content": constructor(it)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=pensar)
            for it in lote]
        enc = tok(textos, return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, num_return_sequences=K,
                                 pad_token_id=tok.pad_token_id, **GEN)
        largo = enc["input_ids"].shape[1]
        # orden de salida: [item0_s0, item0_s1, ..., item1_s0, ...]
        for j in range(len(lote)):
            salidas, toks = [], []
            for s in range(K):
                sec = gen[j * K + s][largo:]
                toks.append(int((sec != tok.pad_token_id).sum()))
                salidas.append(tok.decode(sec, skip_special_tokens=True))
            crudos.append(salidas)
            n_tokens.append(toks)
    return crudos, n_tokens


def voto(normalizadas):
    """Mayoria; empate -> la muestra 0 (fallback honesto, no el que mas convenga)."""
    c = collections.Counter(normalizadas)
    top, n = c.most_common(1)[0]
    if sum(1 for _v, m in c.items() if m == n) > 1:
        return normalizadas[0]
    return top


def adaptativo(normalizadas):
    """Si las 2 primeras coinciden, esa. Si no, mayoria de 3. Devuelve (resp, coste)."""
    if normalizadas[0] == normalizadas[1]:
        return normalizadas[0], 2
    return voto(normalizadas), 3


def signo(a_por_item, b_por_item):
    """Contraste pareado sobre los DISCORDANTES (misma logica que paired_contrast)."""
    solo_a = sum(1 for x, y in zip(a_por_item, b_por_item) if x and not y)
    solo_b = sum(1 for x, y in zip(a_por_item, b_por_item) if y and not x)
    n, k = solo_a + solo_b, min(solo_a, solo_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)) if n else 1.0
    return {"solo_A": solo_a, "solo_B": solo_b, "discordantes": n, "p_signo": round(p, 4)}


SALIDA = "/content/vote_bench.json"
res = {"modelo": MODEL, "K": K, "tareas": {}}

for etiqueta, kind, items, constructor in TAREAS:
    log(f"=== {etiqueta} ({len(items)} items) ===")
    bloque = {"n_pedidos": len(items), "brazos": {}, "diagnostico": {}}

    for nombre_pasada, pensar, max_new in (("sin_think", False, MAX_PLAIN),
                                           ("con_think", True, MAX_THINK)):
        log(f"  pasada {nombre_pasada} (K={K}, max_new={max_new}) ...")
        crudos, n_tokens = muestrear(items, constructor, pensar, max_new)
        usados = items[:len(crudos)]
        norm = [[normalize(t, kind) for t in fila] for fila in crudos]

        acuerdo_total = sum(1 for f in norm if len(set(f)) == 1)
        acuerdo_2 = sum(1 for f in norm if f[0] == f[1])
        tok_medio = sum(sum(f) for f in n_tokens) / max(1, len(n_tokens))

        una = [1 if f[0] == it["answer"] else 0 for f, it in zip(norm, usados)]
        votada = [1 if voto(f) == it["answer"] else 0 for f, it in zip(norm, usados)]
        ad = [adaptativo(f) for f in norm]
        adapt = [1 if r == it["answer"] else 0 for (r, _c), it in zip(ad, usados)]
        coste_adapt = sum(c for _r, c in ad) / max(1, len(ad))

        for suf, vec, factor in ((f"1x_{nombre_pasada}", una, 1.0),
                                 (f"{K}x_{nombre_pasada}_voto", votada, float(K)),
                                 (f"adapt_{nombre_pasada}", adapt, coste_adapt)):
            bloque["brazos"][suf] = {
                "n": len(vec), "aciertos": sum(vec),
                "precision": round(sum(vec) / max(1, len(vec)), 3),
                "tokens_por_decision": round(tok_medio / K * factor, 1),
                "por_item": vec}
        bloque["diagnostico"][nombre_pasada] = {
            "acuerdo_3_de_3": round(acuerdo_total / max(1, len(norm)), 3),
            "acuerdo_2_primeras": round(acuerdo_2 / max(1, len(norm)), 3),
            "tokens_por_muestra": round(tok_medio / K, 1),
            "coste_adaptativo_muestras": round(coste_adapt, 2),
            "ejemplo": [t.strip()[:300] for t in crudos[0]] if crudos else []}

        for suf in (f"1x_{nombre_pasada}", f"{K}x_{nombre_pasada}_voto",
                    f"adapt_{nombre_pasada}"):
            b = bloque["brazos"][suf]
            log(f"    {suf:24} {b['aciertos']:3}/{b['n']:3} = {b['precision']:6.1%} | "
                f"{b['tokens_por_decision']:7.1f} tok/decision")
        d = bloque["diagnostico"][nombre_pasada]
        log(f"    acuerdo 3/3 = {d['acuerdo_3_de_3']:.1%} | "
            f"2 primeras = {d['acuerdo_2_primeras']:.1%}")

        res["tareas"][etiqueta] = bloque
        with open(SALIDA, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)

    br = bloque["brazos"]
    if f"{K}x_con_think_voto" in br and "1x_sin_think" in br:
        n = min(len(br["1x_sin_think"]["por_item"]),
                len(br[f"{K}x_con_think_voto"]["por_item"]))
        bloque["pareados"] = {
            "voto_con_think vs 1x_sin_think":
                signo(br["1x_sin_think"]["por_item"][:n],
                      br[f"{K}x_con_think_voto"]["por_item"][:n]),
            "voto_sin_think vs 1x_sin_think":
                signo(br["1x_sin_think"]["por_item"], br[f"{K}x_sin_think_voto"]["por_item"]),
            "voto_con_think vs 1x_con_think":
                signo(br["1x_con_think"]["por_item"], br[f"{K}x_con_think_voto"]["por_item"])}
        for k, v in bloque["pareados"].items():
            log(f"  pareado {k}: A={v['solo_A']} B={v['solo_B']} "
                f"(n={v['discordantes']}, p={v['p_signo']})")

    res["tareas"][etiqueta] = bloque
    with open(SALIDA, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

# --- contraste AGRUPADO: es el que decide segun la regla pre-registrada ---
pool_base, pool_voto = [], []
for bloque in res["tareas"].values():
    br = bloque["brazos"]
    if "1x_sin_think" in br and f"{K}x_con_think_voto" in br:
        n = min(len(br["1x_sin_think"]["por_item"]), len(br[f"{K}x_con_think_voto"]["por_item"]))
        pool_base += br["1x_sin_think"]["por_item"][:n]
        pool_voto += br[f"{K}x_con_think_voto"]["por_item"][:n]
if pool_base:
    res["agrupado"] = {
        "n": len(pool_base),
        "base_1x_sin_think": round(sum(pool_base) / len(pool_base), 3),
        "voto_3x_con_think": round(sum(pool_voto) / len(pool_voto), 3),
        "pareado": signo(pool_base, pool_voto)}
    a = res["agrupado"]
    log(f"AGRUPADO n={a['n']}: base {a['base_1x_sin_think']:.1%} vs "
        f"voto {a['voto_3x_con_think']:.1%} | {a['pareado']}")
    # OJO: no encadenar `and p` aqui — p=0.0 es el caso MAS significativo y un
    # test de verdad lo tumbaria por falsy. Es el tipo de fallo que ya nos dio
    # dos condiciones con el mismo numero.
    p = a["pareado"]["p_signo"]
    gana = p < 0.05 and a["pareado"]["solo_B"] > a["pareado"]["solo_A"]
    res["veredicto"] = ("PASA a produccion (regla pre-registrada)" if gana
                        else "NO pasa: el voto no compra acierto de forma significativa")
    log(res["veredicto"])

with open(SALIDA, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)

print("\n===== VOTE BENCH =====", flush=True)
_compacto = {k: {"brazos": {n: {kk: vv for kk, vv in b.items() if kk != "por_item"}
                            for n, b in v["brazos"].items()},
                 "diagnostico": v["diagnostico"], "pareados": v.get("pareados")}
             for k, v in res["tareas"].items()}
print(json.dumps({"modelo": MODEL, "tareas": _compacto,
                  "agrupado": res.get("agrupado"),
                  "veredicto": res.get("veredicto")},
                 indent=2, ensure_ascii=False)[:7000], flush=True)
log("done")

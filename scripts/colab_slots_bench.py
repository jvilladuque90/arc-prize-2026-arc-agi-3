#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] BANCO DE RANURAS: ¿el modelo de mundo etiquetado sobrevive al desalojo?

LA PREGUNTA. El harness desaloja el historial (32k, sin resumen) pero persiste 7
ranuras etiquetadas que se reinyectan cada turno. Medido en produccion: el modelo
solo llena 2/7 porque el prompt las declara "optional". El parche --slots las hace
obligatorias (DESIGN 8.27). Antes de gastar G4 validandolo, esto mide el MECANISMO:

    si el modelo escribe las 7 ranuras tras ver la informacion, y luego pierde el
    historial, ¿puede decidir bien usando SOLO las ranuras?

DISENO (dos fases, tres brazos, mismos items — pareado exacto):

  fase 1 (aprender): el modelo ve la tabla de efectos MEDIDA de un juego (la
    informacion que en produccion saldria de sus propias observaciones) y se le
    pide su modelo de mundo:
      brazo M (mandatorio): formato REQUERIDO de 7 lineas (el texto del parche)
      brazo O (opcional):   la frase original del harness ("helpful optional...")
  eviccion: se tira TODO menos las lineas etiquetadas que el modelo escribio
    (parseo como el harness: lineas que empiezan por la etiqueta).
  fase 2 (decidir): posicion + objetivo + SOLO las ranuras. ¿Que accion acerca?
      brazo A (algoritmico): ranuras rellenadas por el anfitrion desde la tabla
        (cota superior: si A falla, el mecanismo no da mas y el parche no vale)

METRICAS pre-registradas:
  primaria: acierto fase 2, pareado M vs O (test de signo). El parche merece G4
    si M > O con p<0.05 Y M se acerca a A (>= 80% de A).
  diagnosticas: tasa de llenado por ranura en fase 1 (M deberia ~7/7, O ~2/7 como
    produccion); tokens por fase.
  sanity: los prompts de fase 2 de M y O deben DIFERIR en la mayoria de items
    (leccion de los dos-numeros-iguales); si no difieren, el banco no midio nada.

Uso: powershell -File scripts\\colab_run.ps1 scripts\\colab_slots_bench.py 7000
"""

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
MODEL = os.environ.get("SLOTS_MODEL", "Qwen/Qwen3-4B")
BATCH = int(os.environ.get("SLOTS_BATCH", "8"))
MAX_F1 = int(os.environ.get("SLOTS_MAX_F1", "256"))
MAX_F2 = int(os.environ.get("SLOTS_MAX_F2", "48"))
PRESUPUESTO_S = int(os.environ.get("SLOTS_BUDGET", str(75 * 60)))

t0 = time.time()


def log(m):
    print(f"[slots {time.time()-t0:6.0f}s] {m}", flush=True)


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


def fetch(path):
    with urllib.request.urlopen(f"{RAW}/{path}?cb={int(time.time())}", timeout=60) as r:
        return r.read().decode("utf-8")


ITEMS = [json.loads(l) for l in fetch("micro_bench.jsonl").splitlines() if l.strip()]
_mp = {}
exec(compile(fetch("scripts/micro_prompts.py"), "micro_prompts.py", "exec"), _mp)
normalize, dir_words = _mp["normalize"], _mp["dir_words"]
_parse_shift = _mp["_parse_shift"]

PLAN = [i for i in ITEMS if i["type"] == "plan_action"]
log(f"{len(PLAN)} items de planificacion")

ETIQUETAS = ["World model", "Goal model", "Action model", "Recent findings",
             "Open questions", "Plan", "Cross-level notes"]

MANDATO = ("REQUIRED FORMAT: write your world model as EXACTLY these seven labeled "
           "lines, one per line, in this order: `World model:`, `Goal model:`, "
           "`Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, "
           "`Cross-level notes:`. Update every line from the evidence; write "
           "`unknown` where truly unknown; never omit a label. `Action model:` "
           "lists each valid action and its observed effect.")
OPCIONAL = ("If you write notes, keep them short. Helpful optional prefixes are "
            "`World model:`, `Goal model:`, `Action model:`, `Recent findings:`, "
            "`Open questions:`, `Plan:`, and `Cross-level notes:`.")


def tabla(item):
    return "\n".join(f"  {a}: moves {dir_words(*_parse_shift(v))}"
                     for a, v in item["effects_table"].items())


def prompt_f1(item, mandato: bool) -> str:
    return "\n".join([
        "You are a coding agent solving a grid-based puzzle game.",
        "You have been probing the controls. MEASURED effect of each action:",
        tabla(item), "",
        "Your message history will be evicted; only your labeled note lines survive.",
        MANDATO if mandato else OPCIONAL, "",
        "Write your notes now:"])


def parse_ranuras(texto: str) -> dict:
    """Como el harness: la linea que empieza por `Etiqueta:` llena la ranura."""
    out = {}
    for ln in (texto or "").splitlines():
        ln = ln.strip().lstrip("*").strip()
        for e in ETIQUETAS:
            m = re.match(rf"^`?{re.escape(e)}`?\s*:\s*(.+)$", ln, re.IGNORECASE)
            if m and e not in out:
                out[e] = m.group(1).strip()
    return out


def ranuras_algoritmicas(item) -> dict:
    return {"World model": "grid puzzle; an object must reach a target cell",
            "Action model": "; ".join(
                f"{a} moves {dir_words(*_parse_shift(v))}"
                for a, v in item["effects_table"].items()),
            "Plan": "move toward the target with the action that reduces distance most"}


def prompt_f2(item, ranuras: dict) -> str:
    notas = "\n".join(f"  {e}: {v}" for e, v in ranuras.items() if v) or "  (no notes)"
    return "\n".join([
        "You are a coding agent solving a grid-based puzzle game.",
        "Your history was evicted. Your surviving notes from earlier turns:",
        notas, "",
        f"Current object position (row, column): {item['player']}",
        f"Target position (row, column): {item['target']}",
        "Which action brings the object closest to the target?",
        "Answer with the action name only (e.g. ACTION1). Answer:"])


import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

log(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA'}")
log(f"cargando {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
model.eval()


def generar(prompts, max_new):
    outs, toks = [], []
    for i in range(0, len(prompts), BATCH):
        if time.time() - t0 > PRESUPUESTO_S:
            log(f"  presupuesto agotado en {i}/{len(prompts)}")
            break
        lote = prompts[i:i + BATCH]
        textos = [tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False) for p in lote]
        enc = tok(textos, return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(lote)):
            sec = gen[j][enc["input_ids"].shape[1]:]
            toks.append(int((sec != tok.pad_token_id).sum()))
            outs.append(tok.decode(sec, skip_special_tokens=True))
    return outs, toks


def signo(a, b):
    sa = sum(1 for x, y in zip(a, b) if x and not y)
    sb = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = sa + sb, min(sa, sb)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)) if n else 1.0
    return {"solo_A": sa, "solo_B": sb, "discordantes": n, "p_signo": round(p, 4)}


res = {"modelo": MODEL, "n": len(PLAN), "brazos": {}, "diagnostico": {}}
SALIDA = "/content/slots_bench.json"

# ---- fase 1 en los dos brazos con modelo ----
notas = {}
for brazo, mandato in (("M", True), ("O", False)):
    log(f"fase 1 brazo {brazo} ...")
    outs, toks = generar([prompt_f1(it, mandato) for it in PLAN], MAX_F1)
    filas = [parse_ranuras(o) for o in outs]
    notas[brazo] = filas
    llenado = {e: sum(1 for f in filas if f.get(e)) for e in ETIQUETAS}
    res["diagnostico"][f"llenado_{brazo}"] = llenado
    res["diagnostico"][f"tokens_f1_{brazo}"] = round(sum(toks) / max(1, len(toks)), 1)
    log(f"  llenado {brazo}: " + " ".join(f"{e.split()[0]}={llenado[e]}" for e in ETIQUETAS))

n_usable = min(len(notas["M"]), len(notas["O"]))
PLAN = PLAN[:n_usable]

# sanity: los prompts de fase 2 deben diferir entre brazos
iguales = sum(1 for i, it in enumerate(PLAN)
              if prompt_f2(it, notas["M"][i]) == prompt_f2(it, notas["O"][i]))
res["diagnostico"]["f2_prompts_identicos_M_vs_O"] = iguales
log(f"sanity: prompts f2 identicos M vs O: {iguales}/{len(PLAN)} "
    + ("<-- SOSPECHOSO, el banco no midio nada" if iguales > len(PLAN) * 0.5 else "(ok)"))

# ---- fase 2: tres brazos ----
for brazo, fuente in (("M", lambda i, it: notas["M"][i]),
                      ("O", lambda i, it: notas["O"][i]),
                      ("A", lambda i, it: ranuras_algoritmicas(it))):
    log(f"fase 2 brazo {brazo} ...")
    outs, toks = generar([prompt_f2(it, fuente(i, it)) for i, it in enumerate(PLAN)], MAX_F2)
    vec = [1 if normalize(o, "plan_action") == it["answer"] else 0
           for o, it in zip(outs, PLAN)]
    res["brazos"][brazo] = {
        "n": len(vec), "aciertos": sum(vec),
        "precision": round(sum(vec) / max(1, len(vec)), 3),
        "tokens_f2": round(sum(toks) / max(1, len(toks)), 1),
        "por_item": vec,
        "ejemplos": [o.strip()[:80] for o in outs[:2]]}
    b = res["brazos"][brazo]
    log(f"  {brazo}: {b['aciertos']}/{b['n']} = {b['precision']:.1%}")
    with open(SALIDA, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

if "M" in res["brazos"] and "O" in res["brazos"]:
    n = min(res["brazos"]["M"]["n"], res["brazos"]["O"]["n"], res["brazos"]["A"]["n"])
    pm, po, pa = (res["brazos"][x]["por_item"][:n] for x in ("M", "O", "A"))
    res["pareados"] = {"M_vs_O": signo(po, pm), "M_vs_A": signo(pm, pa)}
    pM, pA = res["brazos"]["M"]["precision"], res["brazos"]["A"]["precision"]
    s = res["pareados"]["M_vs_O"]
    gana = s["p_signo"] < 0.05 and s["solo_B"] > s["solo_A"] and pA > 0 and pM >= 0.8 * pA
    res["veredicto"] = ("MERECE G4: mandatorio > opcional (p<0.05) y cerca de la cota"
                       if gana else
                       "NO merece G4 aun: el mecanismo no preserva lo suficiente")
    log(f"pareado M vs O: {s} | M={pM:.1%} A={pA:.1%}")
    log(res["veredicto"])

with open(SALIDA, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)

print("\n===== SLOTS BENCH =====", flush=True)
comp = {k: {kk: vv for kk, vv in v.items() if kk != "por_item"}
        for k, v in res["brazos"].items()}
print(json.dumps({"brazos": comp, "diagnostico": res["diagnostico"],
                  "pareados": res.get("pareados"), "veredicto": res.get("veredicto")},
                 indent=2, ensure_ascii=False)[:6000], flush=True)
log("done")

#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Barrido de los PARAMETROS DE GENERACION que produccion sí expone.

LA VETA. El agente desplegado lee todo esto de variables de entorno
(inference/agent/tool_agent.py:137-148), o sea que cambiarlas no toca codigo:

    LOCAL_ANALYZER_ENABLE_THINKING = True     <- ¡activado por defecto!
    LOCAL_ANALYZER_TEMPERATURE     = 0.6
    LOCAL_ANALYZER_TOP_P           = 0.95
    LOCAL_ANALYZER_TOP_K           = 20

Y TODO nuestro banco se midio con thinking apagado y temperatura 0. Es el mismo
tipo de desajuste de regimen que ya nos mordio dos veces (el v5 y el idioma).

POR QUE PUEDE SER GRANDE. La fisica del presupuesto (DESIGN §8.3) midio 556
tokens por accion. Si la mayoria son tokens de PENSAMIENTO, apagarlo no cambia
solo la calidad: multiplica las acciones que caben en el presupuesto. Por eso
aqui se mide calidad Y COSTE, y lo que decide es la razon entre ambos — que es
la moneda real de produccion.

DISENO. 2x2 sobre el mismo conjunto de items (plan_action, el mas
discriminativo): {greedy, muestreo de produccion} x {thinking on, off}. Cuatro
brazos, mismos prompts, pareados.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_sampling_sweep.py
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

RAW = "https://raw.githubusercontent.com/jvilladuque90/arc-prize-2026-arc-agi-3/main"
MODEL = os.environ.get("SWEEP_MODEL", "Qwen/Qwen3-4B")
BATCH = int(os.environ.get("SWEEP_BATCH", "4"))
MAX_NEW = int(os.environ.get("SWEEP_MAX_NEW", "512"))   # thinking necesita sitio

t0 = time.time()
def log(m):
    print(f"[sweep {time.time()-t0:6.0f}s] {m}", flush=True)

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
prompt_plan_words, normalize = _mp["prompt_plan_words"], _mp["normalize"]

PLAN = [i for i in ITEMS if i["type"] == "plan_action"]
log(f"{len(PLAN)} items de planificacion")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

log(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA'}")
log(f"cargando {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
model.eval()

# (nombre, kwargs de generacion, thinking) — 'produccion' replica el agente real
ARMS = [
    ("greedy_sin_think",  dict(do_sample=False), False),
    ("greedy_con_think",  dict(do_sample=False), True),
    ("prod_sin_think",    dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20), False),
    ("prod_con_think",    dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20), True),
]


def correr(gen_kwargs, thinking):
    aciertos, por_item, tokens = 0, [], []
    ejemplos = []
    for i in range(0, len(PLAN), BATCH):
        lote = PLAN[i:i + BATCH]
        textos = [tok.apply_chat_template(
            [{"role": "user", "content": prompt_plan_words(it)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            for it in lote]
        enc = tok(textos, return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW,
                                 pad_token_id=tok.pad_token_id, **gen_kwargs)
        for j, it in enumerate(lote):
            salida = gen[j][enc["input_ids"].shape[1]:]
            n_tok = int((salida != tok.pad_token_id).sum())
            tokens.append(n_tok)
            txt = tok.decode(salida, skip_special_tokens=True)
            ok = normalize(txt, "plan_action") == it["answer"]
            aciertos += ok
            por_item.append(1 if ok else 0)
            if len(ejemplos) < 2:
                ejemplos.append({"tokens": n_tok, "crudo": txt.strip()[:200]})
    return {"aciertos": aciertos, "n": len(PLAN),
            "precision": round(aciertos / len(PLAN), 3),
            "tokens_medios": round(sum(tokens) / len(tokens), 1),
            "por_item": por_item, "ejemplos": ejemplos}


res = {"modelo": MODEL, "n": len(PLAN), "brazos": {}}
for nombre, kw, think in ARMS:
    r = correr(kw, think)
    res["brazos"][nombre] = r
    # la moneda real de produccion: acierto por cada 100 tokens gastados
    eficiencia = 100 * r["precision"] / max(1.0, r["tokens_medios"])
    log(f"  {nombre:18} {r['aciertos']:>3}/{r['n']} = {r['precision']:>6.1%} | "
        f"{r['tokens_medios']:>6.1f} tok/resp | eficiencia {eficiencia:.3f}")

print("\n===== SWEEP RESULT =====", flush=True)
print(json.dumps(res, indent=2, ensure_ascii=False)[:6000], flush=True)
log("done")

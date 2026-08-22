#!/usr/bin/env -S colab run --gpu T4
"""[Colab T4] Banco micro con modelos pequenos — hermano GPU de micro_eval_local.py.

CUANDO USAR ESTE Y NO EL LOCAL. El runner local (CPU, 8 hilos) corre el banco entero
con 0.6B en ~45 min, asi que para el 0.6B **no hace falta GPU**. Este script existe
para lo que la CPU no alcanza: modelos de 4B/8B, o barridos de muchas variantes.

NO usa vLLM: en la T4 gratis sus workers se llevan la RAM del host (~12.7 GB) y matan
el kernel de Jupyter -> "Timeout waiting for output". transformers con batching sobra
para generaciones cortas y greedy.

Los prompts NO se definen aqui: se descargan de scripts/micro_prompts.py del repo
publico, el mismo fichero que usa el runner local. Si se definieran dos veces, el A/B
acabaria midiendo la divergencia entre copias en vez del formato.

Uso: colab --auth=adc run --gpu T4 --timeout 7200 scripts/colab_micro_eval.py
Salida: tabla por variante + JSON completo en stdout (la VM se destruye al terminar).
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

# Los env del host NO llegan a la VM de Colab: lo que se quiera cambiar por
# corrida se fija aqui antes de leerlo.
os.environ.setdefault("MICRO_ONLY", "G.")      # esta corrida: solo regimen largo
# 64 basta aqui: el regimen mixto (marco EN + nota ES) responde escueto — medido
os.environ.setdefault("MICRO_MAX_NEW", "64")

RAW = "https://raw.githubusercontent.com/jvilladuque90/arc-prize-2026-arc-agi-3/main"
# Solo el 4B: es el unico de los dos que supera el suelo util (el 1.7B queda
# por debajo de la base trivial en planificacion). Cargar los dos seguidos
# agoto la VM gratis a mitad del segundo. Los numeros del 1.7B ya estan en
# docs/DESIGN.md 8.11 y no hace falta repetirlos.
MODELS = os.environ.get("MICRO_MODELS", "Qwen/Qwen3-4B").split(",")
# lote 4: los prompts del regimen largo rondan 2.000-2.500 tokens y con 16
# el cache KV se come la T4 (leccion del OOM del 8B)
# lote 2: con prompts de ~2.500 tokens (tablero 64x64) el lote 4 dio OOM en
# la T4 aun con expandable_segments
BATCH = int(os.environ.get("MICRO_BATCH", "2"))
# 12 tokens bastaban en espanol (el modelo contesta "ACTION3" y para) pero NO en
# ingles: ahi arranca con "To determine which action brings the object closest to
# the t..." y el corte llegaba antes de la respuesta. El brazo ingles marcaba 1/99,
# que no era un efecto de idioma sino truncamiento. Con margen suficiente los dos
# idiomas pueden responder y la comparacion mide lo que dice medir.
MAX_NEW = int(os.environ.get("MICRO_MAX_NEW", "64"))

t0 = time.time()
def log(m):
    print(f"[micro {time.time()-t0:6.0f}s] {m}", flush=True)

def _hb():
    while True:
        time.sleep(25)
        print(f"[hb {time.time()-t0:6.0f}s]", flush=True)
threading.Thread(target=_hb, daemon=True).start()

log("preparando entorno ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=4.51", "accelerate", "bitsandbytes"], check=False)
os.environ.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                   "TRANSFORMERS_NO_TORCHVISION": "1", "TOKENIZERS_PARALLELISM": "false",
                   # el 8B en 4 bits cabe, pero la fragmentacion del cache KV lo tumbaba
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

log("descargando banco y prompts del repo publico ...")
with urllib.request.urlopen(f"{RAW}/micro_bench.jsonl", timeout=60) as r:
    ITEMS = [json.loads(l) for l in r.read().decode("utf-8").splitlines() if l.strip()]
with urllib.request.urlopen(f"{RAW}/scripts/micro_prompts.py", timeout=60) as r:
    _mp = {}
    exec(compile(r.read().decode("utf-8"), "micro_prompts.py", "exec"), _mp)
VARIANTS, normalize = _mp["VARIANTS"], _mp["normalize"]
trivial_baselines, paired_contrast = _mp["trivial_baselines"], _mp["paired_contrast"]
log(f"{len(ITEMS)} items")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

log(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA'}")
BASELINES = trivial_baselines(ITEMS)
log(f"linea base trivial: {json.dumps(BASELINES, ensure_ascii=False)}")


def run_model(model_id):
    batch = BATCH
    log(f"cargando {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Los >4B no caben en fp16 en una T4 de 16 GB (un 8B son ~16 GB solo de pesos),
    # asi que van en 4 bits. La cuantizacion afecta a los DOS brazos por igual, y lo
    # que se compara aqui son formatos de prompt dentro del mismo modelo, no modelos
    # entre si.
    kw = {"dtype": torch.float16, "device_map": "cuda"}
    if any(t in model_id for t in ("8B", "7B", "14B")):
        from transformers import BitsAndBytesConfig
        kw = {"device_map": "cuda", "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")}
        log(f"{model_id}: cargando en 4 bits (no cabe en fp16 en T4)")
        batch = 4   # con lote 16 el cache KV del 8B agota los 14.5 GiB de la T4
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()

    def generate(prompts):
        outs = []
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
                     for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=8192).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            outs += [tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                     for j in range(len(chunk))]
        return outs

    solo = os.environ.get("MICRO_ONLY", "")
    res = {}
    for name, kind, builder in VARIANTS:
        if solo and not name.startswith(tuple(solo.split(","))):
            continue
        subset = [i for i in ITEMS if i["type"] == kind]
        if not subset:
            continue
        raws = generate([builder(it) for it in subset])
        per_item, hits, examples = [], 0, []
        for it, raw in zip(subset, raws):
            got = normalize(raw, kind)
            ok = got == it["answer"]
            hits += ok
            per_item.append(1 if ok else 0)
            if len(examples) < 3:
                examples.append({"game": it["game"], "esperado": it["answer"], "obtenido": got,
                                 "crudo": (raw or "").strip()[:300]})
        acc = hits / len(subset)
        res[name] = {"n": len(subset), "aciertos": hits, "precision": round(acc, 3),
                     "por_item": per_item, "ejemplos": examples}
        base = BASELINES.get(kind, {}).get("precision", 0)
        flag = "" if acc > base else "   <- NO supera la base trivial"
        log(f"  {name:16} {hits:3}/{len(subset):3} = {acc:5.1%}  (base {base:.0%}){flag}")

    pares = paired_contrast(res)
    for label, d in pares.items():
        log(f"  pareado [{label}] {d}")
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

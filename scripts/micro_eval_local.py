"""Corre el banco micro EN LOCAL (CPU), sin gastar quota de Kaggle ni de Colab.

POR QUE EN LOCAL. Los prompts del banco son cortos (~330 tokens los de rejilla,
~86 los demas): el run completo son ~93k tokens de prefill. Con un modelo de 0.6B
eso cabe en unos minutos de CPU. No hacia falta GPU — la estabamos pidiendo por
inercia, y las dos cuentas de Colab estaban en enfriamiento.

Uso:
  python scripts/micro_eval_local.py --model Qwen/Qwen3-0.6B
  python scripts/micro_eval_local.py --model Qwen/Qwen3-0.6B --types plan_action
  python scripts/micro_eval_local.py --model Qwen/Qwen3-1.7B --out micro_eval_1.7b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from micro_prompts import (PAIRS, VARIANTS, normalize, paired_contrast,  # noqa: E402
                           trivial_baselines)

T0 = time.time()


def log(m: str) -> None:
    print(f"[micro {time.time()-T0:6.0f}s] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--bench", default="micro_bench.jsonl")
    ap.add_argument("--types", nargs="*", default=None,
                    help="limitar a ciertos tipos (p.ej. plan_action which_action)")
    ap.add_argument("--limit", type=int, default=0, help="max items por variante (0=todos)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    items = [json.loads(l) for l in (ROOT / args.bench).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if args.types:
        items = [i for i in items if i["type"] in args.types]
    log(f"{len(items)} items, {len(set(i['game'] for i in items))} juegos")

    baselines = trivial_baselines(items)
    log(f"linea base trivial: "
        + ", ".join(f"{k}={v['precision']:.0%} ('{v['clase']}')" for k, v in baselines.items()))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(max(1, (__import__('os').cpu_count() or 4)))

    log(f"cargando {args.model} en CPU ...")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    log("modelo listo")

    def generate(prompts):
        outs = []
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
                     for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True))
            if (i // args.batch) % 2 == 0:
                log(f"    {min(i+args.batch, len(prompts))}/{len(prompts)}")
        return outs

    res = {}
    for name, kind, builder in VARIANTS:
        subset = [i for i in items if i["type"] == kind]
        if args.limit:
            subset = subset[:args.limit]
        if not subset:
            continue
        log(f"{name} ({len(subset)} items) ...")
        raws = generate([builder(it) for it in subset])
        per_item, hits, examples = [], 0, []
        for it, raw in zip(subset, raws):
            got = normalize(raw, kind)
            ok = got == it["answer"]
            hits += ok
            per_item.append(1 if ok else 0)
            if len(examples) < 3:
                examples.append({"game": it["game"], "esperado": it["answer"], "obtenido": got,
                                 "crudo": (raw or "").strip()[:60]})
        acc = hits / len(subset)
        # DIAGNOSTICO DE DEGENERACION: un modelo que contesta siempre lo mismo saca
        # ~la base trivial y parece "casi competente". Sin esta cifra no se
        # distingue de uno que razona mal pero razona.
        from collections import Counter
        dist = Counter(normalize(r, kind) for r in raws)
        top_ans, top_n = dist.most_common(1)[0]
        res[name] = {"n": len(subset), "aciertos": hits, "precision": round(acc, 3),
                     "por_item": per_item, "ejemplos": examples,
                     "respuestas_distintas": len(dist),
                     "moda": {"respuesta": top_ans, "frac": round(top_n / len(subset), 3)}}
        base = baselines.get(kind, {}).get("precision", 0)
        flag = "" if acc > base else "   <- NO supera la base trivial"
        deg = ("   [DEGENERADO: contesta '%s' el %.0f%% de las veces]"
               % (top_ans, 100 * top_n / len(subset))) if top_n / len(subset) > 0.7 else ""
        log(f"  {name:16} {hits:3}/{len(subset):3} = {acc:5.1%}  (base {base:.0%})"
            f" | {len(dist)} respuestas distintas{flag}{deg}")

    pares = paired_contrast(res)
    for label, d in pares.items():
        log(f"  pareado [{label}] {d}")

    out = {"modelo": args.model, "n_items": len(items), "linea_base": baselines,
           "variantes": res, "pareado": pares}
    if args.out:
        (ROOT / args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
        log(f"-> {args.out}")
    print("\n===== RESUMEN =====")
    for name, d in res.items():
        print(f"  {name:16} {d['precision']:6.1%}  ({d['aciertos']}/{d['n']})")
    for label, d in pares.items():
        print(f"  pareado [{label}]: {json.dumps(d, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Brazo del STACK DE SERVICIO: alinear la capacidad de vLLM con la concurrencia del harness.

EL DESAJUSTE, medido en el log de todas nuestras corridas NVFP4:

    Initial free memory 94.43 GiB, reserved 5.0 GiB memory for KV Cache
    GPU KV cache size: 105,202 tokens
    Maximum concurrency for 32,768 tokens per request: 3.21x

El harness lanza `concurrency = 28` juegos a la vez y la cache KV da para **3,21**.
Ademas el perfil fija `TAAF_VLLM_MAX_NUM_SEQS = 8`, asi que vLLM sirve 8 peticiones y
las otras 20 hacen cola — mientras ~75 GB de la tarjeta estan sin usar. El propio
serving_setup.py trae `MAX_NUM_SEQS = 28` como defecto: el perfil lo baja a proposito.

CAMBIO (una sola idea, dos mandos acoplados: no se puede subir seqs sin subir KV):
    TAAF_VLLM_KV_CACHE_MEMORY_BYTES  5 GiB -> 46 GiB   (tope del stack: 96 GiB)
    TAAF_VLLM_MAX_NUM_SEQS           8     -> 28       (= concurrency del harness)

Aritmetica: 105.202 tokens con 5 GiB => ~21.040 tokens/GiB. Para 28 peticiones de
32.768 tokens hacen falta 917.504 => ~43,6 GiB; con 46 GiB quedan ~30x de margen.
Huella: ~14 GB de pesos NVFP4 + 46 de KV + graficos/activaciones ~5 = ~65 de 94.

RIESGO DECLARADO: con MTP, los lotes grandes bajan la aceptacion del borrador (por eso
existe `mtp_dynamic_batch_schedule`). Puede que el autor midiera justo eso y por eso
eligiera 8. Lo que el banco decide es si servir 28 en paralelo compensa esa perdida.
NO se toca el modelo, ni MTP, ni el dtype de KV, ni prefix caching.

Uso:  python scripts/build_nvfp4_kv.py nvfp4_carry_long
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = "PUBLIC25_VLLM_PROFILE_ENV = {"
KV_BYTES = 46 * 1024 ** 3          # 49.392.123.904
MAX_SEQS = 28


def main() -> int:
    nombre = sys.argv[1] if len(sys.argv) > 1 else "nvfp4_carry_long"
    src = ROOT / "notebooks" / f"{nombre}.ipynb"
    out = ROOT / "notebooks" / f"{nombre.replace('_long', '')}_kv_long.ipynb"

    nb = json.loads(src.read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"])
           if c.get("cell_type") == "code" and HOOK in "".join(c["source"])]
    if len(idx) != 1:
        print(f"esperaba 1 celda con el perfil de vLLM, hay {len(idx)}")
        return 1
    i = idx[0]
    s = "".join(nb["cells"][i]["source"])
    viejo_kv = '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",'
    viejo_sq = '"TAAF_VLLM_MAX_NUM_SEQS": "8",'
    if s.count(viejo_kv) != 1 or s.count(viejo_sq) != 1:
        print("no encontre los dos mandos exactos en el perfil")
        return 1
    nuevo = (s.replace(viejo_kv, f'"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "{KV_BYTES}",')
              .replace(viejo_sq, f'"TAAF_VLLM_MAX_NUM_SEQS": "{MAX_SEQS}",')
              .replace("PUBLIC25_VLLM_PROFILE_NAME = 'kv5-bf16-mtp3-c8-cg32'",
                       "PUBLIC25_VLLM_PROFILE_NAME = 'kv46-bf16-mtp3-c28-cg32'"))
    compile(nuevo, "<celda-perfil>", "exec")
    nb["cells"][i]["source"] = nuevo.splitlines(keepends=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    orig = json.loads(src.read_text(encoding="utf-8"))["cells"]
    new = json.loads(out.read_text(encoding="utf-8"))["cells"]
    dif = [k for k in range(len(orig)) if "".join(orig[k]["source"]) != "".join(new[k]["source"])]
    ok = len(new) == len(orig) and dif == [i]
    malas = []
    for k, c in enumerate(new, 1):
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c["source"]), f"<c{k}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {k}: {e}")
    print(f"{out.name}: KV 5 GiB -> {KV_BYTES/1024**3:.0f} GiB, max_num_seqs 8 -> {MAX_SEQS}")
    print(f"  una sola celda cambiada: {'OK' if ok else 'FALLA ' + str(dif)} | compilan: {'OK' if not malas else malas}")
    return 0 if ok and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

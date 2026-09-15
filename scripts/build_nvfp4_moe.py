"""Brazo del MODELO OBJETIVO: nucleo fusionado de expertos de Blackwell.

EL HALLAZGO. `TAAF_VLLM_MOE_BACKEND` no esta en el perfil ganador
(`kv5-bf16-mtp3-c8-cg32` fija ocho claves y esta no es una) y acepta exactamente un valor
no nulo (`serving_setup.py:404`):

    if moe_backend not in {None, "flashinfer_b12x"}:
        raise RuntimeError(...)

Por defecto es `None`, asi que hoy corremos el camino de expertos de serie.

POR QUE ES LA PALANCA GRANDE DE ESTE EJE. El modelo es `Qwen4ExpForConditionalGeneration`
con **512 expertos en 48 capas** (`serving_setup.py:1495-1500`): la ruta de expertos es el
grueso del computo de cada paso hacia adelante del modelo OBJETIVO. Comparar con el brazo
anterior (IndexShare, DESIGN 8.58): aquel actuaba sobre la cabeza borradora de MTP, que tiene
**una sola capa** (`mtp_num_hidden_layers: 1`), y rindio **+0,3%**. Este actua sobre tres
ordenes de magnitud mas de computo.

Y `b12x` es el nucleo de **Blackwell**, que es literalmente nuestra tarjeta: el propio setup
fija `TORCH_CUDA_ARCH_LIST = "12.0"` (`serving_setup.py:1423`) y la maquina es la RTX PRO 6000.
El mando esta hecho para este hardware exacto.

CALIDAD: un nucleo fusionado calcula la misma funcion (misma seleccion de expertos, mismos
pesos); solo cambia la implementacion. Igual que en 8.58, este mando mueve VELOCIDAD, y la
velocidad suma porque el presupuesto de pensamiento del agente es POR TIEMPO (<=60 s/turno):
mas tokens por segundo = mas razonamiento dentro del mismo turno.

CAMBIO (una sola idea, un solo mando):
    TAAF_VLLM_MOE_BACKEND   (ausente, = None)  ->  "flashinfer_b12x"

No se toca el modelo, ni MTP, ni la KV, ni la memoria, ni el agente.

RIESGO DECLARADO, y este es distinto al de los brazos anteriores: `--moe-backend` **no esta en
la lista blanca de banderas** que el setup verifica contra `vllm serve --help=all`
(`serving_setup.py:1771-1790`). El autor anade la bandera al comando sin comprobar que la
version pineada la soporte. Si `0.1.dev20073+g8e685d198` no la reconoce, el servidor muere en el
arranque por argumento desconocido y el brazo cuesta ~10 min. A favor: el autor escribio
validacion para el mando en tres sitios (setup 404 y 511, teardown 513), lo que sugiere que lo
uso. Segundo riesgo: el nucleo de flashinfer podria reservar sus propios buffers de trabajo, y
el margen de memoria es de 12,6 GiB (DESIGN 8.56).

Uso:  python scripts/build_nvfp4_moe.py nvfp4_carry_long
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = "PUBLIC25_VLLM_PROFILE_ENV = {"
ANCLA = '    "TAAF_VLLM_MTP_TOKENS": "3",'
NUEVA = '    "TAAF_VLLM_MOE_BACKEND": "flashinfer_b12x",'
VIEJO_NOMBRE = "PUBLIC25_VLLM_PROFILE_NAME = 'kv5-bf16-mtp3-c8-cg32'"
NUEVO_NOMBRE = "PUBLIC25_VLLM_PROFILE_NAME = 'kv5-bf16-mtp3-c8-cg32-moeb12x'"

ESPERADOS = {
    "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0",
    "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",
    "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
    "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
    "TAAF_VLLM_MAX_NUM_SEQS": "8",
    "TAAF_VLLM_MTP_TOKENS": "3",
    "TAAF_VLLM_OMP_THREADS": "1",
}


def main() -> int:
    nombre = sys.argv[1] if len(sys.argv) > 1 else "nvfp4_carry_long"
    src = ROOT / "notebooks" / f"{nombre}.ipynb"
    out = ROOT / "notebooks" / f"{nombre.replace('_long', '')}_moe_long.ipynb"

    nb = json.loads(src.read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"])
           if c.get("cell_type") == "code" and HOOK in "".join(c["source"])]
    if len(idx) != 1:
        print(f"esperaba 1 celda con el perfil de vLLM, hay {len(idx)}")
        return 1
    i = idx[0]
    s = "".join(nb["cells"][i]["source"])
    if s.count(ANCLA) != 1 or s.count(VIEJO_NOMBRE) != 1:
        print("no encontre el ancla exacta ni el nombre del perfil")
        return 1
    if "MOE_BACKEND" in s:
        print("el mando ya estaba puesto; nada que hacer")
        return 1
    nuevo = (s.replace(ANCLA, NUEVA + "\n" + ANCLA)
              .replace(VIEJO_NOMBRE, NUEVO_NOMBRE))
    compile(nuevo, "<celda-perfil>", "exec")
    nb["cells"][i]["source"] = nuevo.splitlines(keepends=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # --- compuertas ---
    orig = json.loads(src.read_text(encoding="utf-8"))["cells"]
    new = json.loads(out.read_text(encoding="utf-8"))["cells"]
    dif = [k for k in range(len(orig)) if "".join(orig[k]["source"]) != "".join(new[k]["source"])]
    una_celda = len(new) == len(orig) and dif == [i]

    perfil: dict[str, str] = {}
    for nodo in ast.parse("".join(new[i]["source"])).body:
        if (isinstance(nodo, ast.Assign)
                and any(getattr(t, "id", None) == "PUBLIC25_VLLM_PROFILE_ENV"
                        for t in nodo.targets)):
            perfil = ast.literal_eval(nodo.value)
    claves_ok = len(perfil) == 9 and perfil.get("TAAF_VLLM_MOE_BACKEND") == "flashinfer_b12x"
    intactos = all(perfil.get(k) == v for k, v in ESPERADOS.items())

    malas = []
    for k, c in enumerate(new, 1):
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c["source"]), f"<c{k}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {k}: {e}")

    print(f"{out.name}: MOE_BACKEND ausente(=None) -> flashinfer_b12x")
    print(f"  una sola celda cambiada : {'OK' if una_celda else 'FALLA ' + str(dif)}")
    print(f"  perfil 9 claves, mando ok: {'OK' if claves_ok else 'FALLA ' + str(perfil)}")
    print(f"  los 8 mandos previos intactos: {'OK' if intactos else 'FALLA'}")
    print(f"  todas las celdas compilan: {'OK' if not malas else malas}")
    return 0 if (una_celda and claves_ok and intactos and not malas) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Brazo de DECODIFICACION ESPECULATIVA: encender IndexShare entre iteraciones de MTP.

EL HALLAZGO. El perfil ganador del autor se llama `kv5-bf16-mtp3-c8-cg32` y fija ocho
variables. De los tres mandos de MTP que `serving_setup.py` expone, **solo uno esta en el
perfil** (`TAAF_VLLM_MTP_TOKENS = 3`). Los otros dos corren en su valor por defecto:

    TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION   -> _binary_environment(default=False)  APAGADO
    TAAF_VLLM_MTP_DYNAMIC_BATCH_SCHEDULE      -> None                                APAGADO

La fuente vendida en el propio bundle dice que el primero deberia estar ENCENDIDO para esta
arquitectura. En `src/sglang-rtxpro6000/src/sglang/srt/configs/qwen4_exp.py`:

    index_share_for_mtp_iteration=True,          # <- valor por defecto del constructor
    ...
    # MTP draft decode steps reuse the draft-extend indexer selection
    # (GLM-5.2 IndexShare); default on for Qwen4-Exp, checkpoint config
    # or --json-model-override-args can disable it.

Y nuestro modelo es exactamente `qwen4_exp`: `serving_setup.py:724-726` exige
`architectures == ["Qwen4ExpForConditionalGeneration"]` y `model_type == "qwen4_exp"`.

MECANICA. Con 3 tokens de borrador, cada paso de decodificacion especulativa recalcula la
seleccion del indexador de atencion. IndexShare la calcula UNA vez en el draft-extend y la
reutiliza en las iteraciones del borrador. Es puro ahorro de computo en el modelo borrador.

POR QUE NO CAMBIA LA CALIDAD. La decodificacion especulativa verifica cada token del borrador
contra el modelo objetivo por muestreo de rechazo: la distribucion de salida es la misma
tokene a token. Un borrador peor solo baja la tasa de aceptacion (hoy 64-76%); nunca altera
el texto final. Asi que este mando solo mueve VELOCIDAD.

POR QUE LA VELOCIDAD AQUI SUMA, y no es de dos colas. El presupuesto de pensamiento del
agente es **por tiempo** (<=60 s por turno), no por tokens. Mas tokens por segundo significa
mas razonamiento DENTRO del mismo turno de 60 s, con el mismo numero de turnos por juego. Los
dos fracasos medidos del eje caen del otro lado: 0 s de pensamiento -> 12 niveles (poco
razonamiento) y <=180 s -> 17 niveles (pocos turnos). Ir mas rapido a presupuesto de tiempo
constante empuja la variable buena de las dos.

CAMBIO (una sola idea, un solo mando):
    TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION   (ausente, = 0)  ->  "1"

No se toca el modelo, ni el numero de tokens de MTP, ni la KV, ni la memoria, ni el agente.
Riesgo de memoria: ninguno; el mando no reserva nada.

RIESGO DECLARADO: el autor pudo medirlo y descartarlo sin dejarlo en el nombre del perfil. Si
vLLM no acepta el campo `index_share_for_mtp_iteration` en `--speculative-config`, el motor no
arranca y el brazo muere en el arranque (coste ~10 min), como paso con el brazo de KV.

Uso:  python scripts/build_nvfp4_mtpshare.py nvfp4_carry_long
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = "PUBLIC25_VLLM_PROFILE_ENV = {"
ANCLA = '    "TAAF_VLLM_MTP_TOKENS": "3",'
NUEVA = '    "TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION": "1",'
VIEJO_NOMBRE = "PUBLIC25_VLLM_PROFILE_NAME = 'kv5-bf16-mtp3-c8-cg32'"
NUEVO_NOMBRE = "PUBLIC25_VLLM_PROFILE_NAME = 'kv5-bf16-mtp3ix-c8-cg32'"


def main() -> int:
    nombre = sys.argv[1] if len(sys.argv) > 1 else "nvfp4_carry_long"
    src = ROOT / "notebooks" / f"{nombre}.ipynb"
    out = ROOT / "notebooks" / f"{nombre.replace('_long', '')}_mtpshare_long.ipynb"

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
    if "MTP_INDEX_SHARE" in s:
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

    # el diccionario resultante debe tener 9 claves y el mando en "1"
    perfil: dict[str, str] = {}
    for nodo in ast.parse("".join(new[i]["source"])).body:
        if (isinstance(nodo, ast.Assign)
                and any(getattr(t, "id", None) == "PUBLIC25_VLLM_PROFILE_ENV"
                        for t in nodo.targets)):
            perfil = ast.literal_eval(nodo.value)
    claves_ok = len(perfil) == 9 and perfil.get("TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION") == "1"
    intactos = all(perfil.get(k) == v for k, v in {
        "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0",
        "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
        "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",
        "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
        "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
        "TAAF_VLLM_MAX_NUM_SEQS": "8",
        "TAAF_VLLM_MTP_TOKENS": "3",
        "TAAF_VLLM_OMP_THREADS": "1",
    }.items())

    malas = []
    for k, c in enumerate(new, 1):
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c["source"]), f"<c{k}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {k}: {e}")

    print(f"{out.name}: MTP_INDEX_SHARE_FOR_ITERATION ausente(=0) -> 1")
    print(f"  una sola celda cambiada : {'OK' if una_celda else 'FALLA ' + str(dif)}")
    print(f"  perfil 9 claves, mando=1: {'OK' if claves_ok else 'FALLA ' + str(perfil)}")
    print(f"  los 8 mandos previos intactos: {'OK' if intactos else 'FALLA'}")
    print(f"  todas las celdas compilan: {'OK' if not malas else malas}")
    return 0 if (una_celda and claves_ok and intactos and not malas) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Genera notebooks/franzen_d1.ipynb: copia fiel de Franzen + UNA celda de entorno (brazo D1).

D1 = historia mas profunda A IGUAL MEMORIA DE KV (plan.md, DESIGN 8.86).
  * ARC3_CONTEXT_DRAIN_TOKENS es CUANTO SE DESCARTA en cada recorte (no lo que se conserva): con el
    presupuesto de entrada de 118.272 tokens, Franzen usa 58K (acotado a presupuesto/2 = 59.136), asi
    que la historia oscila entre 59K y 118K (media ~89K). Con 24K oscila entre ~94K y 118K (media ~106K,
    +19 %), a costa de recortar ~2,5 veces mas a menudo (mas prefijo invalidado y mas reencolados).
  * ARC3_MAX_ACTIVE_STREAMS 10 -> 8: el pool de KV es de 1.011.264 tokens y hoy trabaja al 0,89-0,98;
    con 8 plazas y media 106K son ~850K (0,84). Menos plazas = menos decodificacion en lote.
Evidencia externa (otra base): historia 49K->69,6K con KV FP8 dio +8,0 (Siriki). Aqui la prueba es en el
oculto; la ganancia o la perdida son del compromiso historia/plazas, y las dos tiradas van juntas a proposito
(a igual KV no se puede subir la historia sin bajar las plazas).

Uso:  python scripts/build_franzen_d1.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "franzen_m2.ipynb"
OUT = ROOT / "notebooks" / "franzen_d1.ipynb"
ANCHOR = "harness patch applied successfully"
MARCA = "# D1 HISTORIA MAS PROFUNDA"

CELL = MARCA + ''' — brazo de EXPERIMENTO (plan.md, DESIGN 8.86). Solo variables de entorno.
# Se aplica DESPUES de la celda de Franzen (que hace os.environ.update) y ANTES de importar el harness.
os.environ['ARC3_CONTEXT_DRAIN_TOKENS'] = str(24 * 1024)   # Franzen: 58 * 1024
os.environ['ARC3_MAX_ACTIVE_STREAMS'] = '8'                # Franzen: 10
print('D1: ARC3_CONTEXT_DRAIN_TOKENS=%s ARC3_MAX_ACTIVE_STREAMS=%s' % (
    os.environ['ARC3_CONTEXT_DRAIN_TOKENS'], os.environ['ARC3_MAX_ACTIVE_STREAMS']), flush=True)
'''


def main() -> int:
    compile(CELL, "<celda-d1>", "exec")
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"])
           if c.get("cell_type") == "code" and ANCHOR in "".join(c["source"])]
    if len(idx) != 1:
        print(f"esperaba 1 celda con el ancla del parche, hay {len(idx)}")
        return 1
    at = idx[0] + 1
    cfg = "".join(nb["cells"][idx[0]]["source"])
    if "os.environ.update({k: str(v) for k,v in setup_env.items()})" not in cfg:
        print("la celda de Franzen ya no termina en os.environ.update: revisar el orden")
        return 1
    nb["cells"].insert(at, {"cell_type": "code", "execution_count": None, "metadata": {},
                            "outputs": [], "source": CELL.splitlines(keepends=True)})
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    orig = json.loads(SRC.read_text(encoding="utf-8"))["cells"]
    new = json.loads(OUT.read_text(encoding="utf-8"))["cells"]
    ok = len(new) == len(orig) + 1
    for i, c in enumerate(orig):
        j = i if i < at else i + 1
        if "".join(c["source"]) != "".join(new[j]["source"]):
            print(f"celda original {i} cambio")
            ok = False
    malas = []
    for i, c in enumerate(new, 1):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("%%writefile"):
            continue
        limpio = "\n".join("pass" if l.lstrip().startswith(("%", "!")) else l for l in src.splitlines())
        try:
            compile(limpio, f"<c{i}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {i}: {e}")
    print(f"celda insertada en indice {at} | {OUT.name} {OUT.stat().st_size} bytes")
    print("una sola celda anadida y las demas intactas:", "OK" if ok else "FALLA")
    print("todas las celdas compilan:", "OK" if not malas else malas)
    return 0 if ok and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

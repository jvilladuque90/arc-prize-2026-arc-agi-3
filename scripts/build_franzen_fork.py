"""Genera notebooks/franzen_m2.ipynb: copia FIEL del notebook publico de Daniel Franzen,
`dfranzen/arc-agi-3-milestone-2-solution` (solucion del hito #2).

POR QUE (plan.md, auditoria 2026-10-03): desde el 2026-09-30 ese notebook subio a cientos de
equipos de ~5 a 27-35 en el set oculto. Una reproduccion INDEPENDIENTE sin cambios dio 27,80 y una
instantanea comunitaria de copias sin cambios da media 25,77 y sd 3,93. Nuestra mejor base (sheetu,
vLLM + NVFP4 + harness de Tufa Labs) da 3,78 / 2,49 / 4,79.

NINGUNA EDICION. El notebook ya es barato fuera del rerun: en Save & Run juega 10 juegos publicos
durante 25 minutos (bm.solver.max_runtime_s_per_game = 25*60, concurrency 10), asi que no hace
falta nuestro recorte de ventana. Lo unico que se comprueba aqui es que la copia es identica byte a
byte (celdas) al original descargado y que todas las celdas compilan.

Atribucion: Daniel Franzen (solucion y parche del harness), Tufa Labs (duck harness: Bessis,
Cottaar, Pressman, Smit, Tesnar, Viel), John Pezzulli (Pennyroyal, fork de SGLang), Mamy
Ratsimbazafy y Gabriel Olympie (parches), Intel (cuantizacion AutoRound), Albucino (drafter MTP).
Nuestro derivado se publica abierto.

Uso:  python scripts/build_franzen_fork.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "_tmp_pub" / "dfranzen_arc-agi-3-milestone-2-solution" / "arc-agi-3-milestone-2-solution.ipynb"
OUT = ROOT / "notebooks" / "franzen_m2.ipynb"


def main() -> int:
    if not SRC.exists():
        print(f"falta {SRC}: descargalo con  kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution -p ...")
        return 1
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    orig = json.loads(SRC.read_text(encoding="utf-8"))["cells"]
    new = json.loads(OUT.read_text(encoding="utf-8"))["cells"]
    iguales = len(orig) == len(new) and all(
        "".join(a["source"]) == "".join(b["source"]) and a["cell_type"] == b["cell_type"]
        for a, b in zip(orig, new))
    malas = []
    for i, c in enumerate(new, 1):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        # %%writefile / !comandos son magics de IPython: se compilan sin esas lineas
        limpio = "\n".join("pass" if l.lstrip().startswith(("%%", "%", "!")) else l
                           for l in src.splitlines())
        if src.lstrip().startswith("%%writefile"):
            continue            # el cuerpo es un parche, no Python
        try:
            compile(limpio, f"<c{i}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {i}: {e}")
    print(f"{OUT.name}: {len(new)} celdas | {OUT.stat().st_size} bytes")
    print("copia identica al original celda a celda:", "OK" if iguales else "FALLA")
    print("celdas de codigo compilan:", "OK" if not malas else malas)
    return 0 if iguales and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

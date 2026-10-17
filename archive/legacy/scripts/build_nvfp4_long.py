"""Variantes de 60 minutos (regimen) de los notebooks NVFP4, para kernels de experimento.

La ventana de 25 min no ejercita el nivel 2 (DESIGN 8.9: el instrumento valido es la
corrida larga en regimen). Este script toma un notebook NVFP4 ya generado y cambia
UNICAMENTE el default de la ventana offline dentro de nuestro bloque de recorte
(la cadena es unica en el notebook). Compuerta: exactamente esa subcadena y nada mas.

Nunca se aplica al notebook del envio (notebooks/nvfp4.ipynb -> slug arc-agi3-nvfp4):
la tarea de envio lee kernel_versions.json al dispararse, asi que publicar una version
nueva de ese slug cambiaria lo que se envia. Las variantes van a slugs propios.

Uso:
  python scripts/build_nvfp4_long.py nvfp4        -> notebooks/nvfp4_long.ipynb
  python scripts/build_nvfp4_long.py nvfp4_carry  -> notebooks/nvfp4_carry_long.ipynb
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEJO = 'os.environ.get("TAAF_OFFLINE_SOFT_MIN", "25")'
NUEVO = 'os.environ.get("TAAF_OFFLINE_SOFT_MIN", "60")'


def main() -> int:
    nombre = sys.argv[1] if len(sys.argv) > 1 else "nvfp4"
    src = ROOT / "notebooks" / f"{nombre}.ipynb"
    out = ROOT / "notebooks" / f"{nombre}_long.ipynb"
    nb = json.loads(src.read_text(encoding="utf-8"))
    hits = sum("".join(c["source"]).count(VIEJO) for c in nb["cells"])
    if hits != 1:
        print(f"esperaba exactamente 1 aparicion de la ventana offline, hay {hits}")
        return 1
    for c in nb["cells"]:
        s = "".join(c["source"])
        if VIEJO in s:
            c["source"] = s.replace(VIEJO, NUEVO, 1).splitlines(keepends=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # compuerta: la unica diferencia es esa subcadena
    a = json.loads(src.read_text(encoding="utf-8"))["cells"]
    b = json.loads(out.read_text(encoding="utf-8"))["cells"]
    ok = len(a) == len(b)
    distintas = 0
    for x, y in zip(a, b):
        sx, sy = "".join(x["source"]), "".join(y["source"])
        if sx != sy:
            distintas += 1
            if sy.replace(NUEVO, VIEJO, 1) != sx:
                ok = False
    ok = ok and distintas == 1
    malas = []
    for i, c in enumerate(b, 1):
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c["source"]), f"<c{i}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {i}: {e}")
    print(f"{out.name}: ventana 25 -> 60 min | una sola celda cambiada y solo por eso: "
          f"{'OK' if ok else 'FALLA'} | compilan: {'OK' if not malas else malas}")
    return 0 if ok and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

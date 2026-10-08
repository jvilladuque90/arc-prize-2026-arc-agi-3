"""Genera notebooks/franzen_h1.ipynb: copia fiel de Franzen + UNA celda con H1 (endurecimiento).

H1 = src/arc3/franzen_hardening.py: (R1) un timeout/caida del sandbox ya no vacia la biblioteca de
funciones persistentes; (R2) un `result` list/dict enorme se recorta. Una variable, sin riesgo de
modelo, verificada sobre el ToolAgent real (scripts/test_franzen_hardening.py).

La celda va justo DESPUES de la de configuracion de Franzen (donde se aplica su parche con git apply)
y ANTES de que el notebook importe tool_agent. Compuerta: exactamente una celda anadida, ninguna otra
tocada, todo compila.

Uso:  python scripts/build_franzen_h1.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "franzen_m2.ipynb"
OUT = ROOT / "notebooks" / "franzen_h1.ipynb"
MOD = ROOT / "src" / "arc3" / "franzen_hardening.py"
ANCHOR = "harness patch applied successfully"
MARCA = "# H1 ENDURECIMIENTO"

CELL_TEMPLATE = MARCA + ''' — brazo de EXPERIMENTO (plan.md paso 3, DESIGN 8.86).
# Una variable: R1 la biblioteca de funciones persistentes sobrevive a un timeout/caida del sandbox;
# R2 un result list/dict enorme se recorta. Se aplica sobre el tool_agent.py ya parcheado por Franzen.
import base64 as _b64h
_nsh = {}
exec(compile(_b64h.b64decode("__B64__").decode("utf-8"), "franzen_hardening.py", "exec"), _nsh)
_ta_path = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
with open(_ta_path, "r", encoding="utf-8", newline="") as _fh:
    _src_before = _fh.read()
_src_after = _nsh["apply_to_source"](_src_before)
assert _src_after != _src_before and "H1/R1" in _src_after and "H1/R2" in _src_after
compile(_src_after, str(_ta_path), "exec")
with open(_ta_path, "w", encoding="utf-8", newline="") as _fh:
    _fh.write(_src_after)
print("H1 hardening applied to", _ta_path, flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-h1>", "exec")

    nb = json.loads(SRC.read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"])
           if c.get("cell_type") == "code" and ANCHOR in "".join(c["source"])]
    if len(idx) != 1:
        print(f"esperaba 1 celda con el ancla del parche, hay {len(idx)}")
        return 1
    at = idx[0] + 1
    nb["cells"].insert(at, {"cell_type": "code", "execution_count": None, "metadata": {},
                            "outputs": [], "source": cell_src.splitlines(keepends=True)})
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
    fiel = base64.b64decode(b64) == MOD.read_bytes()
    print(f"celda insertada en indice {at} | {OUT.name} {OUT.stat().st_size} bytes")
    print("una sola celda anadida y las demas intactas:", "OK" if ok else "FALLA")
    print("el modulo embebido se reconstruye exacto:", "OK" if fiel else "FALLA")
    print("todas las celdas compilan:", "OK" if not malas else malas)
    return 0 if ok and fiel and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

"""Genera un notebook E1: base + UNA celda que fija el esfuerzo de razonamiento (src/arc3/franzen_effort.py).

Uso:  python scripts/build_franzen_e1.py <base.ipynb> <medium|low|xhigh> <salida.ipynb>
Ej.:  python scripts/build_franzen_e1.py notebooks/franzen_g15.ipynb medium notebooks/franzen_g15_med.ipynb

La celda va justo DESPUES de la de configuracion de Franzen (donde se aplica su parche con git apply) y ANTES
de que el notebook importe el harness. Compuerta: exactamente una celda anadida, ninguna otra tocada,
el modulo embebido se reconstruye exacto y todo compila.
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "src" / "arc3" / "franzen_effort.py"
ANCHOR = "harness patch applied successfully"
MARCA = "# E1 ESFUERZO DE RAZONAMIENTO FIJO"

CELL = MARCA + ''' — brazo de EXPERIMENTO (DESIGN 8.88). Una variable: reasoning_effort = __NIVEL__.
# La plantilla de Qwen3.8 usa xhigh por defecto ("piensa con cuidado, valida supuestos, considera
# alternativas"). medium no inyecta instruccion; low inyecta "piensa breve". Llega al servidor como
# chat_template_kwargs["reasoning_effort"].
import base64 as _b64e
_nse = {}
exec(compile(_b64e.b64decode("__B64__").decode("utf-8"), "franzen_effort.py", "exec"), _nse)
_ta_path = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
with open(_ta_path, "r", encoding="utf-8", newline="") as _fh:
    _src_before = _fh.read()
_src_after = _nse["apply_to_source"](_src_before)
assert _src_after != _src_before and "ARC3_STATIC_REASONING_EFFORT" in _src_after
compile(_src_after, str(_ta_path), "exec")
with open(_ta_path, "w", encoding="utf-8", newline="") as _fh:
    _fh.write(_src_after)
os.environ["ARC3_STATIC_REASONING_EFFORT"] = "__NIVEL__"
print("E1 reasoning_effort fijado a", os.environ["ARC3_STATIC_REASONING_EFFORT"], "en", _ta_path, flush=True)
'''


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    base, nivel, salida = ROOT / sys.argv[1], sys.argv[2], ROOT / sys.argv[3]
    if nivel not in ("medium", "low", "xhigh"):
        print("nivel invalido")
        return 1
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL.replace("__B64__", b64).replace("__NIVEL__", nivel)
    compile(cell_src, "<celda-e1>", "exec")

    nb = json.loads(base.read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"])
           if c.get("cell_type") == "code" and ANCHOR in "".join(c["source"])]
    if len(idx) != 1:
        print(f"esperaba 1 celda con el ancla del parche, hay {len(idx)}")
        return 1
    at = idx[0] + 1
    nb["cells"].insert(at, {"cell_type": "code", "execution_count": None, "metadata": {},
                            "outputs": [], "source": cell_src.splitlines(keepends=True)})
    salida.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    orig = json.loads(base.read_text(encoding="utf-8"))["cells"]
    new = json.loads(salida.read_text(encoding="utf-8"))["cells"]
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
    print(f"{salida.name}: celda en indice {at}, nivel={nivel}, {salida.stat().st_size} bytes")
    print("una sola celda anadida y las demas intactas:", "OK" if ok else "FALLA")
    print("modulo embebido exacto:", "OK" if fiel else "FALLA", "| todas compilan:", "OK" if not malas else malas)
    return 0 if ok and fiel and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

"""Genera notebooks/sheetu_effects.ipynb: sheetu_long + UNA celda con S1 (plan.md), el predictor
de efectos como funcion del sandbox (src/arc3/sandbox_effects.py).

Nada se narra en el prompt de cada turno: se anade `action_effects()` al sandbox y UNA linea de
documentacion en STRUCTURED_RUNTIME_STATE_ADDENDUM, junto a `valid_actions`.

La celda va DESPUES de la de ajustes (y por tanto despues de agentfix, que ya parcheo la
plantilla con F13). El parche es independiente del orden (scripts/test_sandbox_effects.py).

Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda anadida, ninguna otra tocada.

Uso:  python scripts/build_sheetu_effects.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "sheetu_long.ipynb"
OUT = ROOT / "notebooks" / "sheetu_effects.ipynb"
MOD = ROOT / "src" / "arc3" / "sandbox_effects.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="
MARCA = "# S1 PREDICTOR DE EFECTOS EN EL SANDBOX"

CELL_TEMPLATE = MARCA + ''' — kernel de EXPERIMENTO (plan.md, S1).
# "Frames yes, narration no": el predictor es una FUNCION del sandbox, no una nota por turno.
# action_effects() cuenta, por accion y por color clicado, cuantas veces cambio el tablero en el
# nivel actual (ignorando la barra de HUD). Una sola linea de documentacion en el system prompt.
try:
    import base64 as _b64x
    import inference.agent.python_tool_sandbox as _sbx
    import inference.agent.tool_agent as _tax
    _nsx = {}
    exec(compile(_b64x.b64decode("__B64__").decode("utf-8"), "sandbox_effects.py", "exec"), _nsx)
    _plantilla = _nsx["patch_sandbox_source"](_sbx._SANDBOX_BOOTSTRAP)
    compile(_plantilla, "<sandbox+s1>", "exec")
    _doc = _nsx["patch_addendum"](_tax.STRUCTURED_RUNTIME_STATE_ADDENDUM)
    _sbx._SANDBOX_BOOTSTRAP = _plantilla
    _tax.STRUCTURED_RUNTIME_STATE_ADDENDUM = _doc
    assert "_fx_action_effects" in _sbx._SANDBOX_BOOTSTRAP
    assert "self.frame_count" in _sbx._SANDBOX_BOOTSTRAP, "F13 de agentfix ya no esta en la plantilla"
    assert "action_effects()" in _tax._build_system_prompt(tool_output_tokens=1024)
    print("SANDBOX_EFFECTS installed: template+%d chars, system prompt carries action_effects()"
          % len(_nsx["HELPER_SRC"]), flush=True)
except Exception as exc:
    print("[sandbox_effects] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-s1>", "exec")

    nb = json.loads(SRC.read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"])
           if c.get("cell_type") == "code" and HOOK_ANCHOR in "".join(c["source"])]
    if len(idx) != 1:
        print(f"esperaba 1 celda con el ancla del hook, hay {len(idx)}")
        return 1
    at = idx[0] + 1
    nb["cells"].insert(at, {"cell_type": "code", "execution_count": None, "metadata": {},
                            "outputs": [], "source": cell_src.splitlines(keepends=True)})
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")

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
        try:
            compile("".join(c["source"]), f"<c{i}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {i}: {e}")
    fiel = base64.b64decode(b64) == MOD.read_bytes()
    print(f"celda insertada en indice {at} | tamano notebook {OUT.stat().st_size} bytes")
    print("una sola celda anadida y las demas intactas:", "OK" if ok else "FALLA")
    print("el modulo embebido se reconstruye exacto:", "OK" if fiel else "FALLA")
    print("todas las celdas compilan:", "OK" if not malas else malas)
    return 0 if ok and fiel and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

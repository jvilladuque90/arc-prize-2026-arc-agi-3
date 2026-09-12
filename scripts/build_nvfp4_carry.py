"""Genera notebooks/nvfp4_carry.ipynb: el kernel NVFP4 verbatim + UNA celda nuestra
con la consolidacion al ganar nivel (src/arc3/level_carry.py) en la costura C.

Kernel de EXPERIMENTO. El modulo va embebido en base64 y se ejecuta en un
namespace; el parche es sobre la clase ToolAgent del harness (aqui no hay
SchemaHelpersToolAgent: no hay injertos), y degrada al prompt del padre ante
cualquier error. COMPUERTA: exactamente una celda anadida en el hueco documentado
por el autor, ninguna otra tocada.
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_carry.ipynb"
MOD = ROOT / "src" / "arc3" / "level_carry.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# CONSOLIDACION AL GANAR NIVEL (nuestra) SOBRE EL STACK NVFP4 — kernel de EXPERIMENTO.
# El anfitrion detecta en el historial que accion subio cada nivel y que cambio en el
# tablero, y lo inyecta como texto desde el nivel 2. Cero escritura exigida al modelo,
# sin dependencia de la animacion (un nivel completado es inequivoco). Apunta al muro
# actual: v24 = 3.55 cierra el nivel 1 casi siempre y casi nunca el 2.
try:
    import base64 as _b64c
    import inference.agent.tool_agent as _tac
    _nsc = {}
    exec(compile(_b64c.b64decode("__B64__").decode("utf-8"), "level_carry.py", "exec"), _nsc)
    _transiciones = _nsc["transiciones_ganadoras"]
    _render_carry = _nsc["render_carry_note"]
    _orig_bup_c = _tac.ToolAgent._build_user_prompt

    def _bup_with_carry(self, action_num, **kw):
        base = _orig_bup_c(self, action_num, **kw)
        try:
            fr = kw.get("current_frame")
            nivel = int(getattr(fr, "level", 1) or 1) if fr is not None else 1
            nota = _render_carry(nivel, _transiciones(kw.get("history_entries") or []))
        except Exception:
            return base
        return base + "\\n" + nota if nota else base

    _tac.ToolAgent._build_user_prompt = _bup_with_carry
    print("LEVEL_CARRY injected on seam C:", len(_nsc), "symbols", flush=True)
except Exception as exc:
    print("[level_carry] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-carry>", "exec")

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
    print(f"celda insertada en indice {at} | tamano notebook {OUT.stat().st_size} bytes")
    print("una sola celda anadida y las demas intactas:", "OK" if ok else "FALLA")
    print("todas las celdas compilan:", "OK" if not malas else malas)
    return 0 if ok and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

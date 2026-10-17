"""Genera notebooks/nvfp4_manual.ipynb: el kernel NVFP4 verbatim + UNA celda nuestra con
el MANUAL DEL JUEGO (src/arc3/game_manual.py sobre src/arc3/level_carry.py) en la
costura C. Sustituye a la nota de consolidacion (la incluye y la amplia con la
localizacion del objeto ganador y las afordancias positivas del juego).

Kernel de EXPERIMENTO. Los dos modulos van embebidos en base64 y se ejecutan en el
mismo namespace (game_manual usa las funciones de level_carry por nombre); el parche
es sobre la clase ToolAgent; degrada al prompt del padre ante cualquier error nuestro.
COMPUERTA: exactamente una celda anadida en el hueco documentado por el autor.
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_manual.ipynb"
MOD_CARRY = ROOT / "src" / "arc3" / "level_carry.py"
MOD_MANUAL = ROOT / "src" / "arc3" / "game_manual.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# MANUAL DEL JUEGO (nuestro) SOBRE EL STACK NVFP4 — kernel de EXPERIMENTO.
# Amplia la consolidacion al ganar nivel (la unica palanca que paso la compuerta de
# 60 min: 26 niveles / 4.392 vs 23 / 4.114) con: donde esta AHORA el objeto del color
# que gano el nivel anterior (objetualidad + analogia estructural) y los controles y
# clicks con efecto PROBADO en este juego (solo positivos). Cero escritura exigida,
# texto solo desde el nivel 2, sin depender de la animacion.
try:
    import base64 as _b64m
    import inference.agent.tool_agent as _tam
    _nsm = {}
    exec(compile(_b64m.b64decode("__B64_CARRY__").decode("utf-8"), "level_carry.py", "exec"), _nsm)
    exec(compile(_b64m.b64decode("__B64_MANUAL__").decode("utf-8"), "game_manual.py", "exec"), _nsm)
    _construir_manual = _nsm["construir_manual"]
    _render_manual = _nsm["render_manual_note"]
    _orig_bup_m = _tam.ToolAgent._build_user_prompt

    def _bup_with_manual(self, action_num, **kw):
        base = _orig_bup_m(self, action_num, **kw)
        try:
            nota = _render_manual(_construir_manual(kw.get("history_entries") or [],
                                                    kw.get("current_frame")))
        except Exception:
            return base
        return base + "\\n" + nota if nota else base

    _tam.ToolAgent._build_user_prompt = _bup_with_manual
    print("GAME_MANUAL injected on seam C:", len(_nsm), "symbols", flush=True)
except Exception as exc:
    print("[game_manual] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64c = base64.b64encode(MOD_CARRY.read_bytes()).decode("ascii")
    b64m = base64.b64encode(MOD_MANUAL.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64_CARRY__", b64c).replace("__B64_MANUAL__", b64m)
    compile(cell_src, "<celda-manual>", "exec")

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

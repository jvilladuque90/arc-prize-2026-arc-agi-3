"""Genera notebooks/nvfp4_carrynow.ipynb: kernel NVFP4 verbatim + UNA celda nuestra con
la consolidacion v7 (src/arc3/level_carry_now.py) en la costura C.

Ultima hipotesis distinta del eje (DESIGN 8.64). El enganche de la nota mide 16,8 / 18,8 /
19,5 / 37,2 / 14,9 / 15,6% en v1 / v2 / v3 / v4 / v5 / v6: solo v4 se movio (14-2 contra
v1, p=0,0042). No fue el vocabulario (v5 lo conserva y pierde), ni la forma (v6 la restaura
y sigue perdido), ni la novedad (v5 y v6 son igual de nuevas y no movieron nada). Lo que
queda: el "despues" que v4 describia ERA EL TABLERO QUE EL MODELO MIRA, asi que la nota era
verificable contra su propia vista; v5 y v6 citan piezas del nivel anterior que ya no estan
en pantalla.

v7 resuelve la referencia contra `current_frame` y cita SOLO objetos que existen ahora,
diciendo donde estan. Si no se resuelve ninguno, nota vacia.

Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda anadida en el hueco documentado
por el autor, ninguna otra tocada.

Uso:  python scripts/build_nvfp4_carrynow.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_carrynow.ipynb"
MOD = ROOT / "src" / "arc3" / "level_carry_now.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# CONSOLIDACION v7 (ANCLADA EN LA VISTA ACTUAL) — kernel de EXPERIMENTO.
# Enganche de la nota medido en cuatro brazos: v1 16,8% | v2 18,8% | v3 19,5% | v4 37,2% |
# v5 14,9% | v6 15,6%. Solo v4 se movio (14-2 vs v1, p=0,0042). No fue el vocabulario (v5 lo
# conserva y pierde el efecto), ni la forma (v6 la restaura y sigue perdido), ni la novedad
# (v5 y v6 son igual de nuevas). Lo que queda con mecanismo: el "despues" que v4 describia ES
# EL TABLERO QUE EL MODELO ESTA MIRANDO, asi que podia verificar la nota contra su vista;
# v5/v6 citan piezas del nivel anterior (mediana 10 celdas) que ya no estan en pantalla.
# v7 segmenta current_frame y cita SOLO objetos que existen ahora, con su posicion actual.
# Si no se resuelve ninguno, nota vacia: no se manda al modelo a buscar lo que no hay.
try:
    import base64 as _b64w
    import inference.agent.tool_agent as _taw
    _nsw = {}
    exec(compile(_b64w.b64decode("__B64__").decode("utf-8"), "level_carry_now.py", "exec"), _nsw)
    _marcas_now = _nsw["marcas_victoria"]
    _render_now = _nsw["render_nota_actual"]
    _orig_bup_w = _taw.ToolAgent._build_user_prompt

    def _bup_with_now(self, action_num, **kw):
        base = _orig_bup_w(self, action_num, **kw)
        try:
            fr = kw.get("current_frame")
            nivel = int(getattr(fr, "level", 1) or 1) if fr is not None else 1
            nota = _render_now(nivel, _marcas_now(kw.get("history_entries") or []),
                               getattr(fr, "grid", None))
        except Exception:
            return base
        return base + "\\n" + nota if nota else base

    _taw.ToolAgent._build_user_prompt = _bup_with_now
    print("LEVEL_CARRY_NOW injected on seam C:", len(_nsw), "symbols", flush=True)
except Exception as exc:
    print("[level_carry_now] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-carrynow>", "exec")

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

    # --- compuertas ---
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
    # el modulo embebido debe reconstruirse byte a byte
    devuelto = base64.b64decode(b64)
    fiel = devuelto == MOD.read_bytes()
    print(f"celda insertada en indice {at} | tamano notebook {OUT.stat().st_size} bytes")
    print("una sola celda anadida y las demas intactas:", "OK" if ok else "FALLA")
    print("el modulo embebido se reconstruye exacto:", "OK" if fiel else "FALLA")
    print("todas las celdas compilan:", "OK" if not malas else malas)
    return 0 if ok and fiel and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

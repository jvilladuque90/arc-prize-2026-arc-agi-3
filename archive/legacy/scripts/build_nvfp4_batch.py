"""Genera notebooks/nvfp4_batch.ipynb: kernel NVFP4 verbatim + UNA celda nuestra con el
empuje al agrupamiento de acciones (src/arc3/action_budget.py) en la costura C.

BASE: `nvfp4.ipynb`, la del ENVIO, sin nota de consolidacion. El eje de la nota quedo
cerrado en DESIGN 8.65, asi que la comparacion limpia es contra `nvfp4_long` (sin nota:
23 niveles, media 4,114, 3,34 acciones por turno).

MOTIVO (DESIGN 8.66): la restriccion que manda son las ACCIONES, no el razonamiento. En
22 de 25 juegos es aritmeticamente imposible alcanzar el baseline de nivel 1+2 (46
acciones disponibles de mediana frente a 83 necesarias). El agente ya puede agrupar
—`action(['LEFT','LEFT','DOWN'])` es legal— con dispersion de 1,2 a 10,4 por turno.

Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda anadida en el hueco documentado
por el autor, ninguna otra tocada.

Uso:  python scripts/build_nvfp4_batch.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_batch.ipynb"
MOD = ROOT / "src" / "arc3" / "action_budget.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# PRESUPUESTO DE ACCIONES SOBRE EL STACK NVFP4 — kernel de EXPERIMENTO.
# DESIGN 8.66: la restriccion que manda son las ACCIONES. De 140 juegos que completan el
# nivel 1 y se atascan, el 76,4% APENAS INTENTO el siguiente (mediana 0,16x su baseline) y
# solo el 2,1% se perdio; el nivel 1 lo resuelven por debajo del baseline (0,73x). Las 225
# partidas acaban en 'cancelled': se acaba el reloj. Acciones disponibles por juego: 46 de
# mediana frente a un baseline de nivel 1+2 de 83 -> en 22 de 25 juegos es ARITMETICAMENTE
# imposible llegar. El agente ya agrupa acciones pero con dispersion de 1,2 a 10,4 por turno.
# La nota hace explicito el coste del turno (que el agente no puede ver) y SOLO aparece donde
# el ratio esta por debajo de 3,0: donde ya agrupa bien, calla. Asi no es texto de cada turno,
# que es lo unico que sabemos que no paga (8.41).
try:
    import base64 as _b64b
    import inference.agent.tool_agent as _tab
    _nsb = {}
    exec(compile(_b64b.b64decode("__B64__").decode("utf-8"), "action_budget.py", "exec"), _nsb)
    _render_presu = _nsb["render_nota_presupuesto"]
    _turnos_de = _nsb["turnos_de"]
    _orig_bup_b = _tab.ToolAgent._build_user_prompt

    def _bup_with_budget(self, action_num, **kw):
        base = _orig_bup_b(self, action_num, **kw)
        try:
            turnos = _turnos_de(self)
            acciones = len(kw.get("history_entries") or [])
            nota = _render_presu(acciones, turnos)
        except Exception:
            return base
        return base + "\\n" + nota if nota else base

    _tab.ToolAgent._build_user_prompt = _bup_with_budget
    print("ACTION_BUDGET injected on seam C:", len(_nsb), "symbols", flush=True)
except Exception as exc:
    print("[action_budget] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-batch>", "exec")

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
    fiel = base64.b64decode(b64) == MOD.read_bytes()
    print(f"celda insertada en indice {at} | tamano notebook {OUT.stat().st_size} bytes")
    print("una sola celda anadida y las demas intactas:", "OK" if ok else "FALLA")
    print("el modulo embebido se reconstruye exacto:", "OK" if fiel else "FALLA")
    print("todas las celdas compilan:", "OK" if not malas else malas)
    return 0 if ok and fiel and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

"""Genera notebooks/nvfp4_carryobj.ipynb: kernel NVFP4 verbatim + UNA celda nuestra con
la consolidacion v4 (src/arc3/level_carry_obj.py) en la costura C.

QUE CAMBIA RESPECTO A build_nvfp4_carry.py: solo el modulo inyectado. v1/v2/v3 nombraban
la ACCION ganadora; v4 describe los OBJETOS y la RELACION que cambio, en el mismo
vocabulario (y con los mismos `hash`) que el modelo ve en `current_frame.segmentation`.
Mismo presupuesto de tokens, misma costura, mismo disparador. Motivo y medidas en
DESIGN 8.61.

Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda anadida en el hueco documentado
por el autor, ninguna otra tocada.

Uso:  python scripts/build_nvfp4_carryobj.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_carryobj.ipynb"
MOD = ROOT / "src" / "arc3" / "level_carry_obj.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# CONSOLIDACION v4 (OBJETOS) SOBRE EL STACK NVFP4 — kernel de EXPERIMENTO.
# v1/v2/v3 nombraban la ACCION que gano el nivel. Medido sobre 722 turnos (DESIGN 8.61):
# el reuso de esa accion es 55.2% y 43.9% en dos replicas CON nota y 47.5% SIN nota (las
# replicas encierran al control), en 85-89% de los turnos el razonamiento no menciona la
# nota, y en 8 de 19 juegos la accion "ganadora" ya era >=80% de TODAS las acciones:
# informacion cero. El modelo, en cambio, consolida solo y en OBJETOS ("charcoal piece
# overlapped the yellow target"). v4 le habla en ese idioma, con los mismos hash que ve
# en current_frame.segmentation. Cambia el VOCABULARIO, no la cantidad.
try:
    import base64 as _b64o
    import inference.agent.tool_agent as _tao
    _nso = {}
    exec(compile(_b64o.b64decode("__B64__").decode("utf-8"), "level_carry_obj.py", "exec"), _nso)
    _tr_obj = _nso["transiciones_objeto"]
    _render_obj = _nso["render_nota_objetos"]
    _orig_bup_o = _tao.ToolAgent._build_user_prompt

    def _bup_with_obj(self, action_num, **kw):
        base = _orig_bup_o(self, action_num, **kw)
        try:
            fr = kw.get("current_frame")
            nivel = int(getattr(fr, "level", 1) or 1) if fr is not None else 1
            nota = _render_obj(nivel, _tr_obj(kw.get("history_entries") or []))
        except Exception:
            return base
        return base + "\\n" + nota if nota else base

    _tao.ToolAgent._build_user_prompt = _bup_with_obj
    print("LEVEL_CARRY_OBJ injected on seam C:", len(_nso), "symbols", flush=True)
except Exception as exc:
    print("[level_carry_obj] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-carryobj>", "exec")

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

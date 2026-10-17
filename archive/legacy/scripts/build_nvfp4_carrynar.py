"""Genera notebooks/nvfp4_carrynar.ipynb: kernel NVFP4 verbatim + UNA celda nuestra con
la consolidacion v6 (src/arc3/level_carry_nar.py) en la costura C.

v6 = el CONTENIDO de v5 con la FORMA de v4. Es la unica prueba que zanja DESIGN 8.63: v4
(contenido erroneo, forma narrativa) subio la mencion de la nota al 31,9% frente al 11-15%
de v1-v3; v5 (contenido correcto, forma de inventario, con salvedad epistemica) la hundio
al 9,4%, pareado 1-13 contra v4 (p=0,0018). Como v5 conserva el vocabulario de objetos, lo
que pagaba no era el vocabulario. v6 deja el contenido de v5 intacto -- el modulo se genero
sustituyendo SOLO la funcion que redacta, y hay test de equivalencia -- y le devuelve la
sintaxis de v4.

Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda anadida en el hueco documentado
por el autor, ninguna otra tocada.

Uso:  python scripts/build_nvfp4_carrynar.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_carrynar.ipynb"
MOD = ROOT / "src" / "arc3" / "level_carry_nar.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# CONSOLIDACION v6 (CONTENIDO DE v5, FORMA DE v4) — kernel de EXPERIMENTO.
# v4 (contenido erroneo, forma narrativa) subio la mencion de la nota en el razonamiento al
# 31,9% frente al 11-15% de v1-v3. v5 (contenido CORRECTO -- mediana 10 celdas en vez de 44,
# 6,6% de objetos >=100 celdas en vez de 38,9% -- pero forma de inventario y con salvedad
# epistemica) la hundio al 9,4%: pareado 1-13 de 14 juegos contra v4, p=0,0018. Como v5
# conserva el vocabulario de objetos, lo que pagaba en v4 no era el vocabulario. v6 deja el
# contenido de v5 BYTE A BYTE (el modulo se genero sustituyendo solo la funcion que redacta,
# con test de equivalencia) y le devuelve la sintaxis de v4: secuencia con sujeto, verbo y
# consecuencia, sin la salvedad, con cierre imperativo.
try:
    import base64 as _b64n
    import inference.agent.tool_agent as _tan
    _nsn = {}
    exec(compile(_b64n.b64decode("__B64__").decode("utf-8"), "level_carry_nar.py", "exec"), _nsn)
    _marcas_nar = _nsn["marcas_victoria"]
    _render_nar = _nsn["render_nota_narrativa"]
    _orig_bup_n = _tan.ToolAgent._build_user_prompt

    def _bup_with_nar(self, action_num, **kw):
        base = _orig_bup_n(self, action_num, **kw)
        try:
            fr = kw.get("current_frame")
            nivel = int(getattr(fr, "level", 1) or 1) if fr is not None else 1
            nota = _render_nar(nivel, _marcas_nar(kw.get("history_entries") or []))
        except Exception:
            return base
        return base + "\\n" + nota if nota else base

    _tan.ToolAgent._build_user_prompt = _bup_with_nar
    print("LEVEL_CARRY_NAR injected on seam C:", len(_nsn), "symbols", flush=True)
except Exception as exc:
    print("[level_carry_nar] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-carrynar>", "exec")

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

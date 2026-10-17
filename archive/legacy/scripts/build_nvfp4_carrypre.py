"""Genera notebooks/nvfp4_carrypre.ipynb: kernel NVFP4 verbatim + UNA celda nuestra con
la consolidacion v5 (src/arc3/level_carry_pre.py) en la costura C.

v4 (DESIGN 8.62) probo que el canal se abre escribiendo en OBJETOS: la mencion de la nota
en el razonamiento subio de 11-15% a 30,9% (pareado 13-3 de 16 juegos, p=0,0213). Pero
auditando su contenido aparecio un fallo presente DESDE v1: la transicion se marcaba con
el fotograma POSTERIOR a la accion, que ya es el primer tablero del nivel siguiente, asi
que la nota describia el REDIBUJADO DEL CAMBIO DE NIVEL (208 "desaparecio" contra 112 "se
movio"; 39% de los objetos citados con >=100 celdas, maximo 650 de 4096).

v5 describe lo unico observable y que transfiere: la PRECONDICION, leida SOLO en
fotogramas del nivel que se gana. Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda
anadida en el hueco documentado por el autor, ninguna otra tocada.

Uso:  python scripts/build_nvfp4_carrypre.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_carrypre.ipynb"
MOD = ROOT / "src" / "arc3" / "level_carry_pre.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# CONSOLIDACION v5 (PRECONDICION) SOBRE EL STACK NVFP4 — kernel de EXPERIMENTO.
# v4 abrio el canal (mencion de la nota 11-15% -> 30,9%, pareado 13-3, p=0,0213) pero su
# contenido era erroneo DESDE v1: la transicion se marcaba con el fotograma POSTERIOR a la
# accion, que ya es el primer tablero del nivel siguiente, asi que describia el redibujado
# del cambio de nivel (208 "desaparecio" vs 112 "se movio"; 39% de objetos de >=100 celdas,
# max 650 de 4096). El efecto de la jugada es INOBSERVABLE: no hay fotograma intermedio.
# v5 cuenta la POSICION GANADORA leyendo SOLO fotogramas del nivel que se gana: que pieza
# respondia a las jugadas, sobre que se apunto, y con que estaba en contacto.
try:
    import base64 as _b64p
    import inference.agent.tool_agent as _tap
    _nsp = {}
    exec(compile(_b64p.b64decode("__B64__").decode("utf-8"), "level_carry_pre.py", "exec"), _nsp)
    _marcas_pre = _nsp["marcas_victoria"]
    _render_pre = _nsp["render_nota_previa"]
    _orig_bup_p = _tap.ToolAgent._build_user_prompt

    def _bup_with_pre(self, action_num, **kw):
        base = _orig_bup_p(self, action_num, **kw)
        try:
            fr = kw.get("current_frame")
            nivel = int(getattr(fr, "level", 1) or 1) if fr is not None else 1
            nota = _render_pre(nivel, _marcas_pre(kw.get("history_entries") or []))
        except Exception:
            return base
        return base + "\\n" + nota if nota else base

    _tap.ToolAgent._build_user_prompt = _bup_with_pre
    print("LEVEL_CARRY_PRE injected on seam C:", len(_nsp), "symbols", flush=True)
except Exception as exc:
    print("[level_carry_pre] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-carrypre>", "exec")

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

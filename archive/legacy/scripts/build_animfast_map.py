"""Genera notebooks/animfast_map.ipynb: animfast_long verbatim + UNA celda nuestra con el
mapa cognitivo (src/arc3/cognitive_map.py) en la costura C.

POR QUE AHORA (plan de AGENTS.md, punto 2): el mapa se construyo en ee4677a y se aparco en
DESIGN 8.44 porque la base NVFP4 no tenia guardia de no-ops ni senal de animacion, y sin
ellas solo quedaba el fotograma final (el error de sandbox_nav). ANIMFAST monta el stack
NVFP4 CON el solver anim (`hard_noop_guard=True`, `animation_awareness=True`): la condicion
que pedia 8.44 ya se cumple.

DIFERENCIA con el parche viejo (build_duck_notebook.py): aquel se enganchaba a
`taaf_grafts.schema_helpers.SchemaHelpersToolAgent`, que animfast NO carga. Aqui se engancha
a `inference.agent.tool_agent.ToolAgent`, la clase que animfast usa de verdad.

Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda anadida tras la de ajustes, ninguna
otra tocada.

Uso:  python scripts/build_animfast_map.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "animfast_long.ipynb"
OUT = ROOT / "notebooks" / "animfast_map.ipynb"
MOD = ROOT / "src" / "arc3" / "cognitive_map.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="
MARCA = "# MAPA COGNITIVO SOBRE ANIMFAST"

CELL_TEMPLATE = MARCA + ''' — kernel de EXPERIMENTO (AGENTS.md, punto 2).
# El anfitrion mantiene el grafo de estados del nivel (nodo = firma del tablero, arista =
# accion) y lo resume en el prompt: estados repetidos, acciones ya probadas desde AQUI y
# la frontera sin probar. El modelo no gasta ni un turno construyendolo.
# El registrador ENVUELVE el guardia de no-ops del harness anim: hereda su correccion de
# animacion (una accion que devolvio varios fotogramas NO es inerte aunque el tablero final
# sea identico) y delega en el el bloqueo duro sin tocarlo.
try:
    import base64 as _b64m
    import inference.agent.tool_agent as _tam
    import inference.agent.noop_guard as _ngm
    _nsm = {}
    exec(compile(_b64m.b64decode("__B64__").decode("utf-8"), "cognitive_map.py", "exec"), _nsm)
    _MapRecorder = _nsm["MapRecorder"]
    _render_map_note = _nsm["render_map_note"]
    _orig_ensure_m = _tam.ToolAgent._ensure_session
    _orig_bup_m = _tam.ToolAgent._build_user_prompt

    def _ensure_with_map(self, state_path):
        _orig_ensure_m(self, state_path)
        try:
            g = getattr(self, "_noop_guard", None)
            if g is not None and not isinstance(g, _MapRecorder):
                self._noop_guard = _MapRecorder(g)
        except Exception:
            pass

    def _bup_with_map(self, action_num, **kw):
        base = _orig_bup_m(self, action_num, **kw)
        try:
            rec = getattr(self, "_noop_guard", None)
            if not isinstance(rec, _MapRecorder):
                return base
            fr = kw.get("current_frame")
            nivel = fr.level if fr is not None else 1
            firma = _ngm.board_signature(fr.grid) if fr is not None else None
            nota = _render_map_note(rec.registros, nivel, firma, kw.get("valid_actions"))
        except Exception:
            return base
        return base + "\\n" + nota if nota else base

    _tam.ToolAgent._ensure_session = _ensure_with_map
    _tam.ToolAgent._build_user_prompt = _bup_with_map
    print("COGNITIVE_MAP injected on seam C:", len(_nsm), "symbols", _tam.__file__, flush=True)
except Exception as exc:
    print("[cognitive_map] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-mapa>", "exec")

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

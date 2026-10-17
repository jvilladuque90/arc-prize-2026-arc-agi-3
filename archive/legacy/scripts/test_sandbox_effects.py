"""Tests de S1 (src/arc3/sandbox_effects.py). CPU, cero cuota.

1. Unidad: el helper, ejecutado tal cual vivira en el sandbox, sobre un nivel sintetico con barra
   de HUD, un movimiento util, uno inerte y clics sobre dos colores.
2. Composicion: el parche se aplica sobre la plantilla REAL del bundle NVFP4 y sobre la misma ya
   parcheada por F13 (agentfix_anim de sheetu12b), en ambos ordenes, y todo compila.
3. Extremo a extremo: se lanza el sandbox real con la plantilla parcheada y el codigo del modelo
   llama a `action_effects()` (se omite, avisandolo, si el sandbox no arranca en esta plataforma).

Uso:  python scripts/test_sandbox_effects.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "_tmp_nvfp4_bundle" / "src" / "ARC3-Inference"))

from arc3 import sandbox_effects as fx  # noqa: E402

fallos: list[str] = []


def check(ok: bool, nombre: str, detalle: str = "") -> None:
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def tablero(jugador_col: int, hud: int, extra: dict | None = None, nivel: int = 1):
    filas = [["." for _ in range(10)] for _ in range(6)]
    for c in range(hud):
        filas[0][c] = "g"                 # barra de HUD: crece una celda por accion
    filas[3][jugador_col] = "b"          # jugador
    filas[5][5] = "r"                    # objeto clicable
    for (r, c), ch in (extra or {}).items():
        filas[r][c] = ch
    asc = "\n".join("".join(f) for f in filas)
    return types.SimpleNamespace(ascii=asc, level=nivel, _grid=[list(f) for f in filas],
                                 step=0, shape=(6, 10))


def transicion(accion, antes, despues):
    return types.SimpleNamespace(action=accion, before_frame=antes, after_frame=despues)


def unidad() -> None:
    print("1. UNIDAD DEL HELPER")
    ns: dict = {}
    exec(compile(fx.HELPER_SRC, "<helper>", "exec"), ns)
    f = ns["_fx_action_effects"]

    s = [tablero(1, 0)]
    trans = []
    # RIGHT mueve al jugador (util), LEFT contra la pared... aqui: LEFT no mueve (inerte, solo HUD)
    plan = [("RIGHT", 2), ("LEFT_NOOP", None), ("RIGHT", 3), ("LEFT_NOOP", None),
            ("MOUSE(row=1, col=1)", None), ("MOUSE(row=2, col=4)", None),
            ("MOUSE(row=5, col=5)", "click_r")]
    col, hud = 1, 0
    for accion, efecto in plan:
        hud += 1
        antes = s[-1]
        extra = None
        if isinstance(efecto, int):
            col = efecto
        if efecto == "click_r":
            extra = {(5, 5): "y"}
        despues = tablero(col, hud, extra)
        nombre = "LEFT" if accion == "LEFT_NOOP" else accion
        trans.append(transicion(nombre, antes, despues))
        s.append(despues)

    validas = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"]
    info = f(trans, validas, s[-1])
    print("     ", json.dumps(info, ensure_ascii=False))
    check(info["hud_edge_lines_ignored"] == ["r0"], "detecta la barra de HUD (fila 0) y la ignora", str(info["hud_edge_lines_ignored"]))
    check(info["actions"]["RIGHT"]["changed"] == 2, "RIGHT cambio el tablero 2 de 2")
    check(info["actions"]["LEFT"]["changed"] == 0, "LEFT solo movio el HUD: 0 cambios reales")
    check(info["clicks_by_color"]["."]["changed"] == 0 and info["clicks_by_color"]["."]["tried"] == 2,
          "clics sobre fondo: 2 probados, 0 cambios")
    check(info["clicks_by_color"]["r"]["changed"] == 1, "clic sobre 'r' cambio el tablero")
    check(info["untested"] == ["UP", "DOWN"], "untested en nombres de modelo", str(info["untested"]))
    check("b" in info["untested_click_colors"], "colores presentes sin clicar ('b')",
          str(info["untested_click_colors"]))
    p_fondo = f(trans, validas, s[-1], {"action": "MOUSE", "row": 4, "col": 0})
    p_r = f(trans, validas, s[-1], "MOUSE(row=5, col=5)")
    check(p_fondo < 0.3, "un clic NUEVO sobre fondo se predice inerte", str(p_fondo))
    check(p_r is not None, "clic sobre color ya probado devuelve probabilidad", str(p_r))
    check(f(trans, validas, s[-1], "UP") == 0.5, "accion sin datos: 0,5 (sin informacion)")
    check(f(trans, validas, s[-1], "ACTION4") > 0.6, "acepta nombres de motor (ACTION4 = RIGHT)")

    # otro nivel: nada del nivel 1 contamina
    otro = tablero(1, 0, nivel=2)
    info2 = f(trans, validas, otro)
    check(info2["transitions_this_level"] == 0 and info2["actions"] == {},
          "cambio de nivel: empieza vacio")
    check(isinstance(f([], [], None), dict), "sin datos no revienta")


def _anim_patcher():
    nb = json.loads((ROOT / "notebooks" / "sheetu.ipynb").read_text(encoding="utf-8"))
    s = "".join(nb["cells"][10]["source"])
    i = s.find("_ANIM_SB_EDITS = (")
    j = s.find("# The kernel inlines every agentfix module")
    ns: dict = {}
    exec(s[i:j], ns)
    return ns["_anim_patch_sandbox_source"]


def composicion() -> str:
    print("2. COMPOSICION CON LA PLANTILLA REAL Y CON F13")
    import inference.agent.python_tool_sandbox as sb
    base = sb._SANDBOX_BOOTSTRAP
    anim = _anim_patcher()
    a = fx.patch_sandbox_source(base)
    compile(a, "<s1>", "exec")
    check(True, "S1 sobre la plantilla stock compila")
    b = fx.patch_sandbox_source(anim(base))
    compile(b, "<f13+s1>", "exec")
    check(True, "S1 sobre F13 compila")
    c = anim(fx.patch_sandbox_source(base))
    compile(c, "<s1+f13>", "exec")
    check(True, "F13 sobre S1 compila (orden indiferente)")
    check(b == c, "ambos ordenes dan el mismo programa")
    check(fx.patch_sandbox_source(b) == b, "idempotente")
    try:
        fx.patch_sandbox_source(base.replace("def _json_safe(value):", "def _x(value):"))
        check(False, "sin ancla lanza")
    except RuntimeError:
        check(True, "sin ancla lanza y no toca nada")
    import inference.agent.prompts as pr
    doc = fx.patch_addendum(pr.STRUCTURED_RUNTIME_STATE_ADDENDUM)
    check(doc.count("action_effects()") == 1 and fx.patch_addendum(doc) == doc,
          "la documentacion se anade una vez")
    check(doc.index("action_effects()") > doc.index("`valid_actions` is the current list"),
          "la linea va junto a valid_actions")
    return b


def extremo_a_extremo(programa: str) -> None:
    print("3. EXTREMO A EXTREMO EN EL SANDBOX REAL")
    import inference.agent.python_tool_sandbox as sb
    sb._SANDBOX_BOOTSTRAP = programa
    f0 = tablero(1, 0)
    f1 = tablero(2, 1)
    f2 = tablero(2, 2)

    def fp(fr):
        return {"ascii": fr.ascii, "step": 0, "level": fr.level, "shape": [6, 10], "grid": fr._grid}

    estado = {
        "current_frame": fp(f2),
        "history": [{"action": "", "frame": fp(f0)}, {"action": "RIGHT", "frame": fp(f1)},
                    {"action": "LEFT", "frame": fp(f2)}],
        "last_action_result": {},
        "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
    }
    codigo = "e = action_effects()\nprint(e['actions'], e['untested'], action_effects('LEFT'))"
    try:
        out = sb.run_sandboxed_python(code=codigo, timeout_seconds=20, initial_state=estado,
                                      action_handler=lambda acts: {})
    except Exception as exc:  # noqa: BLE001
        print(f"     (omitido: el sandbox no arranca aqui: {type(exc).__name__}: {exc})")
        return
    err = out.get("error")
    if err and ("resource" in err or "fork" in err or "preexec" in err):
        print(f"     (omitido: el sandbox depende de POSIX: {err[:120]})")
        return
    print("     ", out.get("stdout", "").strip()[:300], err or "")
    check(not err and "'RIGHT'" in out.get("stdout", "") and "['UP', 'DOWN']" in out.get("stdout", ""),
          "el modelo llama action_effects() dentro del sandbox real")


def main() -> int:
    unidad()
    programa = composicion()
    extremo_a_extremo(programa)
    print(f"\n{'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(main())

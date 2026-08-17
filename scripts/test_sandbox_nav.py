"""Test local (CPU, gratis) de los helpers de navegacion antes de gastar GPU.

Verifica lo que puede fallar en produccion:
  1. el prelude compila y solo usa nombres de SAFE_BUILTINS (se ejecuta con builtins
     RESTRINGIDOS, igual que el sandbox real);
  2. aprende el modelo de movimiento de transiciones simuladas;
  3. descarta acciones erraticas y las que no mueven nada;
  4. plan_moves devuelve una ruta correcta y respeta el limite;
  5. degrada a vacio sin transiciones (nunca revienta el turno del agente).

Uso: python scripts/test_sandbox_nav.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3.sandbox_nav import NAV_PROMPT_NOTE, build_nav_prelude  # noqa: E402

SAFE_BUILTIN_NAMES = (
    "abs all any ascii bin bool bytearray bytes callable chr complex dict dir divmod "
    "enumerate Exception filter float format frozenset getattr hasattr hash hex int "
    "isinstance issubclass iter len list map max min next oct ord pow print range repr "
    "reversed round set slice sorted str sum tuple TypeError type ValueError "
    "RuntimeError zip"
).split()


class Frame:
    def __init__(self, grid):
        self._grid = grid


class Transition:
    def __init__(self, action, before, after):
        self.action = action
        self.before_frame = Frame(before)
        self.after_frame = Frame(after)


def blank(n=8):
    return [[0] * n for _ in range(n)]


def with_player(r, c, color=3, n=8):
    g = blank(n)
    g[r][c] = color
    return g


def main() -> int:
    src = build_nav_prelude()
    print(f"prelude: {len(src)} chars, {len(src.splitlines())} lineas")

    # Sandbox restringido: solo SAFE_BUILTINS. Si el codigo usa algo fuera de la
    # lista (p.ej. KeyError, sum con key, __import__), esto revienta aqui y no en la G4.
    import builtins

    safe = {}
    for name in SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    env = {"__builtins__": safe}
    exec(compile(src, "<nav_prelude>", "exec"), env)  # noqa: S102 — es el punto del test
    print("compila y ejecuta bajo builtins restringidos: OK")

    fails = []

    # (2)+(3) modelo de movimiento: UP consistente, NOISE erratica, DEAD sin efecto
    trans = []
    for i in range(3):
        trans.append(Transition("UP", with_player(4 + i, 2), with_player(3 + i, 2)))
    for i in range(2):
        trans.append(Transition("RIGHT", with_player(2, 1 + i), with_player(2, 2 + i)))
    trans.append(Transition("NOISE", with_player(1, 1), with_player(5, 5)))
    trans.append(Transition("NOISE", with_player(1, 1), with_player(1, 6)))
    trans.append(Transition("NOISE", with_player(2, 2), with_player(7, 0)))
    trans.append(Transition("DEAD", with_player(3, 3), with_player(3, 3)))
    env["transitions"] = trans

    model = env["motion_model"]()
    print(f"motion_model: {model}")
    if model.get("UP") != [-1, 0]:
        fails.append(f"UP mal aprendido: {model.get('UP')}")
    if model.get("RIGHT") != [0, 1]:
        fails.append(f"RIGHT mal aprendido: {model.get('RIGHT')}")
    if "DEAD" in model:
        fails.append("DEAD (sin efecto) no fue descartada")
    if "NOISE" in model:
        fails.append("NOISE (erratica) no fue descartada")

    # (4) planificacion
    moves = env["plan_moves"](-3, 2)
    print(f"plan_moves(-3, 2) -> {moves}")
    if moves.count("UP") != 3 or moves.count("RIGHT") != 2 or len(moves) != 5:
        fails.append(f"ruta incorrecta: {moves}")
    capped = env["plan_moves"](-50, 0, 6)
    if len(capped) != 6:
        fails.append(f"limite no respetado: {len(capped)}")
    unreachable = env["plan_moves"](0, -5)  # no hay accion que vaya a la izquierda
    if unreachable:
        fails.append(f"deberia no encontrar ruta: {unreachable}")

    # posicion del objeto movil
    pos = env["player_pos"]()
    print(f"player_pos: {pos}")
    if pos is None:
        fails.append("player_pos no encontro el objeto movil")

    # (5) degradacion sin transiciones
    env["transitions"] = []
    if env["motion_model"]() != {} or env["plan_moves"](1, 1) != []:
        fails.append("no degrada a vacio sin transiciones")
    if env["player_pos"]() is not None:
        fails.append("player_pos deberia ser None sin transiciones")

    print(f"\nnota del prompt ({len(NAV_PROMPT_NOTE)} chars):\n  {NAV_PROMPT_NOTE}")
    print("\n" + ("FALLOS: " + "; ".join(fails) if fails else "TODO OK — apto para inyectar"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

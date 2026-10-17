"""Prueba el parche del seam C EXTRAYENDOLO DEL NOTEBOOK YA GENERADO.

Por que del notebook y no de una copia: el parche se construye con una f-string
llena de llaves y comillas escapadas. Un fallo de escapado no se ve leyendo el
codigo fuente, solo en el texto final. Probar una reimplementacion "equivalente"
daria verde mientras el kernel corre otra cosa — y eso solo se descubriria
gastando un envio.

Verifica cuatro cosas:
  1. el parche compila y se aplica sobre una clase de prueba
  2. con historial de un juego REAL, el prompt sale con la nota anexada
  3. sin historial, el prompt queda EXACTAMENTE igual al del padre
  4. si el calculo revienta, degrada al prompt del padre (no propaga la excepcion)

Uso: python scripts/test_seam_c.py [--notebook notebooks/duck_effects.ipynb]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE_PROMPT = "PROMPT DEL PADRE"


class _F:
    def __init__(self, grid):
        self.grid = grid


class _E:
    def __init__(self, action, grid):
        self.action = action
        self.frame = _F(grid)


class _StubAgent:
    def _build_user_prompt(self, action_num, **kw):
        return BASE_PROMPT


def extract_patch(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "EFFECTS_NOTE injected on seam C" in src:
            start = src.index("try:\n    import base64 as _b64")
            end = src.index("[effects_note] injection failed")
            end = src.index("\n", src.index("\n", end) + 1) + 1
            return src[start:end]
    raise SystemExit("no encontre el parche del seam C en el notebook")


def real_history(game_name: str, steps: int = 40):
    from arc3.env import LocalGame, discover_environments
    from arcengine import GameAction
    infos = {i.game_id.split("-")[0]: i
             for i in discover_environments(ROOT / "environment_files")}
    game = LocalGame(infos[game_name])

    def grid_of(fr):
        f = getattr(fr, "frame", None)
        return None if f is None or len(f) == 0 else [[int(v) for v in r] for r in f[-1]]

    grid = grid_of(game.reset())
    entries = [_E(None, grid)]
    rng = random.Random(0)
    for _ in range(steps):
        a = rng.choice(["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"])
        nxt = grid_of(game.step(getattr(GameAction, a)))
        if nxt is None:
            break
        entries.append(_E(a, nxt))
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="notebooks/duck_effects.ipynb")
    ap.add_argument("--games", nargs="+", default=["tu93", "vc33", "re86", "cn04"])
    args = ap.parse_args()

    patch = extract_patch(ROOT / args.notebook)
    print(f"parche extraido: {len(patch)} chars")

    # modulo falso 'taaf_grafts.schema_helpers' para que el parche lo encuentre
    pkg = types.ModuleType("taaf_grafts")
    mod = types.ModuleType("taaf_grafts.schema_helpers")
    mod.SchemaHelpersToolAgent = _StubAgent
    pkg.schema_helpers = mod
    sys.modules["taaf_grafts"] = pkg
    sys.modules["taaf_grafts.schema_helpers"] = mod

    ns: dict = {}
    exec(compile(patch, "<seam_c_patch>", "exec"), ns)
    if _StubAgent._build_user_prompt.__name__ != "_bup_with_effects":
        print("FALLO 1: el parche no reemplazo _build_user_prompt")
        return 1
    print("1. parche aplicado sobre la clase")

    agent = _StubAgent()

    # 3. sin historial -> prompt del padre intacto
    out = agent._build_user_prompt(0, valid_actions=None, history_entries=None)
    if out != BASE_PROMPT:
        print(f"FALLO 3: sin historial deberia quedar intacto, salio {out[:120]!r}")
        return 1
    print("3. sin historial: prompt del padre intacto")

    # 4. historial corrupto -> degrada, no revienta
    out = agent._build_user_prompt(0, valid_actions=None,
                                   history_entries=[object(), object()])
    if out != BASE_PROMPT:
        print("FALLO 4: con historial corrupto deberia degradar al prompt del padre")
        return 1
    print("4. historial corrupto: degrada sin propagar excepcion")

    # 2. historial real -> nota anexada
    ok = 0
    for g in args.games:
        out = agent._build_user_prompt(0, valid_actions=None,
                                       history_entries=real_history(g))
        anexo = out[len(BASE_PROMPT):].strip()
        if not out.startswith(BASE_PROMPT) or not anexo:
            print(f"   {g}: NOTA VACIA  <-- el turno no aportaria nada")
            continue
        ok += 1
        primera = anexo.splitlines()[1] if len(anexo.splitlines()) > 1 else anexo
        print(f"   {g}: +{len(anexo)} chars | {primera.strip()[:88]}")
    print(f"2. historial real: nota no vacia en {ok}/{len(args.games)} juegos")
    return 0 if ok == len(args.games) else 1


if __name__ == "__main__":
    raise SystemExit(main())

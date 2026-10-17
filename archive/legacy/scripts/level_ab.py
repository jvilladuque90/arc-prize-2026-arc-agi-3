"""Banco de iteracion sobre NIVELES GANADOS a presupuesto fijo (CPU, gratis).

POR QUE ESTE BANCO. El banco micro mide comprension (¿entiende la mecanica?); el
envio diario mide niveles pero cuesta una noche y tiene sigma=0.12. Esto mide
**niveles**, que es el objetivo mismo, en minutos de CPU y sin ruido de servidor:
mismo juego, misma semilla, mismo presupuesto de acciones, y lo unico que cambia
es el injerto que se prueba.

LO QUE NO ES. El explorador NO es el agente de produccion (alli decide un 27B).
Un cambio que ayude aqui no se traslada automaticamente. Lo que este banco
responde es si el MECANISMO tiene valor sobre los juegos reales — si ni con la
informacion perfecta el explorador gana mas niveles, la idea esta muerta y no
hace falta gastar una noche de envio para saberlo.

Curva de referencia medida (explorador ciego, 25 juegos locales):
    100 acc -> 3 niveles | 1.000 -> 11 | 10.000 -> 24 | 40.000 -> 25
Produccion da ~94 acciones/juego, asi que el LLM es ~100x mas eficiente por
accion que la busqueda ciega. El presupuesto por defecto (3.000) esta en la zona
de mayor pendiente, que es donde una mejora se nota.

Uso:
  python scripts/level_ab.py --budget 3000                    # control vs firma
  python scripts/level_ab.py --budget 1000 --arms control     # solo el control
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3.agent import GraphExplorer  # noqa: E402
from arc3.env import LocalGame, discover_environments  # noqa: E402
from arc3.features import frame_to_grid  # noqa: E402


class SignatureExplorer(GraphExplorer):
    """Explorador + sesgo por FIRMA: tras la primera subida de nivel, prioriza
    clics sobre celdas del mismo color que la celda que gano el nivel anterior.

    Es la idea que quedo en inventario (DESIGN §8.17) tras fallar por prompt: la
    firma de la meta es 100% consistente entre niveles (12/12 medido en las
    trazas), pero el LLM no sabe usarla cuando se le dice. Aqui no se le dice a
    nadie: se usa algoritmicamente para reordenar los candidatos.
    """

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self._sig_color: int | None = None
        self._last_click: tuple[int, int] | None = None
        self._prev_levels = 0
        self._prev_grid: np.ndarray | None = None
        self.sig_hits = 0          # diagnostico: veces que se aplico el sesgo

    def choose(self, grid, state, levels, avail):
        # una subida de nivel justo despues de un clic revela la firma
        if (levels > self._prev_levels and self._last_click is not None
                and self._sig_color is None and self._prev_grid is not None):
            x, y = self._last_click
            if 0 <= y < self._prev_grid.shape[0] and 0 <= x < self._prev_grid.shape[1]:
                self._sig_color = int(self._prev_grid[y][x])
        self._prev_levels = levels
        self._prev_grid = grid
        aid, x, y = super().choose(grid, state, levels, avail)
        self._last_click = (x, y) if aid == 6 else None
        return aid, x, y

    def _click_candidates(self, grid):
        cands = super()._click_candidates(grid)
        if self._sig_color is None:
            return cands
        firma, resto = [], []
        h, w = grid.shape
        for c in cands:
            _, cx, cy = c
            col = int(grid[cy][cx]) if 0 <= cy < h and 0 <= cx < w else -1
            (firma if col == self._sig_color else resto).append(c)
        if firma:
            self.sig_hits += 1
        return firma + resto


ARMS = {"control": GraphExplorer, "firma": SignatureExplorer}


def play(info, cls, budget: int, seed: int = 0) -> dict:
    from arcengine import GameAction

    game = LocalGame(info, seed=seed)
    agent = cls(info.game_id, max_actions=budget + 10)
    frame = game.reset()
    if frame is None:
        return {"levels": 0, "actions": 0}
    levels, n = 0, 0
    t0 = time.time()
    while frame is not None and n < budget:
        try:
            aid, x, y = agent.choose(
                frame_to_grid(frame.frame), frame.state.value,
                frame.levels_completed, list(frame.available_actions or []))
        except Exception:
            break
        n += 1
        try:
            if aid == 0:
                frame = game.reset()
            else:
                act = GameAction.from_id(aid)
                frame = game.step(act, x, y) if aid == 6 else game.step(act)
        except Exception:
            break
        if frame is None:
            break
        levels = max(levels, int(getattr(frame, "levels_completed", 0) or 0))
        if getattr(frame, "state", None) is not None and frame.state.value == "WIN":
            break
    return {"levels": levels, "actions": n, "seconds": round(time.time() - t0, 1),
            "sig_hits": getattr(agent, "sig_hits", 0),
            "sig_color": getattr(agent, "_sig_color", None)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=["control", "firma"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="level_ab.json")
    args = ap.parse_args()

    infos = {i.game_id.split("-")[0]: i
             for i in discover_environments(ROOT / "environment_files")}
    names = args.games or sorted(infos)
    res: dict = {a: {} for a in args.arms}

    for name in names:
        if name not in infos:
            continue
        linea = f"  {name:6}"
        for arm in args.arms:
            tot = 0
            for s in range(args.seeds):
                r = play(infos[name], ARMS[arm], args.budget, seed=s)
                tot += r["levels"]
                if s == 0:
                    res[arm][name] = r
            linea += f" | {arm}={tot}"
            if arm == "firma" and res[arm][name].get("sig_color") is not None:
                linea += f" (firma={res[arm][name]['sig_color']})"
        print(linea, flush=True)

    print()
    for arm in args.arms:
        t = sum(v["levels"] for v in res[arm].values())
        print(f"{arm:8} TOTAL {t} niveles en {len(res[arm])} juegos "
              f"({t/max(1,len(res[arm])):.2f}/juego)")
    if len(args.arms) == 2:
        a, b = args.arms
        solo_a = [g for g in res[a] if res[a][g]["levels"] > res[b].get(g, {}).get("levels", 0)]
        solo_b = [g for g in res[b] if res[b][g]["levels"] > res[a].get(g, {}).get("levels", 0)]
        print(f"pareado: {b} gana en {len(solo_b)} juegos {solo_b}, "
              f"{a} gana en {len(solo_a)} juegos {solo_a}")
    (ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

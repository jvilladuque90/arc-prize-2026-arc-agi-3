"""H2: ¿la meta del nivel N+1 es predecible desde la transicion ganadora del nivel N?

HIPOTESIS (§8.35). H1 (vision) quedo refutada: mirar el tablero no basta para
indexar la meta. H2 propone que la meta se hereda: el objeto que el agente
alcanzo al ganar el nivel N tiene una firma visual (parche de colores) que
reaparece en el nivel N+1. Si un algoritmo barato (template matching) la
encuentra con hit rate alto, el host puede calcular celdas candidatas y
pasarselas a nav como objetivos — el eslabon que el modelo no logra solo.

VERDAD OBJETIVA. Para cada subida de nivel capturada en traces_goal*.json:
  - posicion del agente al ganar = celdas que cambiaron en la ULTIMA transicion
    de la ventana (el paso que lo dejo en posicion de gatillar la subida)
  - esa posicion ES la meta del nivel (definicion operativa: donde habia que estar)

EXPERIMENTO por cada par (nivel N, nivel N+1) del mismo juego:
  1. meta_N   = posicion extraida de la transicion ganadora de N
  2. firma    = parche PATCH x PATCH de start_grid(N) centrado en meta_N
                (el aspecto del objeto-meta al INICIO del nivel, antes de tocarlo)
  3. candidatos = posiciones de start_grid(N+1) rankeadas por coincidencia con la firma
  4. verdad   = meta_{N+1} extraida de la transicion ganadora de N+1
  5. exito    = verdad dentro de radio TOL de algun candidato top-k

CONTROLES:
  - baseline azar: k celdas uniformes; baseline no-fondo: k celdas de color != moda
  - consistencia: la misma (juego, nivel) aparece en 2 corridas independientes
    (traces_goal.json y traces_goal_3.json); si las metas extraidas no coinciden,
    la extraccion no es confiable y el resto no vale.

Uso: python scripts/h2_goal_transfer.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = ["traces_goal.json", "traces_goal_2.json", "traces_goal_3.json"]
PATCH = 5          # lado del parche-firma
TOL = 3            # radio (chebyshev) para contar acierto
KS = (1, 3, 5)     # top-k evaluados
RNG = np.random.default_rng(7)
N_BASE = 200       # remuestreos del baseline


def goal_pos(event) -> tuple[int, int] | None:
    """Posicion de la meta.

    - trigger MOUSE(r,c): la meta es el click ganador (capture_traces guarda (y,x)).
    - trigger ACTION*: la meta es donde quedo el agente al gatillar la subida =
      centroide del ultimo diff PEQUENO antes de pre_grid. pre_grid vive en
      window[-2] (window[-1] ya es la pantalla del nivel nuevo); los diffs
      grandes intermedios son animaciones y se saltan.
    """
    trig = event["trigger"]
    if trig.startswith("MOUSE("):
        r, c = trig[6:-1].split(",")
        return int(r), int(c)
    w = event["window"]
    pre = np.array(event["pre_grid"])
    pre_idx = next((i for i in range(len(w) - 1, -1, -1)
                    if np.array_equal(np.array(w[i][1]), pre)), None)
    if pre_idx is None or pre_idx == 0:
        return None
    for i in range(pre_idx, 0, -1):
        a = np.array(w[i - 1][1])
        b = np.array(w[i][1])
        diff = np.argwhere(a != b)
        if 0 < len(diff) <= 60:
            r, c = diff.mean(axis=0)
            return int(round(r)), int(round(c))
    return None


def patch_at(grid: np.ndarray, pos: tuple[int, int]) -> np.ndarray:
    h = PATCH // 2
    r0, c0 = pos
    out = np.full((PATCH, PATCH), -1, dtype=int)
    for dr in range(-h, h + 1):
        for dc in range(-h, h + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                out[dr + h, dc + h] = grid[r, c]
    return out


def candidates(grid: np.ndarray, patch: np.ndarray, topk: int) -> list[tuple[int, int]]:
    """Posiciones rankeadas por coincidencia con el parche, ponderando cada
    celda por la rareza de su color en el tablero destino (el fondo no aporta:
    sin esto el matcher degenera y elige esquinas). Supresion local entre picks."""
    h = PATCH // 2
    H, W = grid.shape
    freq = np.bincount(grid.ravel(), minlength=16).astype(float) / grid.size
    w_color = -np.log(np.clip(freq, 1e-6, 1.0))
    w_color[np.argmax(freq)] = 0.0          # el color de fondo no puntua
    score = np.zeros((H, W))
    mask = patch != -1
    for r in range(h, H - h):
        for c in range(h, W - h):
            win = grid[r - h:r + h + 1, c - h:c + h + 1]
            m = (win == patch) & mask
            score[r, c] = float(np.sum(w_color[win[m]])) if m.any() else 0.0
    picks: list[tuple[int, int]] = []
    s = score.copy()
    for _ in range(topk):
        r, c = np.unravel_index(np.argmax(s), s.shape)
        if s[r, c] <= 0:
            break
        picks.append((int(r), int(c)))
        s[max(0, r - TOL):r + TOL + 1, max(0, c - TOL):c + TOL + 1] = -1
    return picks


def hit(picks, truth, k) -> bool:
    return any(max(abs(r - truth[0]), abs(c - truth[1])) <= TOL for r, c in picks[:k])


def main() -> None:
    # (juego, nivel) -> lista de {goal, start_grid, next disponible via nivel+1}
    by_game: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for fn in FILES:
        for g in json.load(open(ROOT / fn)):
            for e in g["events"]:
                pos = goal_pos(e)
                if pos is None:
                    continue
                by_game[g["game"]][e["level"]].append(
                    {"goal": pos, "start": np.array(e["start_grid"]), "src": fn})

    # 1) consistencia de extraccion entre corridas independientes
    cons_ok = cons_tot = 0
    for game, lv in by_game.items():
        for level, obs in lv.items():
            srcs = {o["src"] for o in obs}
            if len(srcs) >= 2:
                cons_tot += 1
                ps = [o["goal"] for o in obs]
                d = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a in ps for b in ps)
                if d <= TOL:
                    cons_ok += 1
                else:
                    print(f"  [inconsistente] {game} nivel {level}: metas {ps}")
    print(f"consistencia extraccion (misma meta en corridas distintas, tol {TOL}): "
          f"{cons_ok}/{cons_tot}")

    # 2) transferencia N -> N+1
    hits = {k: 0 for k in KS}
    base_rand = {k: 0.0 for k in KS}
    base_fg = {k: 0.0 for k in KS}
    pairs = 0
    for game, lv in by_game.items():
        for level in sorted(lv):
            if level + 1 not in lv:
                continue
            src = lv[level][0]          # firma desde la primera corrida disponible
            dst = lv[level + 1][0]
            patch = patch_at(src["start"], src["goal"])
            truth = dst["goal"]
            grid = dst["start"]
            picks = candidates(grid, patch, max(KS))
            pairs += 1
            row = f"  {game} {level}->{level+1}: verdad={truth} picks={picks[:3]}"
            for k in KS:
                ok = hit(picks, truth, k)
                hits[k] += ok
                if k == max(KS):
                    row += f" top{k}={'HIT' if ok else 'miss'}"
            print(row)
            # baselines por par
            H, W = grid.shape
            fg = np.argwhere(grid != np.bincount(grid.ravel()).argmax())
            for k in KS:
                r = RNG.integers(0, H, (N_BASE, k)), RNG.integers(0, W, (N_BASE, k))
                base_rand[k] += np.mean([
                    any(max(abs(rr - truth[0]), abs(cc - truth[1])) <= TOL
                        for rr, cc in zip(r[0][i], r[1][i])) for i in range(N_BASE)])
                if len(fg):
                    idx = RNG.integers(0, len(fg), (N_BASE, k))
                    base_fg[k] += np.mean([
                        any(max(abs(fg[j][0] - truth[0]), abs(fg[j][1] - truth[1])) <= TOL
                            for j in idx[i]) for i in range(N_BASE)])

    print(f"\npares evaluados: {pairs}")
    for k in KS:
        print(f"top-{k}: firma {hits[k]}/{pairs} = {hits[k]/max(pairs,1)*100:.0f}%"
              f" | azar {base_rand[k]/max(pairs,1)*100:.0f}%"
              f" | azar-no-fondo {base_fg[k]/max(pairs,1)*100:.0f}%")


if __name__ == "__main__":
    main()

"""GraphExplorer: agente de exploración de grafo de estados para ARC-AGI-3.

Síntesis de lo mejor del leaderboard público (ver docs/STRATEGY.md):
  - Grafo de estados con hashing enmascarado (borde 3px + máscara de contador aprendida),
    BFS sobre el grafo aprendido para volver a nodos con acciones pendientes, y replay
    tras RESET aprovechando el determinismo de los juegos.  [estilo v47, LB 0.54]
  - Clicks por componentes conexas ordenadas por button-likeness (compacto+pequeño+color
    raro) + rejilla gruesa de cobertura; supresión "deadsig" de clases estructuralmente
    inertes con protección de clases alguna vez efectivas.   [estilo 2º milestone]
  - Orden de acciones simples por P(cambio) aprendida online; no-ops se hunden, no se podan.

Lógica pura sobre numpy: el runner (local o gateway) le pasa frames y ejecuta lo que elige.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np

from .features import connected_components

GRID = 64
BORDER = 3           # borde enmascarado del hash: HUD/contadores viven ahí
COUNTER_WARMUP = 12  # transiciones para aprender la máscara de contador
COUNTER_FRACTION = 0.8   # celda contador si cambia en >=80% de las transiciones
COUNTER_MAX_INTERIOR = 0.2  # la máscara aprendida no puede tapar >20% del interior
CLICK_CAP = 64       # candidatos de click por nodo
DEAD_K = 2           # clase de click muerta tras K usos inertes
MAX_EXHAUSTED_RESETS = 8
RESET_LOOP_BREAK = 20

# ids de acción: 0=RESET, 1..5 y 7 simples, 6=click(x,y)
SIMPLE_IDS = (1, 2, 3, 4, 5, 7)
RESET_KEY = (0, -1, -1)


class _Node:
    __slots__ = ("pending", "tried")

    def __init__(self) -> None:
        self.pending: deque[tuple[int, int, int]] = deque()
        self.tried: set[tuple[int, int, int]] = set()


class GraphExplorer:
    """Elige (action_id, x, y). El caller ejecuta y devuelve el frame en el próximo choose()."""

    def __init__(self, game_id: str = "", max_actions: int = 15000) -> None:
        self.game_id = game_id
        self.max_actions = max_actions
        self.actions_taken = 0

        self._nodes: dict[int, _Node] = {}
        self._edges: dict[tuple[int, tuple[int, int, int]], int] = {}
        self._adj: dict[int, list[tuple[tuple[int, int, int], int]]] = {}

        self._counter_counts = np.zeros((GRID, GRID), dtype=np.int32)
        self._counter_seen = 0
        self._counter_mask: Optional[np.ndarray] = None

        # stats por acción simple: [cambios, usos, nodos_nuevos]
        self._act_stats: dict[int, list[int]] = {a: [0, 0, 0] for a in SIMPLE_IDS}
        # deadsig por clase estructural de click (color, size, is_rect)
        self._dead_sigs: dict[tuple[int, int, bool], int] = {}
        self._eff_sigs: set[tuple[int, int, bool]] = set()

        self._last_key: Optional[int] = None
        self._last_action: Optional[tuple[int, int, int]] = None
        self._last_grid: Optional[np.ndarray] = None
        self._last_levels = 0
        self._replay: deque[tuple[int, int, int]] = deque()
        self._replay_target: Optional[int] = None
        self._exhausted_resets = 0
        self._consecutive_resets = 0
        self.done = False

    # ---------- hashing ----------

    def _mask(self) -> np.ndarray:
        m = np.zeros((GRID, GRID), dtype=bool)
        m[:BORDER, :] = m[-BORDER:, :] = m[:, :BORDER] = m[:, -BORDER:] = True
        if self._counter_mask is not None:
            m |= self._counter_mask
        return m

    def _key(self, grid: np.ndarray) -> int:
        g = grid.copy()
        g[self._mask()] = 0
        return hash(g.tobytes())

    def _learn_counter_mask(self, prev: np.ndarray, nxt: np.ndarray) -> None:
        if self._counter_mask is not None or prev is None:
            return
        self._counter_counts += prev != nxt
        self._counter_seen += 1
        if self._counter_seen >= COUNTER_WARMUP:
            cand = self._counter_counts >= COUNTER_FRACTION * self._counter_seen
            cand[:BORDER, :] = cand[-BORDER:, :] = cand[:, :BORDER] = cand[:, -BORDER:] = False
            interior = (GRID - 2 * BORDER) ** 2
            # si "todo cambia siempre" (animación global) la máscara sería inútil: borde solo
            self._counter_mask = cand if cand.sum() <= COUNTER_MAX_INTERIOR * interior \
                else np.zeros((GRID, GRID), dtype=bool)

    # ---------- candidatos ----------

    def _click_sig(self, obj: dict[str, Any]) -> tuple[int, int, bool]:
        y0, x0, y1, x1 = obj["bbox"]
        is_rect = obj["size"] == (y1 - y0 + 1) * (x1 - x0 + 1)
        return (obj["color"], obj["size"], is_rect)

    def _click_candidates(self, grid: np.ndarray) -> list[tuple[int, int, int]]:
        counts = np.bincount(grid.ravel(), minlength=16)
        background = int(counts.argmax())
        total = grid.size
        objs = connected_components(grid, background)
        scored = []
        for o in objs:
            sig = self._click_sig(o)
            if self._dead_sigs.get(sig, 0) >= DEAD_K and sig not in self._eff_sigs:
                continue
            rarity = 1.0 - counts[o["color"]] / total
            y0, x0, y1, x1 = o["bbox"]
            fill = o["size"] / ((y1 - y0 + 1) * (x1 - x0 + 1))
            size_score = 1.0 if o["size"] <= 4 else 0.8 if o["size"] <= 16 else \
                0.5 if o["size"] <= 64 else 0.25 if o["size"] <= 256 else 0.0
            score = 0.4 * rarity + 0.3 * size_score + 0.3 * fill
            cy, cx = o["centroid"]
            scored.append((score, int(round(cx)), int(round(cy))))
        scored.sort(key=lambda t: -t[0])
        cands = [(6, x, y) for _, x, y in scored[:CLICK_CAP]]
        # rejilla gruesa de cobertura (paso 8, centrada)
        for gy in range(4, GRID, 8):
            for gx in range(4, GRID, 8):
                if len(cands) >= CLICK_CAP:
                    break
                cands.append((6, gx, gy))
        return cands[:CLICK_CAP]

    def _simple_order(self, available: list[int]) -> list[int]:
        acts = [a for a in SIMPLE_IDS if not available or a in available]

        def score(a: int) -> float:
            chg, uses, _new = self._act_stats[a]
            return 0.5 if uses == 0 else chg / uses

        return sorted(acts, key=lambda a: -score(a))

    def _fill_pending(self, node: _Node, grid: np.ndarray, available: list[int]) -> None:
        for a in self._simple_order(available):
            k = (a, -1, -1)
            if k not in node.tried:
                node.pending.append(k)
        if not available or 6 in available:
            for k in self._click_candidates(grid):
                if k not in node.tried:
                    node.pending.append(k)

    # ---------- grafo ----------

    def _record_edge(self, src: int, action: tuple[int, int, int], dst: int) -> None:
        if (src, action) not in self._edges:
            self._edges[(src, action)] = dst
            self._adj.setdefault(src, []).append((action, dst))

    def _bfs_to_pending(self, start: int) -> Optional[list[tuple[int, int, int]]]:
        """Camino más corto (en aristas conocidas) hasta un nodo con pendientes."""
        seen = {start}
        q: deque[tuple[int, list[tuple[int, int, int]]]] = deque([(start, [])])
        while q:
            key, path = q.popleft()
            node = self._nodes.get(key)
            if node and node.pending and key != start:
                return path
            if len(path) >= 60:
                continue
            for action, dst in self._adj.get(key, []):
                if dst not in seen:
                    seen.add(dst)
                    q.append((dst, path + [action]))
        return None

    # ---------- API ----------

    def choose(
        self,
        grid: np.ndarray,
        state: str,
        levels_completed: int,
        available_actions: list[int],
    ) -> tuple[int, int, int]:
        """Devuelve (action_id, x, y); x=y=-1 para acciones simples/RESET."""
        self.actions_taken += 1
        if self.actions_taken > self.max_actions:
            self.done = True

        # --- digerir el resultado de la acción anterior ---
        if self._last_grid is not None and self._last_action is not None:
            changed = bool((self._last_grid != grid).any())
            self._learn_counter_mask(self._last_grid, grid)
            aid, ax, ay = self._last_action
            if aid in self._act_stats:
                self._act_stats[aid][1] += 1
                self._act_stats[aid][0] += int(changed)
                self._act_stats[aid][2] += int(self._key(grid) not in self._nodes)
            if aid == 6:
                sig = self._sig_at(self._last_grid, ax, ay)
                if sig is not None:
                    if changed or levels_completed != self._last_levels:
                        self._eff_sigs.add(sig)
                    else:
                        self._dead_sigs[sig] = self._dead_sigs.get(sig, 0) + 1

        if levels_completed > self._last_levels:
            # nivel nuevo: lo inerte de antes puede ser la clave ahora
            self._dead_sigs.clear()
            self._eff_sigs.clear()
            self._replay.clear()
            self._replay_target = None
        self._last_levels = levels_completed

        key = self._key(grid)
        if self._last_key is not None and self._last_action is not None:
            self._record_edge(self._last_key, self._last_action, key)

        # --- game over / not played ---
        if state in ("NOT_PLAYED", "GAME_OVER"):
            self._consecutive_resets += 1
            if self._consecutive_resets >= RESET_LOOP_BREAK:
                self.done = True
            return self._commit(grid, key, RESET_KEY)
        self._consecutive_resets = 0

        # --- replay en curso (verificando determinismo) ---
        if self._replay:
            if self._replay_target is not None and key != self._replay_target:
                self._replay.clear()  # el mundo no siguió el grafo: abortar replay
                self._replay_target = None
            else:
                action = self._replay.popleft()
                self._replay_target = self._edges.get((key, action))
                return self._commit(grid, key, action)

        node = self._nodes.get(key)
        if node is None:
            node = _Node()
            self._nodes[key] = node
            self._fill_pending(node, grid, available_actions)

        if node.pending:
            action = node.pending.popleft()
            node.tried.add(action)
            return self._commit(grid, key, action)

        # nodo agotado: BFS al nodo pendiente más cercano
        path = self._bfs_to_pending(key)
        if path:
            self._replay = deque(path)
            action = self._replay.popleft()
            self._replay_target = self._edges.get((key, action))
            return self._commit(grid, key, action)

        # grafo alcanzable agotado: RESET para reintentar desde el inicio del nivel
        self._exhausted_resets += 1
        if self._exhausted_resets >= MAX_EXHAUSTED_RESETS:
            self.done = True
        return self._commit(grid, key, RESET_KEY)

    def _sig_at(self, grid: np.ndarray, x: int, y: int) -> Optional[tuple[int, int, bool]]:
        counts = np.bincount(grid.ravel(), minlength=16)
        background = int(counts.argmax())
        if not (0 <= x < GRID and 0 <= y < GRID) or grid[y, x] == background:
            return None
        for o in connected_components(grid, background):
            y0, x0, y1, x1 = o["bbox"]
            if y0 <= y <= y1 and x0 <= x <= x1:
                return self._click_sig(o)
        return None

    def _commit(self, grid: np.ndarray, key: int, action: tuple[int, int, int]) -> tuple[int, int, int]:
        self._last_grid = grid.copy()
        self._last_key = key
        self._last_action = action
        return action

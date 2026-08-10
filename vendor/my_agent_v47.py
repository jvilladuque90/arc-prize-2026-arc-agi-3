# VENDORED (verbatim) from the public in-competition Kaggle notebook:
#   poby7722/versrion-47-is-bset-score-0-54  (hidden LB 0.54, CPU-only exploration)
# which itself ports techniques from Occam (github.com/g-baskin/occam, MIT) and the
# MIT-licensed 3rd-place 'just-explore' solution. Kept byte-faithful as our proven
# 0.54 baseline; our improvements go in subclasses/configs, never by editing this file.
# See docs/DESIGN.md and paper/working_note_es.md for the strategy context.

# ARC-AGI-3 explore2 MyAgent for Kaggle competition rerun.
# Frontier graph exploration + learned counter mask + effective-action ordering.
# Public-set config: default explore2 plus reorder disabled only for cn04.
import time

"""Graph-based exploration agent for ARC-AGI-3.

Strategy (no LLM, no GPU training - see PLAN.md):
  * Treat each distinct frame as a node in a per-level state graph.
  * Edge = (action) that transformed one frame into another.
  * From the current node, try its untried actions first. When a node is
    exhausted, BFS to the nearest node that still has untried actions and
    replay the path to get there ("frontier exploration").
  * For the complex click action (ACTION6) we do not enumerate all 64*64
    coordinates blindly: we segment the frame into same-color connected
    components and propose their centroids as click candidates, ordered by a
    crude "button-likeness" heuristic (small, compact, non-background blobs).

This is intentionally deterministic and cheap. It is the v1 baseline we will
iterate on (world-model, object abstraction) per PLAN.md.
"""

import logging
from collections import deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()


# v3: width of the border band masked out before hashing a frame into a state
# key. ARC-AGI-3 games render step counters / status bars along the edges; those
# change every step and would otherwise explode the state space with spurious
# nodes. Masking the border collapses them (technique ported from the MIT-
# licensed 3rd-place "just-explore" solution, identify_status_bars_crude).
STATUS_BAR_BORDER = 3


def _mask_borders(grid: list, border: int = STATUS_BAR_BORDER) -> tuple:
    """Return a hashable copy of a 2D grid with the outer `border` cells zeroed,
    so edge status bars don't create distinct states."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h <= 2 * border or w <= 2 * border:
        # too small to mask meaningfully; hash as-is
        return tuple(tuple(row) for row in grid)
    out = []
    for r in range(h):
        if r < border or r >= h - border:
            out.append((0,) * w)
        else:
            row = grid[r]
            out.append((0,) * border + tuple(row[border:w - border]) + (0,) * border)
    return tuple(out)


def _grid_key(frame: Optional[list]) -> tuple:
    """Hashable signature of the frame, with edge status bars masked out (v3)."""
    if not frame:
        return ()
    return tuple(_mask_borders(grid) for grid in frame)


def _last_grid(frame: Optional[list]) -> Optional[list]:
    if not frame:
        return None
    return frame[-1]


def _connected_components(grid: list) -> list:
    """4-connectivity same-color components. Returns list of dicts with
    color, size, bbox, and centroid (x=col, y=row)."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    comps: list = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc]:
                continue
            color = grid[sr][sc]
            q = deque([(sr, sc)])
            seen[sr][sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            rows = [c[0] for c in cells]
            cols = [c[1] for c in cells]
            comps.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "centroid": (sum(cols) // len(cells), sum(rows) // len(cells)),
                }
            )
    return comps


def _background_color(grid: list) -> int:
    """Most common color is assumed to be background."""
    counts: dict = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


import os as _os

# Env-tunable click breadth (defaults 64/8 = the verified public-set config).
_CLICK_LIMIT = int(_os.getenv("ARC_E2_CLICK_LIMIT", "64"))
_CLICK_STEP = int(_os.getenv("ARC_E2_CLICK_STEP", "8"))


def _click_candidates(grid: Optional[list], limit: int = _CLICK_LIMIT,
                      grid_step: int = _CLICK_STEP) -> list:
    """Propose (x, y) click points for ACTION6, ordered by priority.

    Single flat pool of non-background blob centroids, sorted by button-likeness
    (compact + small ranks first), then a coarse grid sweep for coverage.

    NOTE: an earlier v3 grouped blobs into 5 hard priority tiers (ported from
    the top-3 solution's frame_segments_to_action_groups). Benchmarks showed it
    REGRESSED: tn36 dropped 112->30 states and 1->0 levels (reproduced twice),
    because hard tiers force-exhaust the "button-like" group before trying
    interaction cells that don't match the button heuristic, so within a 600-step
    budget the agent never reaches them. The flat ordering still surfaces small
    rare-color buttons first (high likeness) without starving other cells, and
    matched or beat tiers on every game. Kept flat by evidence.

    Returns up to `limit` (x, y) = (col, row) points.
    """
    if not grid:
        return []
    h = len(grid)
    w = len(grid[0]) if h else 0
    bg = _background_color(grid)

    scored = []
    for comp in _connected_components(grid):
        if comp["color"] == bg:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        bbox_area = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
        fill = comp["size"] / bbox_area
        likeness = fill / (1.0 + comp["size"])  # compact + small = more button-like
        scored.append((likeness, comp["centroid"]))
    scored.sort(key=lambda t: t[0], reverse=True)

    out: list = []
    seen = set()
    for _, cxy in scored:
        if cxy not in seen:
            seen.add(cxy)
            out.append(cxy)

    # coarse grid sweep (offset by half-step to hit cell centres) for coverage
    half = max(1, grid_step // 2)
    for y in range(half, h, grid_step):
        for x in range(half, w, grid_step):
            p = (x, y)
            if p not in seen:
                seen.add(p)
                out.append(p)

    return out[:limit]


class Explore(Agent):
    """Frontier graph exploration agent."""

    # ls20 enforces an ~80-action episode budget server-side (GAME_OVER after),
    # so a high cap mainly matters for games that allow longer episodes / many
    # RESET-and-retry trajectories. Keep generous; the graph carries across resets.
    MAX_ACTIONS = 600

    # After this many resets with no newly-discovered state, treat the game as a
    # deterministic dead-end for this agent (avoids the ft09 reset loop).
    MAX_EXHAUSTED_RESETS = 8

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._nodes: dict = {}
        self._edges: dict = {}
        self._level_seen_reset = -1
        self._replay_queue: deque = deque()
        self._last_key: Optional[tuple] = None
        self._last_plan: Optional[Any] = None
        self._exhausted_resets = 0

    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        done = latest_frame.state is GameState.WIN
        # Lightweight diagnostics: log state changes + periodic heartbeat so we
        # can see WIN/GAME_OVER transitions (action log lines don't show state).
        st = getattr(latest_frame.state, "name", str(latest_frame.state))
        if st != getattr(self, "_last_logged_state", None):
            logger.info(
                "STATE -> %s (action %d, levels_completed=%d, nodes=%d)",
                st, self.action_counter, latest_frame.levels_completed, len(self._nodes),
            )
            self._last_logged_state = st
        return done

    @staticmethod
    def _as_action(a: Any) -> GameAction:
        """available_actions may arrive as ints (pydantic-coerced) or enums."""
        if isinstance(a, GameAction):
            return a
        return GameAction.from_id(a)

    def _state_key(self, frame: Optional[list]) -> tuple:
        """Hashable state signature for a frame. Default = static border mask.
        Subclasses (e.g. Explore2) override to apply a learned counter mask."""
        return _grid_key(frame)

    def _simple_actions(self, latest_frame: FrameData) -> list:
        avail = getattr(latest_frame, "available_actions", None)
        if avail:
            acts = [self._as_action(a) for a in avail]
            acts = [a for a in acts if a is not GameAction.RESET]
        else:
            acts = [a for a in GameAction if a is not GameAction.RESET]
        return acts

    def _build_plans(self, latest_frame: FrameData) -> list:
        """A plan is either a GameAction (simple) or ('CLICK', x, y) for ACTION6."""
        plans: list = []
        avail = self._simple_actions(latest_frame)
        for a in avail:
            # ACTION6 is the complex click action; handled separately below.
            # (GameAction enum members are shared/mutated by set_data, so
            #  a.is_simple() can flip True after a click — exclude explicitly.)
            if a is GameAction.ACTION6:
                continue
            if a.is_simple():
                plans.append(a)
        if any(a is GameAction.ACTION6 for a in avail):
            grid = _last_grid(latest_frame.frame)
            for (x, y) in _click_candidates(grid):
                plans.append(("CLICK", x, y))
        return plans

    def _plan_to_action(self, plan: Any) -> GameAction:
        if isinstance(plan, tuple) and plan and plan[0] == "CLICK":
            action = GameAction.ACTION6
            action.set_data({"x": plan[1], "y": plan[2]})
            action.reasoning = {"strategy": "explore-click", "x": plan[1], "y": plan[2]}
            return action
        action = plan
        action.reasoning = {"strategy": "explore-simple"}
        return action

    def _register(self, key: tuple, latest_frame: FrameData) -> None:
        if key not in self._nodes:
            self._nodes[key] = {"untried": self._build_plans(latest_frame)}
            self._edges.setdefault(key, [])
            # Discovering a new state means exploration is still productive --
            # clear the stuck counter so reset-loop detection only fires when we
            # truly stop finding anything new.
            self._exhausted_resets = 0

    def _bfs_to_frontier(self, start: tuple) -> Optional[list]:
        """Return plans leading from start to a node with untried plans."""
        q: deque = deque([(start, [])])
        visited = {start}
        while q:
            node, path = q.popleft()
            info = self._nodes.get(node)
            if info and info["untried"] and node != start:
                return path
            for plan, child in self._edges.get(node, []):
                if child not in visited:
                    visited.add(child)
                    q.append((child, path + [plan]))
        return None

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        state = latest_frame.state

        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._replay_queue.clear()
            self._last_key = None
            return GameAction.RESET

        if latest_frame.levels_completed != self._level_seen_reset:
            self._level_seen_reset = latest_frame.levels_completed

        key = self._state_key(latest_frame.frame)
        self._register(key, latest_frame)

        if self._last_key is not None and self._last_plan is not None:
            edges = self._edges.setdefault(self._last_key, [])
            if not any(p == self._last_plan and c == key for p, c in edges):
                edges.append((self._last_plan, key))

        if self._replay_queue:
            plan = self._replay_queue.popleft()
            self._last_key, self._last_plan = key, plan
            return self._plan_to_action(plan)

        info = self._nodes[key]
        if info["untried"]:
            plan = info["untried"].pop(0)
            self._last_key, self._last_plan = key, plan
            return self._plan_to_action(plan)

        path = self._bfs_to_frontier(key)
        if path:
            self._replay_queue = deque(path)
            plan = self._replay_queue.popleft()
            self._last_key, self._last_plan = key, plan
            return self._plan_to_action(plan)

        # Whole reachable graph exhausted. RESET is only useful if the episode
        # actually ended (GAME_OVER) or to retry a non-deterministic level. For
        # a deterministic NOT_FINISHED state, reset returns to the same explored
        # start -> infinite loop (this is exactly what trapped ft09 in v1).
        # Count these and stop fighting once it's clearly stuck.
        self._replay_queue.clear()
        self._last_key, self._last_plan = None, None
        self._exhausted_resets += 1
        if self._exhausted_resets == 1:
            logger.info(
                "Graph exhausted at %d states; resetting to retry.", len(self._nodes)
            )
        if self._exhausted_resets >= self.MAX_EXHAUSTED_RESETS:
            # Genuinely stuck: deterministic dead-end, no new states across many
            # resets. Keep returning RESET (harness/MAX_ACTIONS will end it) but
            # don't pretend we're exploring.
            if self._exhausted_resets == self.MAX_EXHAUSTED_RESETS:
                logger.info(
                    "Stuck: %d resets with no new states (%d total) -- "
                    "deterministic dead-end for this agent.",
                    self._exhausted_resets, len(self._nodes),
                )
        return GameAction.RESET


"""Explore2 — graph exploration with two techniques ported from Occam.

Occam (github.com/g-baskin/occam, MIT, "Sean Donahoe") uses the same reset-replay
graph-exploration core as our Explore but reaches 17/25 public games (57.6% RHAE,
CPU-only, no LLM) where the identical core gets 14/25. Two of its upgrades are the
named reasons, both pure-Python / no-training, reimplemented here in our own
architecture (the env API differs, so this is a port, not a copy):

  1. AUTO-LEARNED COUNTER/ANIMATION MASK (Occam: detect_counter_mask). Our Explore
     hard-masks a fixed 3-cell border to kill edge step-counters. But some games
     render the counter / an animation *inside* the scene, so every frame hashes
     unique and the state graph explodes -> BFS never goes deep. Occam ANDs the
     pixels that change on every consecutive transition over ~3 actions and zeros
     them before hashing. We do the robust online version: accumulate, over a
     warmup window, the cells that change in >= MASK_CHANGE_RATIO of transitions,
     then freeze that as the mask. Guarded: never mask more than MASK_MAX_FRACTION
     of the interior (else we'd collapse real game state -> keep border-only).

  2. EFFECTIVE-ACTION ORDERING (Occam: _discover_and_prune + action_effectiveness).
     Occam probes each action once from reset and hard-drops the ones that don't
     change the masked hash. We use the softer online form: track, per simple
     action, P(it changed the masked state), and when expanding a fresh node try
     historically-effective actions first (a disabled/no-op button sinks to the
     back). Softer than a hard one-shot prune because an action that is a no-op in
     one state may matter in another; ordering keeps it available but late.

Everything else (frontier BFS replay, click candidates, reset-loop breaker) is
inherited unchanged from Explore, so `explore` stays the A/B control. Offline
unit tests in tests/smoke_explore2.py drive the mask learner and action ranker
with hand-built frame sequences of known ground truth (no network, no game sim).
"""
import os
from typing import Any, Optional

from arcengine import FrameData, GameAction



class Explore2(Explore):
    """Explore + learned counter mask + effective-action ordering."""

    # Observe this many transitions before freezing the learned counter mask.
    MASK_WARMUP = 12
    # A cell is "counter/animation" if it changed in >= this fraction of the
    # observed transitions during warmup. Env-overridable for sweeps.
    MASK_CHANGE_RATIO = float(os.getenv("ARC_E2_MASK_RATIO", "0.8"))
    # Safety: never mask more than this fraction of interior cells. Env-overridable.
    MASK_MAX_FRACTION = float(os.getenv("ARC_E2_MASK_MAXFRAC", "0.20"))

    # Ablation toggles (env-overridable) so we can A/B which component helps on a
    # given game. Both default ON. ARC_E2_MASK=0 disables the counter mask;
    # ARC_E2_REORDER=0 disables effective-action ordering. Used to diagnose the
    # cn04 regression (explore2 lost a level explore had) without forking code.
    USE_MASK = os.getenv("ARC_E2_MASK", "1") != "0"
    USE_REORDER = os.getenv("ARC_E2_REORDER", "1") != "0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mask_cells: Optional[frozenset] = None  # frozen once learned
        self._change_counts: dict = {}                # (r,c) -> times changed
        self._transitions = 0
        self._prev_interior: Optional[tuple] = None   # border-masked prev grid
        self._prev_emitted: Optional[str] = None      # action name we last sent
        # effectiveness: action name -> [n_changed, n_total]
        self._effect: dict = {}

    # -- learned mask -----------------------------------------------------
    def _state_key(self, frame: Optional[list]) -> tuple:
        if not frame:
            return ()
        out = []
        for grid in frame:
            m = _mask_borders(grid)  # tuple-of-tuples, border already zeroed
            if self._mask_cells and self.USE_MASK:
                rows = [list(r) for r in m]
                h = len(rows)
                w = len(rows[0]) if h else 0
                for (r, c) in self._mask_cells:
                    if 0 <= r < h and 0 <= c < w:
                        rows[r][c] = 0
                m = tuple(tuple(r) for r in rows)
            out.append(m)
        return tuple(out)

    def _observe_transition(self, frame: Optional[list]) -> None:
        """Diff the new interior grid against the previous one; accumulate which
        cells change and how effective the last emitted action was. Freeze the
        counter mask once enough transitions are seen."""
        if not frame:
            return
        interior = _mask_borders(frame[-1])
        prev = self._prev_interior
        self._prev_interior = interior
        if prev is None or len(prev) != len(interior):
            return
        h = len(interior)
        w = len(interior[0]) if h else 0
        changed_any = False
        learning = self._mask_cells is None  # only scan cells while unfrozen
        for r in range(h):
            pr, cr = prev[r], interior[r]
            if pr == cr:
                continue
            changed_any = True
            if learning:
                for c in range(w):
                    if pr[c] != cr[c]:
                        k = (r, c)
                        self._change_counts[k] = self._change_counts.get(k, 0) + 1
        self._transitions += 1
        # attribute effectiveness to the action that produced this transition
        if self._prev_emitted is not None:
            e = self._effect.setdefault(self._prev_emitted, [0, 0])
            e[1] += 1
            if changed_any:
                e[0] += 1
        if learning and self._transitions >= self.MASK_WARMUP:
            self._freeze_mask(h, w)

    def _freeze_mask(self, h: int, w: int) -> None:
        b = STATUS_BAR_BORDER
        interior_area = max(1, (h - 2 * b) * (w - 2 * b))
        threshold = self.MASK_CHANGE_RATIO * self._transitions
        cells = {k for k, n in self._change_counts.items() if n >= threshold}
        # guard: a too-large mask would erase real game state -> keep border-only
        if len(cells) > self.MASK_MAX_FRACTION * interior_area:
            cells = set()
        self._mask_cells = frozenset(cells)
        self._change_counts = {}  # free memory

    # -- effective-action ordering ---------------------------------------
    def _build_plans(self, latest_frame: FrameData) -> list:
        plans = super()._build_plans(latest_frame)
        if not self._effect or not self.USE_REORDER:
            return plans

        def rank(plan: Any) -> float:
            name = getattr(plan, "name", str(plan))
            n_chg, n_tot = self._effect.get(name, (0, 0))
            # unknown actions -> 0.5 (worth exploring); known -> measured P(change)
            return (n_chg / n_tot) if n_tot else 0.5

        simple = [p for p in plans if not isinstance(p, tuple)]
        clicks = [p for p in plans if isinstance(p, tuple)]
        simple.sort(key=rank, reverse=True)  # most effective first
        return simple + clicks               # clicks keep inherited likeness order

    # -- hook into the loop ----------------------------------------------
    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        # Learn from the transition the previous action produced, THEN decide so
        # this step's state key already uses the up-to-date mask.
        self._observe_transition(latest_frame.frame)
        action = super().choose_action(frames, latest_frame)
        self._prev_emitted = getattr(action, "name", str(action))
        return action


class MyAgent(Explore2):
    MAX_ACTIONS = 15000
    RESET_LOOP_BREAK = 20
    MAX_RUNTIME_SECONDS = 8 * 3600 - 300

    DEFAULT_AGENT_CONFIG = {
        "USE_MASK": True,
        "USE_REORDER": True,
        "MASK_CHANGE_RATIO": 0.8,
        "MASK_MAX_FRACTION": 0.20,
    }
    GAME_CONFIG = {
        "cn04": {"USE_REORDER": False},
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self._consecutive_resets = 0
        short = self.game_id.split("-")[0]
        for name, value in self.DEFAULT_AGENT_CONFIG.items():
            setattr(self, name, value)
        for name, value in self.GAME_CONFIG.get(short, {}).items():
            setattr(self, name, value)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if self._consecutive_resets >= self.RESET_LOOP_BREAK:
            return True
        return (time.time() - self.start_time) >= self.MAX_RUNTIME_SECONDS

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        action = super().choose_action(frames, latest_frame)
        if action.name == "RESET":
            self._consecutive_resets += 1
        else:
            self._consecutive_resets = 0
        return action

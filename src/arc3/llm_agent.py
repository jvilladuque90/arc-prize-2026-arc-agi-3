"""LLMAgent: agente VLM con features objetuales inyectadas + fallback a GraphExplorer.

Diseño (síntesis de lo que puntúa alto + nuestro diferenciador):
  - Política = VLM (OpenAI-compatible, servido con vLLM). Cada turno ve la imagen del
    frame Y la descripción textual de objetos/transición (arc3.llm_prompt).
  - Cola de plan: ejecuta 1-3 acciones planeadas sin re-llamar al LLM (ahorra inferencias);
    se aborta si el frame no cambió o el estado se repite.
  - Memoria failed-state: (frame_hash, acción) inefectiva se recuerda y se le comunica al
    LLM; evita repetir lo que no funciona en ese estado exacto.
  - Fallback: ante CUALQUIER fallo del LLM (excepción, timeout, JSON vacío, plan inútil)
    delega en GraphExplorer — nunca crashea ni se queda sin acción.

`chat_fn(system, user_text, image_data_uri) -> str` se inyecta (el notebook la implementa
con el cliente vLLM); así este módulo es testeable sin GPU.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from .agent import SIMPLE_IDS, GraphExplorer
from .features import frame_to_grid
from .features import transition_features
from .llm_prompt import (
    ACTION_NAMES,
    REFLECT_SYSTEM,
    SYSTEM_PROMPT,
    build_reflection_text,
    build_user_text,
    frame_png_data_uri,
    parse_actions,
    parse_goal,
)

ChatFn = Callable[[str, str, Optional[str]], str]
REFLECT_EVERY = 15   # transiciones LLM entre reflexiones


def _hash(grid: np.ndarray, border: int = 3) -> str:
    g = grid[border:-border, border:-border] if border else grid
    return hashlib.sha1(g.tobytes()).hexdigest()


class LLMAgent:
    def __init__(
        self,
        game_id: str,
        chat_fn: ChatFn,
        max_actions: int = 15000,
        plan_max: int = 3,
        use_image: bool = True,
    ) -> None:
        self.game_id = game_id
        self.chat_fn = chat_fn
        self.max_actions = max_actions
        self.plan_max = plan_max
        self.use_image = use_image
        self.actions_taken = 0
        self.done = False

        self._fallback = GraphExplorer(game_id, max_actions=max_actions)
        self._plan: deque[dict[str, Any]] = deque()
        self._failed: dict[str, set[str]] = {}   # frame_hash -> {failure_key}
        self._prev_grid: Optional[np.ndarray] = None
        self._prev_hash: Optional[str] = None
        self._prev_action: Optional[dict[str, Any]] = None
        self._prev_levels = 0
        self._llm_calls = 0
        self._llm_fails = 0
        self._memory: str = ""          # memoria de reflexión (reglas/objetivo/evitar)
        self._history: list[str] = []   # transiciones compactas para reflexionar
        self._since_reflect = 0
        # efectividad por tipo de acción: id -> [cambios, usos]
        self._eff: dict[int, list[int]] = {}
        # búsqueda guiada: modelo de movimiento (acción -> [sum_dy,sum_dx,count]),
        # posición estimada del avatar, y sub-objetivo espacial propuesto por el LLM.
        self._motion: dict[int, list[float]] = {}
        self._avatar_xy: Optional[tuple[float, float]] = None
        self._goal: Optional[dict[str, Any]] = None   # {"type":"reach"|"click_all", ...}
        self._nav_left = 0
        self._nav_used = 0
        self._clicked_goal: set[tuple[int, int]] = set()   # celdas ya clickeadas para click_all
        # diagnóstico (para inspeccionar en los logs de Save & Run):
        self.diag_enabled = False
        self.diag: dict[str, Any] = {
            "fail_exception": 0, "fail_parse_empty": 0, "fail_no_legal": 0,
            "samples": [],   # (user_snippet, raw_reply, parsed) de las primeras decisiones
            "reflections": [],
        }

    # ----- memoria de inefectividad -----

    def _fail_key(self, action: dict[str, Any]) -> str:
        if action["id"] == 6:
            return f"click@{action.get('x')},{action.get('y')}"
        return f"a{action['id']}"

    def _ineffective(self, frame_hash: str) -> list[str]:
        return sorted(self._failed.get(frame_hash, set()))

    def _largest_centroid(self, grid: np.ndarray) -> Optional[tuple[float, float]]:
        from .features import connected_components
        counts = np.bincount(grid.ravel(), minlength=16)
        objs = connected_components(grid, int(counts.argmax()))
        if not objs:
            return None
        cy, cx = objs[0]["centroid"]
        return (cy, cx)   # (y, x)

    def _goal_action(self, grid: np.ndarray, available: list[int]) -> Optional[tuple[int, int, int]]:
        """Despacha el sub-objetivo activo a su controlador (reach / click_all)."""
        if self._goal is None:
            return None
        if self._goal["type"] == "click_all":
            return self._click_all_action(grid, available)
        return self._nav_action(available)

    def _click_all_action(self, grid: np.ndarray, available: list[int]) -> Optional[tuple[int, int, int]]:
        """Clickea el siguiente objeto no clickeado del color objetivo (goal click_all)."""
        if available and 6 not in available:
            self._goal = None
            return None
        from .features import connected_components
        counts = np.bincount(grid.ravel(), minlength=16)
        target = self._goal["color"]
        for o in connected_components(grid, int(counts.argmax())):
            if o["color"] != target:
                continue
            cy, cx = o["centroid"]
            cell = (int(round(cx)), int(round(cy)))
            if cell in self._clicked_goal:
                continue
            self._clicked_goal.add(cell)
            return (6, cell[0], cell[1])
        self._goal = None   # no quedan objetos del color -> objetivo cumplido
        return None

    def _nav_action(self, available: list[int]) -> Optional[tuple[int, int, int]]:
        """Elige la acción de movimiento cuyo vector aprendido más acerca al objetivo."""
        if self._goal is None or self._goal["type"] != "reach" or \
                self._avatar_xy is None or not self._motion:
            return None
        gx, gy = self._goal["x"], self._goal["y"]
        ay, ax = self._avatar_xy
        dist = ((ay - gy) ** 2 + (ax - gx) ** 2) ** 0.5
        if dist <= 2:                       # objetivo alcanzado
            self._goal = None
            return None
        best = None
        best_d = dist
        for aid, (sdy, sdx, n) in self._motion.items():
            if n < 1 or (available and aid not in available):
                continue
            vy, vx = sdy / n, sdx / n
            nd = ((ay + vy - gy) ** 2 + (ax + vx - gx) ** 2) ** 0.5
            if nd < best_d:
                best_d, best = nd, aid
        if best is None:                    # ningún movimiento acerca: abandonar nav
            self._goal = None
            return None
        return (best, -1, -1)

    def _effectiveness_summary(self) -> str:
        """Resumen legible de P(cambio) por acción (nombre) para inyectar al LLM."""
        from .llm_prompt import ACTION_NAMES
        parts = []
        for aid, (chg, uses) in sorted(self._eff.items()):
            if uses >= 2:
                name = ACTION_NAMES.get(aid, f"a{aid}")
                parts.append(f"{name}={chg}/{uses} changed")
        return ", ".join(parts)

    def _digest_previous(self, grid: np.ndarray, levels: int) -> None:
        if self._prev_grid is None or self._prev_action is None:
            return
        changed = bool((self._prev_grid != grid).any())
        leveled = levels != self._prev_levels
        aid = self._prev_action["id"]
        st = self._eff.setdefault(aid, [0, 0])
        st[1] += 1
        st[0] += int(changed)
        # registro compacto de la transición para la reflexión
        tf = transition_features(self._prev_grid, grid)
        # modelo de movimiento: si una acción simple produjo una traslación coherente,
        # aprende su vector y actualiza la posición estimada del avatar.
        if aid in SIMPLE_IDS and tf["move_score"] > 0.6 and (tf["move_dy"] or tf["move_dx"]):
            m = self._motion.setdefault(aid, [0.0, 0.0, 0.0])
            m[0] += tf["move_dy"]; m[1] += tf["move_dx"]; m[2] += 1
            if self._avatar_xy is None:
                self._avatar_xy = self._largest_centroid(grid)
            if self._avatar_xy is not None:
                self._avatar_xy = (self._avatar_xy[0] + tf["move_dy"],
                                   self._avatar_xy[1] + tf["move_dx"])
        self._history.append(
            f"{self._fail_key(self._prev_action)} -> changed={tf['n_changed']} "
            f"move=({tf['move_dy']},{tf['move_dx']}) level_delta={levels - self._prev_levels}")
        if not changed and not leveled and self._prev_hash is not None:
            self._failed.setdefault(self._prev_hash, set()).add(
                self._fail_key(self._prev_action))
        if not changed:  # plan que no mueve nada se aborta
            self._plan.clear()
            # GUARD (v11=0.25 regresó): click_all clickeaba 16 objetos a ciegas y malgastaba
            # acciones donde clickear no sirve. Abortar la excursión click_all en cuanto un
            # click no produce cambio -> solo puede ayudar, nunca dañar. La navegación tiene
            # su propia salida (_nav_action devuelve None si nada acerca) y no se toca aquí.
            if (self._goal is not None and self._goal.get("type") == "click_all"
                    and self._prev_action["id"] == 6 and not leveled):
                self._goal = None
                self._nav_left = 0

    def _maybe_reflect(self, levels: int) -> None:
        """Cada REFLECT_EVERY transiciones LLM, resume el historial en memoria accionable."""
        self._since_reflect += 1
        if self._since_reflect < REFLECT_EVERY or len(self._history) < 5:
            return
        self._since_reflect = 0
        try:
            text = build_reflection_text(self._history, self._memory, levels)
            new_mem = self.chat_fn(REFLECT_SYSTEM, text, None)
            if new_mem and "#" in new_mem:
                self._memory = new_mem.strip()[:1800]
                if self.diag_enabled and len(self.diag["reflections"]) < 4:
                    self.diag["reflections"].append(self._memory[:400])
        except Exception:
            pass

    # ----- API principal -----

    def choose(
        self,
        grid: np.ndarray,
        state: str,
        levels_completed: int,
        available_actions: list[int],
    ) -> tuple[int, int, int]:
        self.actions_taken += 1
        if self.actions_taken > self.max_actions:
            self.done = True

        self._digest_previous(grid, levels_completed)
        if levels_completed > self._prev_levels:
            self._failed.clear()   # lo inefectivo en un nivel puede servir en el siguiente
            self._plan.clear()
            self._history.clear()  # las reglas pueden cambiar de nivel; memoria se re-forma
        self._prev_levels = levels_completed

        if state in ("NOT_PLAYED", "GAME_OVER"):
            return self._emit(grid, {"id": 0})

        frame_hash = _hash(grid)

        # 0) búsqueda guiada: si el LLM fijó un sub-objetivo, perseguirlo con el controlador
        #    correspondiente (navegar / click_all) sin gastar llamadas al LLM en cada paso.
        if self._goal is not None and self._nav_left > 0:
            ga = self._goal_action(grid, available_actions)
            if ga is not None:
                self._nav_left -= 1
                self._nav_used += 1
                act = {"id": ga[0]} if ga[0] != 6 else {"id": 6, "x": ga[1], "y": ga[2]}
                return self._emit(grid, act)

        # 1) plan pendiente que siga siendo legal y no inefectivo
        while self._plan:
            a = self._plan.popleft()
            if available_actions and a["id"] not in available_actions and a["id"] != 6:
                continue
            if self._fail_key(a) in self._failed.get(frame_hash, set()):
                continue
            return self._emit(grid, a)

        # 2) consultar al LLM
        action = self._ask_llm(grid, frame_hash, available_actions, levels_completed)
        if action is not None:
            return self._emit(grid, action)

        # 3) fallback: GraphExplorer (nunca sin acción)
        self._llm_fails += 1
        aid, x, y = self._fallback.choose(grid, state, levels_completed, available_actions)
        self.done = self.done or self._fallback.done
        act = {"id": aid} if aid != 6 else {"id": 6, "x": x, "y": y}
        # no re-emitir por fallback (ya viene del fallback); solo registrar estado
        self._prev_grid = grid.copy()
        self._prev_hash = frame_hash
        self._prev_action = act
        return aid, x, y

    def _ask_llm(
        self, grid: np.ndarray, frame_hash: str,
        available_actions: list[int], levels: int,
    ) -> Optional[dict[str, Any]]:
        try:
            self._maybe_reflect(levels)
            user = build_user_text(grid, self._prev_grid, available_actions, levels,
                                   ineffective=self._ineffective(frame_hash),
                                   memory=self._memory or None,
                                   effectiveness=self._effectiveness_summary() or None)
            img = frame_png_data_uri(grid) if self.use_image else None
            self._llm_calls += 1
            reply = self.chat_fn(SYSTEM_PROMPT, user, img)
            actions = parse_actions(reply)
            goal = parse_goal(reply)
            if goal is not None:
                # reach necesita modelo de movimiento; click_all no
                if goal["type"] == "reach" and not self._motion:
                    goal = None
                if goal is not None:
                    self._goal = goal
                    self._nav_left = 16 if goal["type"] == "click_all" else 12
                    self._clicked_goal.clear()
                    if goal["type"] == "reach":
                        self._avatar_xy = self._avatar_xy or self._largest_centroid(grid)
        except Exception as e:
            self.diag["fail_exception"] += 1
            if self.diag_enabled and len(self.diag["samples"]) < 6:
                self.diag["samples"].append(("EXCEPTION", str(e)[:200], None))
            return None
        if self.diag_enabled and len(self.diag["samples"]) < 6:
            self.diag["samples"].append((user[:300], reply[:400], actions))
        if not actions:
            self.diag["fail_parse_empty"] += 1
            return None
        # filtrar por legalidad e inefectividad conocida
        legal = []
        failed = self._failed.get(frame_hash, set())
        for a in actions:
            if available_actions and a["id"] not in available_actions and a["id"] != 6:
                continue
            if self._fail_key(a) in failed:
                continue
            legal.append(a)
        if not legal:
            self.diag["fail_no_legal"] += 1
            return None
        for a in legal[1:self.plan_max]:
            self._plan.append(a)
        return legal[0]

    def _emit(self, grid: np.ndarray, action: dict[str, Any]) -> tuple[int, int, int]:
        self._prev_grid = grid.copy()
        self._prev_hash = _hash(grid)
        self._prev_action = action
        aid = action["id"]
        return aid, int(action.get("x", -1)), int(action.get("y", -1))


class HybridAgent:
    """Explorador barato para el piso + LLM cuando se estanca, para el techo.

    Racional (docs/STRATEGY.md): la exploración pura está capada en 0.25 (v1=v2=v3), pero
    resuelve gratis los niveles fáciles. El LLM (lento) es la única vía a los niveles que
    exigen entender el objetivo. El híbrido deja al GraphExplorer bankear niveles mientras
    progresa, y delega en el LLMAgent SOLO cuando el explorador se atasca (agota su grafo
    o pasa mucho sin subir de nivel). Garantiza >= piso del explorador y añade el techo LLM,
    gastando pocas inferencias (solo donde hacen falta).
    """

    def __init__(
        self,
        game_id: str,
        chat_fn: ChatFn,
        max_actions: int = 15000,
        stuck_actions: int = 300,
        use_image: bool = False,
    ) -> None:
        self.game_id = game_id
        self.max_actions = max_actions
        self.stuck_actions = stuck_actions
        self.actions_taken = 0
        self.done = False

        self._explorer = GraphExplorer(game_id, max_actions=max_actions)
        self._llm = LLMAgent(game_id, chat_fn, max_actions=max_actions, use_image=use_image)
        self._using_llm = False
        self._best_levels = 0
        self._since_progress = 0

    @property
    def _nodes(self):  # compat con runner.play_game
        return self._explorer._nodes

    @property
    def _llm_calls(self) -> int:
        return self._llm._llm_calls

    @property
    def _llm_fails(self) -> int:
        return self._llm._llm_fails

    @property
    def diag_enabled(self) -> bool:
        return self._llm.diag_enabled

    @diag_enabled.setter
    def diag_enabled(self, v: bool) -> None:
        self._llm.diag_enabled = v

    @property
    def diag(self) -> dict:
        return self._llm.diag

    @property
    def _memory(self) -> str:
        return self._llm._memory

    @property
    def _nav_used(self) -> int:
        return self._llm._nav_used

    def choose(
        self, grid: np.ndarray, state: str, levels_completed: int,
        available_actions: list[int],
    ) -> tuple[int, int, int]:
        self.actions_taken += 1
        if self.actions_taken > self.max_actions:
            self.done = True

        # progreso => resetea el contador de estancamiento; regresión imposible (monótono)
        if levels_completed > self._best_levels:
            self._best_levels = levels_completed
            self._since_progress = 0
            self._using_llm = False   # el explorador volvió a avanzar: dejarlo seguir barato
        else:
            self._since_progress += 1

        # ¿atascado? el explorador agotó su grafo alcanzable, o mucho sin progreso
        explorer_stuck = (self._explorer._exhausted_resets >= 1
                          or self._since_progress >= self.stuck_actions)
        if explorer_stuck and not self._using_llm:
            self._using_llm = True

        if self._using_llm:
            out = self._llm.choose(grid, state, levels_completed, available_actions)
            self.done = self.done or self._llm.done
            return out
        out = self._explorer.choose(grid, state, levels_completed, available_actions)
        # el explorador puede declararse done al agotarse; en híbrido eso solo dispara el LLM
        if self._explorer.done:
            self._explorer.done = False
            self._using_llm = True
        return out

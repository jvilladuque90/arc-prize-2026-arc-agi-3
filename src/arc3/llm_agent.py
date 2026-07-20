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

from .agent import GraphExplorer
from .features import frame_to_grid
from .llm_prompt import (
    ACTION_NAMES,
    SYSTEM_PROMPT,
    build_user_text,
    frame_png_data_uri,
    parse_actions,
)

ChatFn = Callable[[str, str, Optional[str]], str]


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
        self._llm_fails = 0

    # ----- memoria de inefectividad -----

    def _fail_key(self, action: dict[str, Any]) -> str:
        if action["id"] == 6:
            return f"click@{action.get('x')},{action.get('y')}"
        return f"a{action['id']}"

    def _ineffective(self, frame_hash: str) -> list[str]:
        return sorted(self._failed.get(frame_hash, set()))

    def _digest_previous(self, grid: np.ndarray, levels: int) -> None:
        if self._prev_grid is None or self._prev_action is None:
            return
        changed = bool((self._prev_grid != grid).any())
        leveled = levels != self._prev_levels
        if not changed and not leveled and self._prev_hash is not None:
            self._failed.setdefault(self._prev_hash, set()).add(
                self._fail_key(self._prev_action))
        if not changed:  # plan que no mueve nada se aborta
            self._plan.clear()

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
        self._prev_levels = levels_completed

        if state in ("NOT_PLAYED", "GAME_OVER"):
            return self._emit(grid, {"id": 0})

        frame_hash = _hash(grid)

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
            user = build_user_text(grid, self._prev_grid, available_actions, levels,
                                   ineffective=self._ineffective(frame_hash))
            img = frame_png_data_uri(grid) if self.use_image else None
            reply = self.chat_fn(SYSTEM_PROMPT, user, img)
            actions = parse_actions(reply)
        except Exception:
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

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
from .features import transition_features
from .llm_prompt import (
    ACTION_NAMES,
    REFLECT_SYSTEM,
    SYSTEM_PROMPT,
    build_reflection_text,
    build_user_text,
    frame_png_data_uri,
    parse_actions,
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

    def _digest_previous(self, grid: np.ndarray, levels: int) -> None:
        if self._prev_grid is None or self._prev_action is None:
            return
        changed = bool((self._prev_grid != grid).any())
        leveled = levels != self._prev_levels
        # registro compacto de la transición para la reflexión
        tf = transition_features(self._prev_grid, grid)
        self._history.append(
            f"{self._fail_key(self._prev_action)} -> changed={tf['n_changed']} "
            f"move=({tf['move_dy']},{tf['move_dx']}) level_delta={levels - self._prev_levels}")
        if not changed and not leveled and self._prev_hash is not None:
            self._failed.setdefault(self._prev_hash, set()).add(
                self._fail_key(self._prev_action))
        if not changed:  # plan que no mueve nada se aborta
            self._plan.clear()

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
                                   memory=self._memory or None)
            img = frame_png_data_uri(grid) if self.use_image else None
            self._llm_calls += 1
            reply = self.chat_fn(SYSTEM_PROMPT, user, img)
            actions = parse_actions(reply)
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

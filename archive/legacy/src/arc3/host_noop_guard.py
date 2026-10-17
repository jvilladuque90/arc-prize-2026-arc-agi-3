"""Guard de no-ops en el ANFITRION para el harness original (sin conciencia de animacion).

Bloquea, antes de que llegue al entorno, repetir una accion que ya se probo inerte
sobre EXACTAMENTE el mismo tablero del mismo nivel. Cero tokens de prompt: el modelo
recibe el mismo diccionario de resultado que anim devuelve en ese caso
(`stop_reason: known_noop`, "no action budget spent") y decide otra cosa.

ATRIBUCION: `board_signature`, `normalize_action_signature` y `NoopGuard` son la logica
de `inference/agent/noop_guard.py` del fork publico "animation-awareness" de jakobbrggen
(dataset Kaggle jakobbrggen/taaf-kaggle-source-anim-20260807-anim), reproducida aqui
para montarla sobre un harness que no la trae. El montaje (envolver el `step_env` que
`analyze()` recibe) es nuestro.

LO QUE CAMBIA RESPECTO A ANIM, Y POR QUE ES SEGURO
--------------------------------------------------
Anim exime a las acciones ANIMADAS (varios fotogramas) de contar como no-op. El harness
original no expone `frame_count`, asi que aqui `animated` es siempre False. Consecuencia:
en un juego "tipo 1" una accion que solo tuvo efecto en la animacion puede quedar
registrada como no-op y su REPETICION EXACTA sobre el MISMO tablero quedaria bloqueada.
Repetir la misma accion sobre el mismo tablero identico rara vez es util; y el bloqueo
es solo de esa combinacion exacta: el modelo puede hacer cualquier otra cosa.

SOLIDEZ
-------
- Solo se OBSERVA cuando el lote tuvo una unica accion ejecutada (`executed_count == 1`):
  en un lote de varias no se puede atribuir `board_changed` a una accion, y registrar un
  no-op falso seria peor que no registrar nada.
- Solo se BLOQUEA por la PRIMERA accion del lote (conservador): si es un no-op conocido
  sobre el tablero actual, se devuelve el payload de bloqueo sin llamar al entorno.
- Cualquier excepcion dentro del guard degrada a llamar al `step_env` original intacto.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Callable


# ----------------------------------------------------------------- de anim (verbatim)

def board_signature(grid: Any) -> str:
    rows = tuple(tuple(int(cell) for cell in row) for row in grid or ())
    digest = hashlib.blake2b(repr(rows).encode("utf-8"), digest_size=8)
    return digest.hexdigest()


def normalize_action_signature(value: Any) -> str:
    return " ".join(str(value or "").split())


class NoopGuard:
    """(nivel, firma del tablero antes, firma de la accion) probados inertes."""

    def __init__(self, *, max_states_per_level: int = 512, max_actions_per_state: int = 16) -> None:
        self._max_states_per_level = max(1, int(max_states_per_level))
        self._max_actions_per_state = max(1, int(max_actions_per_state))
        self._levels: "OrderedDict[int, OrderedDict[str, OrderedDict[str, None]]]" = OrderedDict()

    def observe(self, *, level: Any, board_before_sig: str, action_sig: str,
                board_changed: bool, animated: bool = False) -> None:
        try:
            level_num = int(level)
        except (TypeError, ValueError):
            return
        action_sig = normalize_action_signature(action_sig)
        if not action_sig:
            return
        states = self._levels.setdefault(level_num, OrderedDict())
        if board_changed or animated:
            entry = states.get(board_before_sig)
            if entry is not None and action_sig in entry:
                del entry[action_sig]
                if not entry:
                    del states[board_before_sig]
            return
        entry = states.get(board_before_sig)
        if entry is None:
            entry = OrderedDict()
            states[board_before_sig] = entry
            while len(states) > self._max_states_per_level:
                states.popitem(last=False)
        entry[action_sig] = None
        while len(entry) > self._max_actions_per_state:
            entry.popitem(last=False)

    def is_known_noop(self, level: Any, board_sig: str, action_sig: str) -> bool:
        try:
            level_num = int(level)
        except (TypeError, ValueError):
            return False
        states = self._levels.get(level_num)
        if not states:
            return False
        entry = states.get(board_sig)
        if not entry:
            return False
        return normalize_action_signature(action_sig) in entry


# ----------------------------------------------------------------- nuestro montaje

_DISPLAY_MAP: dict[str, str] = {}   # p.ej. {"ACTION1": "UP"}; la celda puede rellenarlo


def set_display_map(mapping: dict[str, str] | None) -> None:
    global _DISPLAY_MAP
    _DISPLAY_MAP = {str(k).upper(): str(v) for k, v in (mapping or {}).items()}


def pending_action_signature(action: dict[str, Any]) -> str:
    """Firma de una accion aun no ejecutada, en el formato que el entorno devuelve como
    `action_display` (RIGHT, MOUSE(row=r, col=c)) para poder compararla con lo observado."""
    if not isinstance(action, dict):
        return ""
    name = str(action.get("action") or "").strip()
    if not name:
        return ""
    display = _DISPLAY_MAP.get(name.upper(), name.upper())
    if display.upper() == "MOUSE":
        try:
            row = max(0, min(63, int(action.get("row"))))
            col = max(0, min(63, int(action.get("col"))))
        except (TypeError, ValueError):
            return ""
        return f"MOUSE(row={row}, col={col})"
    return display


class GuardStats:
    """Contadores + una linea JSON por bloqueo en un archivo del working dir, para leerlo
    desde la salida del kernel (el log de estos kernels vuelve vacio)."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.blocked = 0
        self.observed = 0
        self.noops = 0

    def _emit(self, rec: dict[str, Any]) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def on_observe(self, level: Any, action_sig: str, board_changed: bool) -> None:
        self.observed += 1
        if not board_changed:
            self.noops += 1

    def on_block(self, level: Any, action_sig: str, game: str = "") -> None:
        self.blocked += 1
        self._emit({"t": time.time(), "event": "block", "level": level, "action": action_sig,
                    "game": game, "blocked": self.blocked, "observed": self.observed, "noops": self.noops})


def blocked_payload(*, action_sig: str, level: Any, requested: int,
                    last_result: dict[str, Any] | None, valid_actions: list[str]) -> dict[str, Any]:
    last = last_result or {}
    return {
        "executed": False,
        "action_num": last.get("action_num"),
        "level": level,
        "score": last.get("score"),
        "reward": 0.0,
        "state": last.get("state") or "NOT_FINISHED",
        "valid_actions": list(valid_actions or []),
        "board_changed": False,
        "done": False,
        "level_completed": False,
        "game_over": False,
        "run_complete": False,
        "requested_count": int(requested),
        "executed_count": 0,
        "stopped_early": True,
        "stop_reason": "known_noop",
        "stop_detail": (f"{action_sig} already had no effect in this exact board state; "
                        "blocked before execution, no action budget spent."),
    }


def make_guarded_step_env(inner: Callable[[dict[str, Any]], dict[str, Any]], state_path: Any,
                          guard: NoopGuard, load_state: Callable[[Any], tuple[Any, Any]],
                          stats: GuardStats | None = None,
                          valid_actions_getter: Callable[[], list[str]] | None = None,
                          last_result_getter: Callable[[], dict[str, Any] | None] | None = None,
                          game_id: str = "") -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Envuelve el `step_env` del solver. Bloquea antes; observa despues; degrada a `inner`."""

    def guarded(arguments: dict[str, Any]) -> dict[str, Any]:
        actions = (arguments or {}).get("actions") or []
        frame = None
        level = None
        sig_before = None
        try:
            frame, _hist = load_state(state_path)
            if frame is not None:
                level = int(getattr(frame, "level", 1) or 1)
                sig_before = board_signature(getattr(frame, "grid", ()))
        except Exception:  # noqa: BLE001
            frame = None
        # --- bloqueo: la primera accion del lote es un no-op conocido sobre este tablero
        try:
            if frame is not None and actions and isinstance(actions[0], dict):
                a_sig = pending_action_signature(actions[0])
                if a_sig and guard.is_known_noop(level, sig_before, a_sig):
                    if stats is not None:
                        stats.on_block(level, a_sig, game_id)
                    return blocked_payload(
                        action_sig=a_sig, level=level, requested=len(actions),
                        last_result=last_result_getter() if last_result_getter else None,
                        valid_actions=valid_actions_getter() if valid_actions_getter else [])
        except Exception:  # noqa: BLE001
            pass
        raw = inner(arguments)
        # --- observacion: solo lotes de UNA accion ejecutada
        try:
            if (frame is not None and isinstance(raw, dict) and raw.get("executed")
                    and len(actions) == 1 and int(raw.get("executed_count") or 1) == 1):
                disp = str(raw.get("action_display") or raw.get("action_name") or "").strip()
                if not disp:
                    disp = pending_action_signature(actions[0]) if isinstance(actions[0], dict) else ""
                if disp:
                    changed = bool(raw.get("board_changed"))
                    guard.observe(level=level, board_before_sig=sig_before, action_sig=disp,
                                  board_changed=changed, animated=False)
                    if stats is not None:
                        stats.on_observe(level, disp, changed)
        except Exception:  # noqa: BLE001
            pass
        return raw

    return guarded

"""Runner paralelo de juegos ARC-AGI-3 sobre un Arcade (offline o gateway).

En el rerun real cada acción es un request HTTP al gateway (latencia-bound): jugar
N juegos en paralelo multiplica el throughput de acciones (el milestone winner usaba
concurrencia 28). Offline es CPU-bound: pocos workers bastan.

Uso (notebook de submission y eval local):
    arcade = Arcade(operation_mode=..., ...)
    results = run_games(arcade, game_ids, total_budget_s=..., workers=12)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from arcengine import GameAction

from .agent import GraphExplorer
from .features import frame_to_grid

# Fábrica de agente inyectable: el notebook LLM la reemplaza por una que crea LLMAgent.
# Firma: (game_id: str, max_actions: int) -> agente con .choose(...) y .done.
_AGENT_FACTORY: Optional[Callable[[str, int], Any]] = None


def play_game(
    env: Any,
    game_id: str,
    time_budget_s: float,
    max_actions: int = 15000,
    stop_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """Juega un env (EnvironmentWrapper de arc_agi) hasta agotar budget.

    Usa _AGENT_FACTORY si está definida (LLMAgent), si no GraphExplorer.
    """
    agent = (_AGENT_FACTORY or (lambda gid, ma: GraphExplorer(gid, max_actions=ma)))(
        game_id, max_actions)
    if hasattr(agent, "diag_enabled"):
        agent.diag_enabled = True   # capturar muestras LLM/reflexión para los logs
    t0 = time.time()
    try:
        frame = env.observation_space or env.reset()
    except Exception:
        frame = None
    best = 0
    win = False
    while (
        frame is not None
        and not agent.done
        and time.time() - t0 < time_budget_s
        and not (stop_event and stop_event.is_set())
    ):
        try:
            grid = frame_to_grid(frame.frame)
            aid, x, y = agent.choose(
                grid, frame.state.value, frame.levels_completed,
                list(frame.available_actions or []),
            )
            action = GameAction.from_id(aid)
            data: dict[str, Any] = {"game_id": game_id}
            if aid == 6:
                data.update(x=x, y=y)
            frame = env.reset() if aid == 0 else env.step(action, data=data)
        except Exception:
            try:
                frame = env.reset()
            except Exception:
                break
        if frame is not None:
            best = max(best, frame.levels_completed)
            if frame.state.value == "WIN":
                win = True
                break
    return {
        "game_id": game_id,
        "levels": best,
        "win": win,
        "actions": getattr(agent, "actions_taken", 0),
        "seconds": round(time.time() - t0, 1),
        "nodes": len(getattr(agent, "_nodes", ()) or ()),   # GraphExplorer; LLMAgent no tiene
        "llm_calls": getattr(agent, "_llm_calls", 0),
        "llm_fails": getattr(agent, "_llm_fails", 0),
        "diag": getattr(agent, "diag", None),               # muestras LLM/reflexión para logs
        "memory": getattr(agent, "_memory", ""),
    }


def run_games(
    arcade: Any,
    game_ids: list[str],
    total_budget_s: float,
    workers: int = 8,
    max_actions: int = 15000,
    max_game_s: float = 1800.0,
    min_game_s: float = 60.0,
    card_id: Optional[str] = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Juega todos los game_ids con un pool de threads y presupuesto global compartido."""
    t_start = time.time()
    results: list[dict[str, Any]] = []
    queue = list(game_ids)
    lock = threading.Lock()
    stop_event = threading.Event()

    def remaining() -> float:
        return total_budget_s - (time.time() - t_start)

    def worker() -> None:
        while not stop_event.is_set():
            with lock:
                if not queue:
                    return
                games_left = len(queue)
                game_id = queue.pop(0)
            rem = remaining()
            if rem < min_game_s:
                stop_event.set()
                return
            # presupuesto por juego: reparte el tiempo restante entre los juegos que
            # quedan, multiplicado por los workers (corren en paralelo)
            budget = max(min_game_s, min(max_game_s, rem * workers / max(games_left, 1)))
            budget = min(budget, rem)
            try:
                with lock:
                    env = arcade.make(game_id, scorecard_id=card_id)
                if env is None:
                    raise RuntimeError("make() devolvió None")
                r = play_game(env, game_id, budget, max_actions, stop_event)
            except Exception as e:
                r = {"game_id": game_id, "levels": 0, "win": False, "actions": 0,
                     "seconds": 0.0, "nodes": 0, "error": str(e)[:200]}
            with lock:
                results.append(r)
                if verbose:
                    print(f"[{len(results)}/{len(game_ids)}] {r['game_id']}: "
                          f"{r['levels']} niveles, {r['actions']} acciones, "
                          f"{r['seconds']}s{' WIN' if r.get('win') else ''}"
                          f"{' ERROR ' + r['error'] if r.get('error') else ''}",
                          flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=max(0.0, total_budget_s - (time.time() - t_start)) + max_game_s)
    if verbose:
        total = sum(r["levels"] for r in results)
        print(f"TOTAL: {total} niveles en {len(results)} juegos "
              f"({time.time() - t_start:.0f}s)", flush=True)
    return results

"""Preludio de exploración: el explorador CPU juega ANTES que el LLM, en cada juego.

POR QUE (medido, docs/DESIGN.md §8.22). Sobre los 25 juegos locales, con la config
desplegada y régimen de 2 h por juego:

    LLM        9 niveles     solo el LLM gana en: ar25, bp35, sb26
    explorador 18 niveles    solo el explorador en: ls20, m0r0, sc25, sp80, tn36, tu93, vc33
    union      21 niveles

Las dos capacidades son distintas y en parte disjuntas: bp35 y sb26 son juegos donde
el explorador da CERO incluso con 40.000 acciones, y el LLM los resuelve; tu93 es lo
contrario. La union vale 2.3x lo que el LLM solo.

POR QUE SECUENCIAL Y NO EN PARALELO. Intercalar acciones mientras el LLM piensa rompe
tres cosas (§8.21): su accion se aplicaria a un tablero que no vio, sus `history_entries`
contendrian jugadas que no decidio —el modo de fallo que medimos como el mas danino— y
habria carrera sobre el fichero de estado. Corriendo ANTES de que arranque la sesion no
existe ninguno de los tres: el explorador termina, y el LLM abre su historial desde el
estado resultante como si fuera el inicial.

COSTE. ~2.000 acciones sobre el gateway a ~0.15 s = ~5 min de los 132 de la ventana.
El LLM esta limitado por GPU (no por reloj) dentro de su ventana, asi que pierde ~4% de
sus tokens. A cambio arranca en el nivel donde la busqueda se atasco — que es justo donde
su comprension hace falta y la fuerza bruta ya no llega.

CAVEAT HONESTO. Los 25 juegos locales SON los publicos, sobre los que este explorador se
ajusto en julio; en el set oculto marco 0.25 frente al 0.97 del harness. La estimacion
para el oculto (0.25 x 67% disjunto ~ +0.17) esta justo en el liston y con barras de
error grandes. Este modulo es la apuesta, no una conclusion.
"""

from __future__ import annotations

import time
from typing import Any

DEFAULT_MAX_ACTIONS = 2000
DEFAULT_MAX_SECONDS = 420.0        # tope duro: 7 min de la ventana de 132
LOG_PREFIX = "[hybrid_prelude]"


def _grid_from_state(state: Any):
    """Ultimo frame visible como np.ndarray, o None si no se puede leer."""
    import numpy as np

    try:
        raw = state.raw.frame
        if not raw:
            return None
        return np.asarray(raw[-1], dtype=np.int8)
    except Exception:
        return None


def run_prelude(game: Any, max_actions: int = DEFAULT_MAX_ACTIONS,
                max_seconds: float = DEFAULT_MAX_SECONDS,
                should_stop: Any = None) -> dict:
    """Juega `game` con el GraphExplorer hasta agotar acciones, tiempo o progreso.

    Devuelve un resumen; NUNCA lanza. Cualquier fallo deja el juego como estaba y
    la sesion del LLM sigue su curso normal — el peor caso es que este preludio no
    haga nada, no que rompa la partida.
    """
    from arc3.agent import GraphExplorer          # embebido en el notebook
    import arcengine

    res = {"acciones": 0, "niveles": 0, "motivo": "ok"}
    try:
        run = getattr(game, "game_run", None)
        if run is None or getattr(run, "state", None) != "playing":
            res["motivo"] = "el juego no esta en 'playing'"
            return res

        game_id = getattr(run, "game_id", "?")
        agent = GraphExplorer(game_id, max_actions=max_actions + 10)
        t0 = time.monotonic()
        niveles = int(game.current_state.levels_completed)

        for _ in range(max_actions):
            if time.monotonic() - t0 > max_seconds:
                res["motivo"] = "tope de tiempo"
                break
            if should_stop is not None and should_stop():
                res["motivo"] = "stop_event"
                break
            if getattr(run, "state", None) != "playing":
                res["motivo"] = f"estado {getattr(run, 'state', '?')}"
                break

            state = game.current_state
            grid = _grid_from_state(state)
            if grid is None:
                res["motivo"] = "sin frame legible"
                break

            validas = list(state.available_actions or [])
            try:
                # available_actions del harness son INTS ([0,1,2,3,4]) y el
                # explorador los indexa como tales. Pasarlos como "ACTION0"...
                # hacia que no formara ningun candidato valido y diera vueltas:
                # 600 acciones -> 0 niveles. Probado contra el Game real, que es
                # justo lo que un stub no habria detectado.
                aid, x, y = agent.choose(grid, state.raw.state.value,
                                         int(state.levels_completed), list(validas))
            except Exception:
                res["motivo"] = "el explorador fallo al elegir"
                break

            # el harness RECHAZA acciones fuera de available_actions: filtrar aqui
            # evita que una excepcion aborte el preludio entero
            if aid not in validas:
                alternativa = next((v for v in validas if v != 0), None)
                if alternativa is None:
                    res["motivo"] = "sin acciones validas"
                    break
                aid, x, y = alternativa, x, y

            try:
                action_id = arcengine.GameAction.from_id(aid)
                data = {"x": int(x), "y": int(y)} if action_id.is_complex() else {}
                nuevo = game.execute_action(
                    arcengine.ActionInput(id=action_id, data=data))
            except Exception:
                # una accion rechazada no debe tumbar el preludio
                continue

            res["acciones"] += 1
            niveles = max(niveles, int(nuevo.levels_completed))
            if getattr(nuevo, "won", False):
                res["motivo"] = "juego ganado"
                break

        res["niveles"] = niveles
    except Exception as exc:                      # noqa: BLE001 — degradar siempre
        res["motivo"] = f"excepcion {type(exc).__name__}: {exc}"
    return res

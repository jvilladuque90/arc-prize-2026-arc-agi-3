"""Prueba local (CPU, cero cuota) del guard de no-ops del anfitrion, con un step_env falso."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc3.host_noop_guard import (  # noqa: E402
    GuardStats,
    NoopGuard,
    board_signature,
    make_guarded_step_env,
    pending_action_signature,
    set_display_map,
)

fallos: list[str] = []


def check(ok, nombre, detalle=""):
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


@dataclass
class F:
    grid: tuple
    level: int


def tab(v, n=4):
    return tuple(tuple(v for _ in range(n)) for _ in range(n))


class Estado:
    """Simula load_runtime_state: devuelve el fotograma 'actual' que fijemos."""
    def __init__(self):
        self.frame = F(tab(0), 1)

    def load(self, _path):
        return self.frame, []


class Entorno:
    """step_env falso: registra llamadas y devuelve lo que le digamos."""
    def __init__(self):
        self.calls = []
        self.next_changed = False

    def __call__(self, args):
        self.calls.append(args)
        a = args["actions"]
        disp = pending_action_signature(a[0]) if a else ""
        return {"executed": True, "action_num": len(self.calls), "level": 1, "score": 0,
                "reward": 0.0, "state": "NOT_FINISHED", "valid_actions": ["UP", "DOWN", "MOUSE"],
                "board_changed": self.next_changed, "done": False, "level_completed": False,
                "game_over": False, "run_complete": False, "action_display": disp,
                "executed_actions": [disp], "requested_count": len(a), "executed_count": len(a)}


def main() -> int:
    print("PRUEBA DEL GUARD DE NO-OPS DEL ANFITRION")
    check(pending_action_signature({"action": "UP"}) == "UP", "firma simple")
    check(pending_action_signature({"action": "MOUSE", "row": 20, "col": 13}) == "MOUSE(row=20, col=13)",
          "firma MOUSE con el formato de action_display")
    check(pending_action_signature({"action": "MOUSE", "row": "x"}) == "", "MOUSE sin celda valida -> vacia")
    set_display_map({"ACTION1": "UP"})
    check(pending_action_signature({"action": "ACTION1"}) == "UP", "mapa ACTIONn -> nombre de modelo")
    set_display_map({})

    with tempfile.TemporaryDirectory() as d:
        stats = GuardStats(str(Path(d) / "blocks.jsonl"))
        est, env, g = Estado(), Entorno(), NoopGuard()
        step = make_guarded_step_env(env, "estado.json", g, est.load, stats,
                                     lambda: ["UP", "DOWN"], lambda: {"action_num": 7, "score": 0, "state": "NOT_FINISHED"})

        # 1. primera vez: pasa al entorno, no cambia -> se observa como no-op
        r1 = step({"actions": [{"action": "UP"}]})
        check(len(env.calls) == 1 and r1["executed"], "primera UP llega al entorno")
        check(g.is_known_noop(1, board_signature(tab(0)), "UP"), "UP quedo registrada como no-op en ese tablero")
        check(stats.observed == 1 and stats.noops == 1, "estadisticas de observacion")

        # 2. misma accion, mismo tablero -> BLOQUEO sin llamar al entorno
        r2 = step({"actions": [{"action": "UP"}]})
        check(len(env.calls) == 1, "la repeticion NO llega al entorno")
        check(r2["executed"] is False and r2["stop_reason"] == "known_noop", "payload de bloqueo", str(r2.get("stop_reason")))
        check("no action budget spent" in r2["stop_detail"] and r2["action_num"] == 7 and r2["valid_actions"] == ["UP", "DOWN"],
              "el payload lleva el ultimo action_num y las acciones validas")
        check(stats.blocked == 1 and (Path(d) / "blocks.jsonl").read_text().count("\n") == 1,
              "el bloqueo se cuenta y se escribe una linea JSON")

        # 3. otra accion en el mismo tablero -> pasa
        r3 = step({"actions": [{"action": "DOWN"}]})
        check(len(env.calls) == 2 and r3["executed"], "otra accion pasa")

        # 4. mismo UP pero el tablero cambio -> pasa (clave distinta)
        est.frame = F(tab(1), 1)
        r4 = step({"actions": [{"action": "UP"}]})
        check(len(env.calls) == 3 and r4["executed"], "UP sobre otro tablero pasa")

        # 5. lote de varias: se bloquea solo si la PRIMERA es no-op conocida; y no se observa.
        #    OJO: DOWN y UP ya son no-ops conocidos sobre tab(0) por los pasos 1 y 3, asi que
        #    el lote "que pasa" tiene que empezar por una accion FRESCA (SPACE).
        est.frame = F(tab(0), 1)
        n = len(env.calls)
        r5 = step({"actions": [{"action": "SPACE"}, {"action": "UP"}]})
        check(len(env.calls) == n + 1, "lote cuya primera accion no es no-op conocido: pasa entero")
        check(not g.is_known_noop(1, board_signature(tab(0)), "SPACE"),
              "en lotes multiples NO se observa (sin atribucion)")
        r6 = step({"actions": [{"action": "UP"}, {"action": "DOWN"}]})
        check(len(env.calls) == n + 1 and r6["stop_reason"] == "known_noop" and r6["requested_count"] == 2,
              "lote cuya primera accion es no-op conocido: bloqueado entero, requested_count=2")

        # 6. la accion que SI cambio algo queda desbloqueada
        env.next_changed = True
        est.frame = F(tab(2), 1)
        step({"actions": [{"action": "UP"}]})              # observa UP con cambio en tab(2)
        env.next_changed = False
        step({"actions": [{"action": "UP"}]})              # ahora sin cambio -> se registra
        check(g.is_known_noop(1, board_signature(tab(2)), "UP"), "se registra tras el no-op")
        env.next_changed = True
        est.frame = F(tab(2), 1)
        # el guard bloquearia; simulamos que anim borra el registro cuando hay evidencia de cambio:
        g.observe(level=1, board_before_sig=board_signature(tab(2)), action_sig="UP", board_changed=True)
        check(not g.is_known_noop(1, board_signature(tab(2)), "UP"), "evidencia de cambio borra el registro (semantica anim)")

        # 7. nivel distinto = clave distinta
        est.frame = F(tab(0), 2)
        r7 = step({"actions": [{"action": "UP"}]})
        check(r7["executed"], "mismo tablero pero nivel 2: no esta bloqueado")

        # 8. degradacion: si el lector de estado explota, se llama al entorno igual
        def rompe(_p):
            raise RuntimeError("estado corrupto")
        step2 = make_guarded_step_env(env, "x", g, rompe, stats)
        m = len(env.calls)
        r8 = step2({"actions": [{"action": "UP"}]})
        check(len(env.calls) == m + 1 and r8["executed"], "sin estado legible, el entorno se llama igual")

    print(f"\n{'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

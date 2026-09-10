"""Prueba local (CPU, cero cuota) del mapa cognitivo.

Comprueba lo que la revision de codigo no puede: que la nota sea correcta, corta,
y sobre todo que NO repita el error de nav — una accion animada con tablero final
identico jamas puede reportarse como inerte.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc3.cognitive_map import (  # noqa: E402
    MapRecorder,
    construir_grafo,
    render_map_note,
)

fallos: list[str] = []


def check(ok: bool, nombre: str, detalle: str = "") -> None:
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


class GuardiaFalso:
    """Imita la interfaz del NoopGuard del harness."""

    def __init__(self):
        self.vistas = []
        self.bloqueadas = set()

    def observe(self, *, level, board_before_sig, action_sig, board_changed,
                animated=False, **kw):
        self.vistas.append((level, board_before_sig, action_sig, board_changed, animated))
        if not board_changed and not animated:
            self.bloqueadas.add((level, board_before_sig, action_sig))

    def is_known_noop(self, level, board_sig, action_sig):
        return (level, board_sig, action_sig) in self.bloqueadas

    def otra_cosa(self):
        return "delegado"


def main() -> int:
    print("PRUEBA DEL MAPA COGNITIVO")

    real = GuardiaFalso()
    rec = MapRecorder(real)

    # --- 1. delegacion: el guardia real tiene que seguir funcionando igual ---
    rec.observe(level=1, board_before_sig="S1", action_sig="UP",
                board_changed=False, animated=False)
    check(len(real.vistas) == 1, "el guardia real recibe la observacion")
    check(rec.is_known_noop(1, "S1", "UP"), "el bloqueo del guardia real sigue vivo")
    check(rec.otra_cosa() == "delegado", "los atributos desconocidos se delegan")

    # --- 2. EL PUNTO CRITICO: accion animada con tablero identico ------------
    rec.observe(level=1, board_before_sig="S1", action_sig="MOUSE(row=4, col=7)",
                board_changed=False, animated=True)
    g = construir_grafo(rec.registros, 1)
    check("MOUSE" in g["utiles"],
          "una accion ANIMADA cuenta como util aunque el tablero no cambie",
          "es el error exacto que cometia _nav_shift")
    check("MOUSE" not in g["nunca_utiles"], "y no aparece como inerte")
    check(g["solo_animacion"] == 1, "se contabiliza como efecto solo-animacion")

    # --- 3. grafo: encadenado de estados ------------------------------------
    rec.reset()
    real.bloqueadas.clear()
    camino = [
        ("S1", "UP", False, False),
        ("S1", "RIGHT", True, False),
        ("S2", "RIGHT", True, False),
        ("S3", "LEFT", False, False),
        ("S3", "MOUSE(row=20, col=13)", True, False),
        ("S4", "UP", False, False),
    ]
    for antes, acc, chg, anim in camino:
        rec.observe(level=2, board_before_sig=antes, action_sig=acc,
                    board_changed=chg, animated=anim)
    g = construir_grafo(rec.registros, 2)
    check(len(g["aristas"]) == 6, "una arista por accion", str(len(g["aristas"])))
    check(g["aristas"][1][2] == "S2",
          "el destino de una arista es el estado de la siguiente observacion",
          str(g["aristas"][1]))
    check(g["estados"]["S1"] == 2, "cuenta visitas repetidas al mismo estado")
    check("UP" in g["nunca_utiles"],
          "UP nunca sirvio en el nivel -> inerte de verdad")
    check("RIGHT" in g["utiles"] and "RIGHT" not in g["nunca_utiles"],
          "RIGHT si sirvio -> no es inerte")
    check(g["celdas_utiles"] == [(20, 13)], "registra la celda de click que sirvio",
          str(g["celdas_utiles"]))

    # --- 4. aislamiento por nivel -------------------------------------------
    rec.observe(level=3, board_before_sig="T1", action_sig="DOWN",
                board_changed=True, animated=False)
    g2 = construir_grafo(rec.registros, 2)
    g3 = construir_grafo(rec.registros, 3)
    check(len(g2["aristas"]) == 6, "el nivel 2 no ve las acciones del nivel 3")
    check(len(g3["aristas"]) == 1, "el nivel 3 solo ve las suyas")

    # --- 5. la nota ---------------------------------------------------------
    nota = render_map_note(rec.registros, 2, "S1", ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"])
    print("\n--- nota generada ---")
    print(nota)
    print("--- fin ---\n")
    check(bool(nota), "la nota no viene vacia")
    check("ya probaste" in nota, "dice que se probo desde el estado actual")
    check("NO has probado" in nota and "DOWN" in nota,
          "declara la frontera de exploracion (DOWN nunca se probo en S1)")
    frontera_txt = nota.split("NO has probado:")[-1].split("\n")[0]
    check("LEFT" not in frontera_txt,
          "la frontera excluye lo ya demostrado inerte en el nivel",
          "si no, la nota se contradice consigo misma dos lineas mas abajo")
    check("MOUSE" not in nota.split("NO has probado")[-1].split("\n")[0],
          "MOUSE queda FUERA de la frontera (4096 celdas, no es enumerable)")
    check("nunca han hecho nada" in nota and "UP" in nota,
          "declara las acciones inertes del nivel")
    check("bucle" in nota, "avisa del estado repetido")

    aprox_tokens = len(nota) // 4
    check(aprox_tokens <= 140, f"la nota cabe en el presupuesto (~{aprox_tokens} tokens)",
          "los parches anteriores costaban ~311 y no pagaron")

    # --- 6. degradacion -----------------------------------------------------
    check(render_map_note([], 1, "S1", ["UP"]) == "", "sin historial, nota vacia")
    check(render_map_note(rec.registros, 9, "X", ["UP"]) == "",
          "nivel sin datos, nota vacia")
    nota_sin_validas = render_map_note(rec.registros, 2, "S1", None)
    check(isinstance(nota_sin_validas, str), "sin acciones validas no revienta")

    # --- 7. el registrador nunca puede tumbar el guardia --------------------
    class Explosivo(GuardiaFalso):
        pass

    rec2 = MapRecorder(Explosivo())
    rec2.observe(level=None, board_before_sig=None, action_sig=None,
                 board_changed=None, animated=None)
    check(len(rec2._inner.vistas) == 1,
          "con argumentos basura el guardia real sigue recibiendo la llamada")

    print(f"\n{'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

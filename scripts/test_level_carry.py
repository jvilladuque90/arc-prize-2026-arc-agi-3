"""Prueba local (CPU, cero cuota) de la consolidacion al ganar nivel."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc3.level_carry import (  # noqa: E402
    describir_cambio,
    render_carry_note,
    transiciones_ganadoras,
)

fallos: list[str] = []


def check(ok, nombre, detalle=""):
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


@dataclass
class F:
    grid: tuple
    step: int
    level: int


@dataclass
class H:
    action: str
    frame: F


def tablero(v=0, n=6):
    return tuple(tuple(v for _ in range(n)) for _ in range(n))


def con(t, cambios):
    g = [list(r) for r in t]
    for (r, c, v) in cambios:
        g[r][c] = v
    return tuple(tuple(r) for r in g)


def main() -> int:
    print("PRUEBA DE LA CONSOLIDACION AL GANAR NIVEL")

    # 1. sin cruce de nivel -> nada
    h = [H("UP", F(tablero(), 1, 1)), H("RIGHT", F(tablero(), 2, 1))]
    check(transiciones_ganadoras(h) == [], "sin cruce de nivel no hay transiciones")
    check(render_carry_note(1, []) == "", "nivel 1: nota vacia")
    check(render_carry_note(2, []) == "", "sin ganados: nota vacia aunque vayas por el 2")

    # 2. un cruce: click amarillo (11) que borra un bloque -> nivel 2
    t0 = con(tablero(4), [(2, 3, 11), (2, 4, 11), (3, 3, 11), (3, 4, 11)])
    t1 = con(t0, [(2, 3, 4), (2, 4, 4), (3, 3, 4), (3, 4, 4)])
    h = [H("UP", F(t0, 1, 1)), H("LEFT", F(t0, 2, 1)),
         H("MOUSE(row=2, col=3)", F(t1, 3, 2))]
    tr = transiciones_ganadoras(h)
    check(len(tr) == 1, "un cruce -> una transicion", str(len(tr)))
    t = tr[0]
    check(t["nivel_ganado"] == 1, "el nivel ganado es el 1")
    check(t["accion"] == "MOUSE(row=2, col=3)", "la accion ganadora es la del cruce", t["accion"])
    check(t["acciones_en_nivel"] == 3, "cuenta las acciones gastadas en ese nivel", str(t["acciones_en_nivel"]))
    check(t["color_bajo_click"] == "Y", "lee el color bajo el click ANTES de la accion", str(t["color_bajo_click"]))
    check(t["cambio"]["n"] == 4 and t["cambio"]["caja"] == (2, 3, 3, 4),
          "describe el cambio: 4 celdas, caja correcta", str(t["cambio"]))
    check(t["cambio"]["censo"] == {"Y>c": 4}, "censo de colores Y>c x4", str(t["cambio"]["censo"]))

    nota = render_carry_note(2, tr)
    print("\n--- nota (nivel 2) ---\n" + nota + "\n--- fin ---\n")
    check("MECANICAS GANADORAS" in nota, "la nota tiene cabecera")
    check("MOUSE(row=2, col=3)" in nota and "color Y" in nota, "nombra la accion y el color")
    check("4 celdas" in nota and "Y>c x4" in nota, "describe el cambio")
    check("COMPONEN" in nota, "recuerda que los niveles componen")
    check(len(nota) // 4 <= 140, f"cabe en presupuesto (~{len(nota)//4} tokens)")

    # 3. dos cruces: el segundo nivel se gana con RIGHT tras 2 acciones
    t2 = con(t1, [(0, 0, 9)])
    h2 = h + [H("DOWN", F(t1, 4, 2)), H("RIGHT", F(t2, 5, 3))]
    tr2 = transiciones_ganadoras(h2)
    check(len(tr2) == 2, "dos cruces -> dos transiciones", str(len(tr2)))
    check(tr2[1]["nivel_ganado"] == 2 and tr2[1]["accion"] == "RIGHT"
          and tr2[1]["acciones_en_nivel"] == 2,
          "segunda transicion: nivel 2, RIGHT, 2 acciones", str(tr2[1]))
    check(tr2[1]["color_bajo_click"] is None, "sin click no hay color bajo click")
    nota2 = render_carry_note(3, tr2)
    check("nivel 1:" in nota2 and "nivel 2:" in nota2, "la nota del nivel 3 lleva los dos")
    check("llevas 2 ganados" in nota2, "cuenta los ganados")

    # 4. tablero final identico en la transicion (evento de nivel sin cambio visible)
    h3 = [H("UP", F(t0, 1, 1)), H("SPACE", F(t0, 2, 2))]
    tr3 = transiciones_ganadoras(h3)
    nota3 = render_carry_note(2, tr3)
    check(tr3[0]["cambio"]["n"] == 0, "sin cambio visible: n=0")
    check("no lo interpretes como accion inerte" in nota3,
          "sin cambio visible la nota lo dice y NO marca la accion como inerte",
          "es el error de nav que este modulo no repite")

    # 5. basura: no revienta
    check(transiciones_ganadoras(None) == [], "historial None -> vacio")
    check(transiciones_ganadoras([H("X", None)]) == [], "entrada sin frame -> se ignora")
    check(render_carry_note("nivel", tr) == "", "nivel no numerico -> vacia")
    check(describir_cambio((), ()) == {"n": 0, "caja": None, "censo": {}}, "tableros vacios")

    print(f"\n{'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

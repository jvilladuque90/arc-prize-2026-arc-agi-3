"""Prueba local (CPU, cero cuota) del manual del juego."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc3.game_manual import (  # noqa: E402
    afordancias,
    color_ganador,
    componentes,
    construir_manual,
    render_manual_note,
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


def tab(v=0, n=8):
    return tuple(tuple(v for _ in range(n)) for _ in range(n))


def con(t, cambios):
    g = [list(r) for r in t]
    for r, c, v in cambios:
        g[r][c] = v
    return tuple(tuple(r) for r in g)


def main() -> int:
    print("PRUEBA DEL MANUAL DEL JUEGO")

    # componentes: dos bloques amarillos (11) separados
    t = con(tab(4), [(1, 1, 11), (1, 2, 11), (2, 1, 11), (2, 2, 11), (6, 6, 11)])
    comps = componentes(t, "Y")
    check(len(comps) == 2, "dos componentes de color Y", str(len(comps)))
    check(comps[0]["n"] == 4 and comps[0]["caja"] == (1, 1, 2, 2), "la mayor primero, caja correcta", str(comps[0]))
    check(componentes(t, "?") == [] and componentes((), "Y") == [], "color invalido o tablero vacio -> nada")

    # historial: nivel 1 con UP inerte, LEFT util, click amarillo que gana -> nivel 2
    t0 = con(tab(4), [(2, 3, 11), (2, 4, 11), (3, 3, 11), (3, 4, 11)])
    t0b = con(t0, [(0, 0, 9)])                       # LEFT cambio algo
    t1 = con(t0b, [(2, 3, 4), (2, 4, 4), (3, 3, 4), (3, 4, 4)])  # click borra el amarillo -> nivel 2
    t2 = con(tab(4), [(5, 5, 11), (5, 6, 11), (0, 7, 11)])       # nivel 2: amarillo en otro sitio
    h = [H("UP", F(t0, 1, 1)), H("LEFT", F(t0b, 2, 1)),
         H("MOUSE(row=2, col=3)", F(t1, 3, 2)), H("DOWN", F(t2, 4, 2))]

    af = afordancias(h, 2)
    check(af["simples"].get("LEFT") == 1 and "UP" not in af["simples"],
          "afordancias: LEFT tuvo efecto, UP no aparece (solo positivos)", str(af["simples"]))
    check(af["clicks"] == [(2, 3)], "el click que cambio algo queda registrado", str(af["clicks"]))
    check("DOWN" in af["simples"], "DOWN cambio el tablero al pasar al nivel 2 y cuenta")

    m = construir_manual(h, F(t2, 4, 2))
    check(m["nivel"] == 2 and len(m["transiciones"]) == 1, "manual: nivel 2, una transicion")
    check(m["color_ganador"] == "Y", "color ganador = el de bajo el click", str(m["color_ganador"]))
    check(len(m["objetos_ahora"]) == 2 and m["objetos_ahora"][0]["n"] == 2,
          "localiza el amarillo AHORA (2 componentes, la mayor de 2 celdas)", str(m["objetos_ahora"]))

    nota = render_manual_note(m)
    print("\n--- nota (nivel 2) ---\n" + nota + "\n--- fin ---\n")
    check("MANUAL DEL JUEGO" in nota, "cabecera")
    check("MOUSE(row=2, col=3)" in nota and "color Y" in nota, "consolidacion presente")
    check("esta AHORA en" in nota and "filas 5-5 cols 5-6" in nota, "localizacion del objeto ganador")
    check("Controles que SI" in nota and "LEFT" in nota and "UP" not in nota.split("Controles que SI")[1].split("\n")[0],
          "controles positivos, sin listar los inertes")
    check("Clicks que SI" in nota and "(2,3)" in nota, "clicks positivos")
    check("Primer paso sugerido" in nota, "heuristica de un paso al final")
    check(len(nota) // 4 <= 170, f"presupuesto (~{len(nota)//4} tokens)")

    # color ganador sin click: por censo
    tr = {"color_bajo_click": None, "cambio": {"censo": {"R>c": 5, "Y>c": 1}}}
    check(color_ganador(tr) == "R", "sin click, el color de origen mas frecuente del censo")

    # el color ganador no esta en el tablero nuevo
    m2 = construir_manual(h, F(tab(4), 4, 2))
    n2 = render_manual_note(m2)
    check("NO aparece en este tablero" in n2, "avisa si el objeto ganador no esta")

    # degradacion
    check(render_manual_note(construir_manual([], F(t2, 1, 1))) == "", "nivel 1 -> vacio")
    check(render_manual_note(construir_manual(h, F(t2, 4, 1))) == "", "nivel 1 con historial -> vacio")
    check(render_manual_note(construir_manual(None, None)) == "", "None -> vacio")

    print(f"\n{'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

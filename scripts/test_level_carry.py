"""Prueba local (CPU, cero cuota) de la consolidacion al ganar nivel."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc3.level_carry import (  # noqa: E402
    comprimir_racha,
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

    # --- v2: receta comprimida, eje de clics, invariante -------------------
    check(comprimir_racha(["A", "A", "A", "B"]) == "A x3 -> B", "comprime rachas",
          comprimir_racha(["A", "A", "A", "B"]))
    check(comprimir_racha([]) == "" and comprimir_racha(["", " "]) == "", "racha vacia")

    # receta real: las ultimas 5 acciones del nivel, con repeticion explicita
    t0b = con(tablero(4), [(2, 3, 11)])
    h4 = [H("UP", F(t0b, 1, 1)), H("LEFT", F(t0b, 2, 1)),
          H("MOUSE(row=9, col=43)", F(t0b, 3, 1)), H("MOUSE(row=9, col=43)", F(t0b, 4, 1)),
          H("MOUSE(row=31, col=43)", F(con(t0b, [(2, 3, 4)]), 5, 2))]
    tr4 = transiciones_ganadoras(h4)
    check(tr4[0]["receta"] == ["UP", "LEFT", "MOUSE(row=9, col=43)", "MOUSE(row=9, col=43)",
                              "MOUSE(row=31, col=43)"],
          "la receta son las ultimas 5 acciones del nivel", str(len(tr4[0]["receta"])))
    check(tr4[0]["receta_completa"] is True, "marca que la receta cubre el nivel entero")
    n4 = render_carry_note(2, tr4)
    check("RECETA del nivel 1 (asi se gano)" in n4, "la nota lleva la receta del ultimo nivel")
    check("MOUSE(row=9, col=43) x2" in n4, "la repeticion sale comprimida", n4.splitlines()[1][:80])
    check("todos en la columna 43" in n4, "detecta el eje comun de los clics de la receta")

    # receta recortada cuando el nivel fue largo
    largo = [H(f"A{i}", F(t0b, i, 1)) for i in range(8)] + [H("SPACE", F(con(t0b, [(0, 0, 9)]), 9, 2))]
    trl = transiciones_ganadoras(largo)
    check(len(trl[0]["receta"]) == 5 and trl[0]["receta_completa"] is False,
          "en un nivel largo la receta se recorta a 5 y se marca como parcial")
    check("ultimas acciones antes de ganar" in render_carry_note(2, trl),
          "y la nota lo dice, no finge que sea la receta entera")

    # invariante: dos niveles ganados con el mismo tipo de accion Y el mismo color.
    # El color se lee del tablero ANTERIOR a la accion ganadora, asi que la entrada
    # previa a cada cruce debe tener Y (11) en la celda que se clica.
    amarillo = con(tablero(4), [(2, 2, 11)])
    hi = [H("inicio", F(amarillo, 1, 1)),                               # nivel 1, Y en (2,2)
          H("MOUSE(row=2, col=2)", F(tablero(4), 2, 2)),                # cruce: gana nivel 1
          H("ruido", F(amarillo, 3, 2)),                                # nivel 2, Y en (2,2)
          H("MOUSE(row=2, col=2)", F(tablero(4), 4, 3))]                # cruce: gana nivel 2
    tri = transiciones_ganadoras(hi)
    check([t["color_bajo_click"] for t in tri] == ["Y", "Y"],
          "ambos cruces leen color Y bajo el click", str([t["color_bajo_click"] for t in tri]))
    ni = render_carry_note(3, tri)
    check("INVARIANTE" in ni and "niveles 1, 2" in ni,
          "declara el invariante cuando dos niveles se ganan igual", ni)
    check("no una casualidad" in ni, "y lo nombra como mecanica del juego")
    # con un solo nivel ganado no hay invariante
    check("INVARIANTE" not in render_carry_note(2, tr4), "con un solo nivel no hay invariante")

    ap = len(n4) // 4
    check(ap <= 170, f"la nota v2 sigue en presupuesto (~{ap} tokens)",
          "v1 eran ~93; el manual que fallo, ~160 de contenido no ganado")

    # --- v3: decaimiento (hipotesis de Julian: la memoria vieja sesga) -------
    am = con(tablero(4), [(2, 2, 11)])
    hd = [H("ini", F(am, 1, 1)), H("MOUSE(row=2, col=2)", F(tablero(4), 2, 2)),
          H("x", F(am, 3, 2)), H("MOUSE(row=2, col=2)", F(tablero(4), 4, 3)),
          H("y", F(am, 5, 3)), H("MOUSE(row=2, col=2)", F(tablero(4), 6, 4))]
    trd = transiciones_ganadoras(hd)
    check(len(trd) == 3, "tres cruces en el juego profundo", str(len(trd)))
    sin_dec = render_carry_note(4, trd)
    con_dec = render_carry_note(4, trd, decaimiento=True)
    check(sin_dec.count("- nivel ") == 3, "sin decaimiento: detalle de los 3 niveles",
          str(sin_dec.count("- nivel ")))
    check(con_dec.count("- nivel ") == 1, "con decaimiento: detalle SOLO del ultimo",
          str(con_dec.count("- nivel ")))
    check("- nivel 3:" in con_dec and "- nivel 1:" not in con_dec,
          "el detallado es el mas reciente, no el mas viejo")
    check("antes ganaste los niveles 1, 2, todos con MOUSE sobre color Y" in con_dec,
          "lo viejo se colapsa a UNA linea de esencia (tipo de mecanica, sin celdas)")
    check("RECETA" not in con_dec, "el decaimiento quita la receta literal (el anclaje medido)")
    check("INVARIANTE" in con_dec, "pero conserva el invariante, que es la generalizacion")
    check(len(con_dec) < len(sin_dec) * 0.75,
          f"recorta >=25% en juego profundo ({100*(1-len(con_dec)/len(sin_dec)):.0f}%)")
    # caso mayoritario medido (76% de las notas): un solo nivel ganado
    uno = render_carry_note(2, trd[:1], decaimiento=True)
    check("antes ganaste" not in uno, "con un solo nivel no inventa linea de esencia")
    check(len(uno) // 4 <= 100, f"y la nota queda en ~{len(uno)//4} tokens (v1 medida: ~93)")
    # esencia con mecanicas distintas: no finge un invariante que no existe
    hm = [H("ini", F(am, 1, 1)), H("MOUSE(row=2, col=2)", F(tablero(4), 2, 2)),
          H("z", F(tablero(4), 3, 2)), H("UP", F(con(tablero(4), [(0, 0, 9)]), 4, 3)),
          H("w", F(tablero(4), 5, 3)), H("LEFT", F(con(tablero(4), [(1, 1, 9)]), 6, 4))]
    trm = transiciones_ganadoras(hm)
    nm = render_carry_note(4, trm, decaimiento=True)
    check("mecanicas distintas" in nm, "si las mecanicas viejas difieren, lo dice", nm.splitlines()[1][:90])
    check("INVARIANTE" not in nm, "y no declara invariante donde no lo hay")

    # 5. basura: no revienta
    check(transiciones_ganadoras(None) == [], "historial None -> vacio")
    check(transiciones_ganadoras([H("X", None)]) == [], "entrada sin frame -> se ignora")
    check(render_carry_note("nivel", tr) == "", "nivel no numerico -> vacia")
    check(describir_cambio((), ()) == {"n": 0, "caja": None, "censo": {}}, "tableros vacios")

    print(f"\n{'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

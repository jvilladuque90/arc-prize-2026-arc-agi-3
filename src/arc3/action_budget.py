"""Presupuesto de acciones: empujar el agrupamiento SOLO donde hay hueco.

POR QUE (DESIGN 8.66)
---------------------
La restriccion que manda no es el razonamiento: son las ACCIONES. Sobre 225 partidas:

- de 140 juegos que completan el nivel 1 y se atascan en el siguiente, el **76,4% apenas
  lo intento** (mediana 0,16x su baseline) y solo el **2,1% se perdio** (>=2x);
- el nivel 1 lo resuelven **por debajo del baseline** (mediana 0,73x): juegan bien;
- las 225 partidas terminan en `cancelled`: siempre se acaba el reloj;
- acciones disponibles por juego **46** (mediana) frente a un baseline de nivel 1+2 de
  **83**: en **22 de 25 juegos es aritmeticamente imposible** llegar al nivel 2.

El coste esta en los tokens por accion (~582-744) y el rendimiento del servicio ya esta
cerrado (+0,3%, DESIGN 8.57-8.59). Pero el agente **ya puede mandar varias acciones en un
turno** —`action(['LEFT','LEFT','DOWN'])` es legal— y lo hace con una dispersion enorme:
`re86` 10,4 acciones por turno frente a `su15` 1,4. Mediana 3,5.

Llevar esa mediana a 7 duplicaria el presupuesto **sin recortar el razonamiento**: no se
piensa menos, se reparte el coste del turno entre mas jugadas.

POR QUE CONDICIONAL Y NO DE CADA TURNO
--------------------------------------
La regla destilada de cinco brazos de prompt (DESIGN 8.41) es: *texto minimo, ganado y raro
paga; generico, abundante o de cada turno, no*. Un recordatorio fijo seria justo lo que ya
sabemos que no paga. Asi que la nota **solo aparece cuando el propio agente esta agrupando
por debajo del umbral**, y lleva sus numeros reales. Donde ya agrupa bien (`re86`, `tn36`),
calla.

QUE NO HACE
-----------
No dice que acciones mandar ni cuantas: solo hace explicito el coste del turno, que el
agente no puede ver. La decision sigue siendo suya.
"""

from __future__ import annotations

MINIMO_TURNOS = 3        # antes de eso no hay ratio fiable
UMBRAL = 3.0             # mediana del banco: por debajo, hay hueco
CONTADOR = "_arc3_turnos"


def ratio(acciones: int, turnos: int) -> float:
    return acciones / turnos if turnos else 0.0


def render_nota_presupuesto(acciones: int, turnos: int) -> str:
    """Nota vacia salvo que el agente este agrupando poco."""
    try:
        acciones, turnos = int(acciones), int(turnos)
    except (TypeError, ValueError):
        return ""
    if turnos < MINIMO_TURNOS or acciones <= 0:
        return ""
    r = ratio(acciones, turnos)
    if r >= UMBRAL:
        return ""
    return (
        "PRESUPUESTO: esta partida se corta por RELOJ, no por numero de acciones, y cada "
        "turno tuyo cuesta lo mismo lleve una accion o cinco. Llevas "
        f"{acciones} acciones en {turnos} turnos ({r:.1f} por turno). "
        "Cuando las siguientes jugadas sean predecibles, mandalas juntas: "
        "action(['LEFT','LEFT','LEFT']) gasta un turno, no tres."
    )


def turnos_de(agente) -> int:
    """Lleva la cuenta de turnos en el propio agente (uno por partida)."""
    n = getattr(agente, CONTADOR, 0) + 1
    try:
        setattr(agente, CONTADOR, n)
    except Exception:
        return 0
    return n

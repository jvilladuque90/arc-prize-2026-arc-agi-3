"""Modelo de efectos MEDIDO a partir del historial: la carga del seam C.

QUE PROBLEMA RESUELVE. La v1 de la amplificacion inyecto una *funcion*
(`plan_moves`) por el prelude del sandbox. Se adopto (726 llamadas en 25/25
juegos desde una sola linea de nota), pero la carga estaba mal elegida por dos
razones:

  1. COSTABA UN TURNO. El modelo tenia que escribir codigo y ejecutarlo para
     enterarse de algo que nosotros ya podiamos calcular por el.
  2. ERA VACIA FUERA DE JUEGOS DE MOVIMIENTO. Si nada se traslada, `plan_moves`
     no devolvia nada util y el turno se perdia entero.

Este modulo corrige las dos. Calcula la tabla de efectos por accion leyendo los
`HistoryEntry` que el agente YA tiene (`action`, `frame.grid`), asi que:

  - cuesta CERO turnos y CERO llamadas al sandbox: el dato llega ya masticado
    en el prompt (seam C, `_build_user_prompt`);
  - NUNCA esta vacia: si una accion no traslada nada, informa igualmente si
    cambia el tablero ("cambia") o si no hace nada ("sin efecto"). Saber que
    ACTION5 es inerte en este juego es informacion accionable — deja de gastarse
    presupuesto en ella.

La deteccion de traslacion excluye el color de fondo (mayoritario). Sin esa
exclusion el detector sigue al fondo y aprende los signos invertidos: ese bug
real costo una tarde y lo cazo scripts/test_sandbox_nav.py antes de gastar GPU.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

MAX_SHIFT = 8          # radio de busqueda de la traslacion
MIN_CELLS = 1          # 1 basta: el jugador de UNA celda es el caso mas comun
                       # (medido: tu93 y ft09 cambian exactamente 2 celdas por paso
                       # = una se vacia y otra se llena). Exigir 2 los descartaba a
                       # todos. El emparejamiento por color ya filtra el ruido: si
                       # los colores no casan, no hay candidato y devuelve None.
MAX_DIFF_CELLS = 400   # mas cambio que esto = repintado/nivel nuevo, no traslacion
MIN_CONF = 0.6         # umbral MEDIDO por prediccion fuera de muestra sobre los 25
                       # juegos locales: conf>=0.6 -> 96.6% (141/146); conf>=0.5 ->
                       # 88.1%; sin filtro -> 85.1%. El salto entre 0.6 y 0.5 es
                       # limpio, asi que ahi se corta.
SIMPLE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")


def _rows(grid: Any) -> list[list[int]]:
    if grid is None:
        return []
    try:
        return [[int(v) for v in row] for row in grid]
    except (TypeError, ValueError):
        return []


def background_of(grid: list[list[int]]) -> int:
    freq: Counter = Counter()
    for row in grid:
        freq.update(row)
    return freq.most_common(1)[0][0] if freq else 0


def shift_between(before: Any, after: Any) -> tuple[int, int] | None:
    """Traslacion (dr, dc) que explica el cambio, o None si no la hay.

    SOLO MIRA LAS CELDAS QUE CAMBIARON. Esta es la correccion que hizo falta:
    los tableros reales son densos (medido: 630-855 celdas no-fondo de 4096), asi
    que buscar el desplazamiento que mejor alinea TODO el conjunto no-fondo es
    ajustar ruido — sobre una textura densa siempre hay algun offset que alinea
    muchas celdas por casualidad. Dos detectores construidos asi se contradecian
    entre si en casi todos los pares, y uno reportaba el MISMO desplazamiento para
    cuatro acciones distintas.

    Lo que de verdad ocurre cuando un objeto se mueve: unas pocas celdas se vacian
    (huella vieja) y otras pocas se llenan (huella nueva); las 600+ restantes no
    cambian. Alineando esas dos huellas el problema queda determinado.

    Los candidatos se generan de los propios pares de igual color (no de un barrido
    de caja), asi que detecta saltos grandes sin coste extra.
    """
    b, a = _rows(before), _rows(after)
    if not b or not a or len(b) != len(a) or len(b[0]) != len(a[0]):
        return None
    changed: list[tuple[int, int]] = []
    for r in range(len(b)):
        row_b, row_a = b[r], a[r]
        for c in range(min(len(row_b), len(row_a))):
            if row_b[c] != row_a[c]:
                changed.append((r, c))
    if not changed or len(changed) > MAX_DIFF_CELLS:
        return None

    # Conteo global: el objeto que se mueve es lo RARO; el campo es lo abundante.
    # Sirve de desempate cuando dos lecturas son geometricamente validas (que la
    # barra de color 9 avanzo, o que el hueco de color 0 retrocedio).
    total: Counter = Counter()
    for row in b:
        total.update(row)

    best_shift, best_hits, best_rarity = None, 0, None
    for color in {b[r][c] for r, c in changed}:
        src = [(r, c) for r, c in changed if b[r][c] == color]
        dst = {(r, c) for r, c in changed if a[r][c] == color}
        if len(src) < MIN_CELLS or len(dst) < MIN_CELLS:
            continue
        cand: Counter = Counter()
        for sr, sc in src:
            for dr_, dc_ in dst:
                d = (dr_ - sr, dc_ - sc)
                if d != (0, 0) and abs(d[0]) <= MAX_SHIFT and abs(d[1]) <= MAX_SHIFT:
                    cand[d] += 1
        if not cand:
            continue
        shift, _ = cand.most_common(1)[0]
        hits = sum(1 for r, c in src if (r + shift[0], c + shift[1]) in dst)
        # la traslacion debe explicar la mayoria de la huella mas pequena
        if hits * 2 < min(len(src), len(dst)):
            continue
        rarity = total.get(color, 0)
        if best_shift is None or (rarity, -hits) < (best_rarity, -best_hits):
            best_shift, best_hits, best_rarity = shift, hits, rarity
    return best_shift


def effects_from_history(entries: list[Any], window: int = 40) -> dict[str, dict]:
    """Agrega el efecto observado de cada accion sobre pares consecutivos.

    `entries` son HistoryEntry (`action: str`, `frame.grid`). Devuelve por accion:
      {"kind": "move"|"cambia"|"sin efecto", "shift": [dr,dc]|None, "n": veces, "conf": 0..1}

    VENTANA DE RECENCIA (`window`): solo las ultimas transiciones. Los efectos son
    dependientes del estado — medido: varios juegos arrancan en una pantalla donde
    NINGUNA accion simple hace nada, y solo despues responden. Sin ventana, ese
    "todo inerte" del arranque envenenaria el consejo durante el resto de la
    partida. Con ventana, la tabla sigue al juego.
    """
    if window and len(entries or []) > window + 1:
        entries = entries[-(window + 1):]
    obs: dict[str, list[tuple[int, int] | None]] = {}
    changed: dict[str, list[bool]] = {}
    prev = None
    for e in entries or []:
        grid = getattr(getattr(e, "frame", None), "grid", None)
        action = getattr(e, "action", None)
        if grid is None:
            continue
        if prev is not None and action:
            b, a = _rows(prev), _rows(grid)
            if b and a:
                changed.setdefault(action, []).append(b != a)
                obs.setdefault(action, []).append(shift_between(prev, grid))
        prev = grid

    table: dict[str, dict] = {}
    for action, shifts in obs.items():
        n = len(shifts)
        real = [s for s in shifts if s is not None]
        ch = changed.get(action, [])
        if real:
            top, cnt = Counter(real).most_common(1)[0]
            if cnt * 2 >= len(real):
                table[action] = {"kind": "move", "shift": [top[0], top[1]],
                                 "n": n, "conf": round(cnt / n, 2)}
                continue
        if ch and not any(ch):
            table[action] = {"kind": "sin efecto", "shift": None, "n": n, "conf": 1.0}
        else:
            frac = (sum(ch) / len(ch)) if ch else 0.0
            table[action] = {"kind": "cambia", "shift": None, "n": n,
                             "conf": round(frac, 2)}
    return table


def dir_words(sr: int, sc: int) -> str:
    """(0,-3) -> '3 a la izquierda'.

    NO es cosmetica. Medido en el banco micro con Qwen3-4B sobre 109 problemas de
    planificacion: con el vector crudo ("move 0 -3") acierta 66.1%; con la misma
    informacion en palabras, 86.2%. Pareado, 24 items a favor de las palabras
    contra 2 (p aprox 0). Interpretar el vector consume razonamiento que el modelo
    necesita para la tarea; nombrar la direccion se lo devuelve. El formato gana
    ademas en los DOS tamanos probados (1.7B y 4B), asi que la direccion es
    estructural del prompt y no un rasgo de un modelo concreto.
    """
    partes = []
    if sr < 0:
        partes.append(f"{-sr} arriba")
    if sr > 0:
        partes.append(f"{sr} abajo")
    if sc < 0:
        partes.append(f"{-sc} a la izquierda")
    if sc > 0:
        partes.append(f"{sc} a la derecha")
    return " y ".join(partes) if partes else "nada"


def render_effects_note(table: dict[str, dict], min_obs: int = 2) -> str:
    """Convierte la tabla en las lineas de texto que se anexan al prompt.

    Solo se reportan acciones con observaciones suficientes: afirmar un efecto
    a partir de una sola muestra es como decidir un experimento con n=1, que es
    exactamente el error que nos costo cuatro noches.
    """
    if not table:
        return ""
    lines = []
    for action in sorted(table):
        d = table[action]
        if d["n"] < min_obs:
            continue
        if d["kind"] == "move" and d["conf"] >= MIN_CONF:
            dr, dc = d["shift"]
            lines.append(f"  {action}: mueve {dir_words(dr, dc)}"
                         f"  [{d['n']} obs, {d['conf']:.0%} consistente]")
        elif d["kind"] == "move":
            # Medido: por debajo de MIN_CONF la prediccion fuera de muestra cae a
            # ~60%. Afirmar ahi meteria hechos FALSOS en el prompt, que es peor que
            # callar. Se degrada a incertidumbre honesta.
            lines.append(f"  {action}: cambia el tablero, pero su efecto NO es "
                         f"constante ({d['conf']:.0%} de {d['n']} obs) — verifica antes de fiarte")
        elif d["kind"] == "sin efecto":
            lines.append(f"  {action}: SIN EFECTO en {d['n']} intentos "
                         f"— no gastes turnos en ella")
        else:
            lines.append(f"  {action}: cambia el tablero sin trasladar nada "
                         f"[{d['n']} obs]")
    if not lines:
        return ""
    note = ("Efecto MEDIDO de cada accion en ESTA partida (calculado de tu propio "
            "historial, no lo recalcules):\n" + "\n".join(lines))

    # Caso medido en 5 de 25 juegos locales: NINGUNA accion simple hace nada, pero
    # el tablero si responde a clics (s5i5 y vc33: 12/12 clics cambian el tablero).
    # Ahi el agente puede quemar la partida entera pulsando botones muertos, asi
    # que la conclusion se dice explicitamente en vez de dejarla deducir.
    simples = [a for a in table if a in SIMPLE_ACTIONS]
    if simples and all(table[a]["kind"] == "sin efecto" for a in simples):
        note += ("\n  => NINGUNA accion simple hace nada aqui: este juego se controla "
                 "por coordenadas (ACTION6). Deja de gastar turnos en ACTION1-5.")
    return note

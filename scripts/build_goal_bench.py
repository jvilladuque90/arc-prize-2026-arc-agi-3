"""Construye los items de INFERENCIA DE META desde las trazas ganadoras.

LA PREGUNTA QUE MIDE (candidato fuerte de STRATEGY §10): el banco ya probo que
el modelo sabe MOVERSE con la tabla de efectos (90.9%); lo que nunca se ha
medido es si sabe HACIA DONDE — inferir el objetivo del nivel. Si no sabe,
ninguna mejora de mecanica lo arregla, y la siguiente carga del seam C tiene
que atacar ahi.

VERDAD VERIFICABLE. De cada subida de nivel registrada por capture_traces:
  - juegos de movimiento: la meta = posicion final del objeto movil en el estado
    que disparo la subida (identificado con el mismo detector validado al 96.6%).
  - juegos de clic: la meta = la celda exacta del MOUSE(...) que disparo la subida.

FORMATO: eleccion multiple entre celdas candidatas (la verdadera + distractores
que son centros de componentes reales del tablero, lejos de la verdadera). Se
eligio eleccion multiple tras las lecciones del parser: una respuesta abierta
"que celda?" mediria el formato de respuesta tanto como la inferencia.

Dos brazos pareados sobre los mismos items:
  I.V0_inicio     : tablero INICIAL del nivel -> ¿cual celda es la meta?
  I.V2_trayecto   : tablero a MITAD del intento + resumen del trayecto recorrido
                    (computable en produccion desde el historial) -> misma pregunta

Uso: python scripts/build_goal_bench.py [--traces traces_goal.json traces_goal_2.json]
                                        [--out goal_bench.jsonl]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3.effects_model import background_of, shift_between  # noqa: E402

MIN_DIST = 8          # Chebyshev minimo entre candidatas: la verdad del objeto
                      # movil es exacta salvo el ultimo empujon (<= un shift), asi
                      # que con candidatas separadas la imprecision no puede
                      # cambiar la respuesta correcta
N_CANDS = 4


def moved_footprint(before, after):
    """Celdas del objeto que se traslado (color y posiciones en `after`), o None.

    Reusa el criterio del detector validado: por color, sobre celdas cambiadas.
    """
    s = shift_between(before, after)
    if s is None:
        return None
    changed = [(r, c) for r in range(len(before)) for c in range(len(before[r]))
               if before[r][c] != after[r][c]]
    for color in {after[r][c] for r, c in changed}:
        src = [(r, c) for r, c in changed if before[r][c] == color]
        dst = [(r, c) for r, c in changed if after[r][c] == color]
        hits = sum(1 for r, c in src if (r + s[0], c + s[1]) in set(dst))
        if dst and hits * 2 >= len(dst):
            return {"color": color, "cells": dst}
    return None


def object_at_levelup(event):
    """Posicion (centroide) del objeto movil en el estado que disparo la subida."""
    win = event.get("window") or []
    grids = [event["start_grid"]] + [g for _, g in win]
    last = None
    for i in range(len(grids) - 1, 0, -1):
        fp = moved_footprint(grids[i - 1], grids[i])
        if fp:
            last = fp
            break
    if not last:
        return None
    rs = [p[0] for p in last["cells"]]
    cs = [p[1] for p in last["cells"]]
    return (round(sum(rs) / len(rs)), round(sum(cs) / len(cs)), last["color"])


def component_centroids(grid, bg, exclude, min_dist):
    """Centros de componentes conexas no-fondo, lejos de `exclude` (distractores)."""
    seen = [[False] * len(row) for row in grid]
    out = []
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if seen[r][c] or grid[r][c] == bg:
                continue
            color = grid[r][c]
            stack, cells = [(r, c)], []
            seen[r][c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < len(grid) and 0 <= nc < len(grid[nr])
                            and not seen[nr][nc] and grid[nr][nc] == color):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if len(cells) < 2:
                continue
            cy = round(sum(p[0] for p in cells) / len(cells))
            cx = round(sum(p[1] for p in cells) / len(cells))
            if max(abs(cy - exclude[0]), abs(cx - exclude[1])) >= min_dist:
                out.append((cy, cx))
    return out


def items_from_event(game, event, rng):
    """0, 1 o 2 items (inicio / trayecto) por evento de subida de nivel."""
    trigger = event.get("trigger", "")
    m = re.match(r"MOUSE\((\d+),\s*(\d+)\)", trigger)
    if m:
        goal = (int(m.group(1)), int(m.group(2)))
        kind, obj_desc = "click", "la celda que hay que CLICAR"
    else:
        pos = object_at_levelup(event)
        if pos is None:
            return []
        goal = (pos[0], pos[1])
        kind, obj_desc = "move", f"la celda a la que debe llegar el objeto (color {pos[2]})"

    grid0 = event["start_grid"]
    bg = background_of(grid0)
    distract = component_centroids(grid0, bg, goal, MIN_DIST)
    if len(distract) < N_CANDS - 1:
        return []
    cands = [list(goal)] + [list(d) for d in rng.sample(distract, N_CANDS - 1)]
    rng.shuffle(cands)

    win = event.get("window") or []
    base = {"game": game, "kind": kind,
            "goal_desc": obj_desc, "candidates": cands,
            "answer": f"{goal[0]} {goal[1]}", "level": event["level"]}
    items = [dict(base, type="goal_inicio", board=grid0, trail=None)]

    # brazo trayecto: estado a mitad de la ventana + resumen del recorrido
    if kind == "move" and len(win) >= 6:
        mid = len(win) // 2
        mid_grid = win[mid][1]
        p0 = object_at_levelup({"start_grid": grid0, "window": win[:2]})
        pm = object_at_levelup({"start_grid": grid0, "window": win[:mid + 1]})
        if p0 and pm:
            trail = (f"el objeto empezo cerca de [{p0[0]}, {p0[1]}] y tras "
                     f"{mid} acciones va por [{pm[0]}, {pm[1]}]")
            items.append(dict(base, type="goal_trayecto", board=mid_grid, trail=trail))
    return items


def goal_signature(event):
    """Firma de la meta de un nivel GANADO: (tipo, color de la celda objetivo).

    Para clics, el color de la celda clicada; para movimiento, el color que habia
    BAJO la posicion final del objeto en el tablero inicial del nivel (el marcador
    que el objeto fue a cubrir). Medido en las trazas: la firma es 100% consistente
    entre niveles en los 4 juegos multinivel de la primera cosecha (12 subidas,
    cero excepciones) — por eso es inyectable: lo ganado en el nivel k ensena el
    objetivo del k+1.
    """
    m = re.match(r"MOUSE\((\d+),\s*(\d+)\)", event.get("trigger", ""))
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        return ("clic", event["pre_grid"][r][c])
    pos = object_at_levelup(event)
    if pos is None:
        return None
    r, c = pos[0], pos[1]
    g = event["start_grid"]
    if not (0 <= r < len(g) and 0 <= c < len(g[0])):
        return None
    return ("mov", g[r][c])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+",
                    default=["traces_goal.json", "traces_goal_2.json"])
    ap.add_argument("--out", default="goal_bench.jsonl")
    args = ap.parse_args()

    rng = random.Random(11)
    items = []
    for tf in args.traces:
        path = ROOT / tf
        if not path.exists():
            print(f"  (no existe {tf}, lo salto)")
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            events = rec.get("events", [])
            for k, ev in enumerate(events):
                got = items_from_event(rec["game"], ev, rng)
                items.extend(got)
                # brazo FIRMA: la pista del nivel ANTERIOR ganado (k>=1), que es
                # exactamente lo que el anfitrion puede inyectar en produccion
                if k >= 1:
                    firma = goal_signature(events[k - 1])
                    if firma:
                        verbo = ("clicando una celda" if firma[0] == "clic"
                                 else "llevando el objeto hasta una celda")
                        for it in items_from_event(rec["game"], ev, rng):
                            if it["type"] != "goal_inicio":
                                continue
                            it2 = dict(it, type="goal_firma",
                                       firma=f"El nivel anterior de este juego se "
                                             f"completo {verbo} de color {firma[1]}.")
                            items.append(it2)

    stats = Counter((i["kind"], i["type"]) for i in items)
    resp = Counter(i["answer"] for i in items)
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    print(f"{len(items)} items -> {args.out}  {dict(stats)}")
    if items:
        top, n = resp.most_common(1)[0]
        print(f"base trivial (respuesta mas comun): {n}/{len(items)} = {n/len(items):.0%}")
    return 0 if items else 1


if __name__ == "__main__":
    raise SystemExit(main())

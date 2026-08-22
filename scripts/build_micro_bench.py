"""Construye el BANCO MICRO: preguntas con respuesta verificable sobre la mecánica.

POR QUE. Nuestro único instrumento fiable era el envío diario: **un dato por noche**,
con varianza 0.41. Con eso hacen falta 3-4 días para distinguir dos configuraciones,
y llevamos cuatro experimentos sin norte. El banco micro cambia la escala: mide
**directamente lo que nos falta** — que el agente infiera la mecánica del juego — con
cientos de preguntas por minuto y respuestas objetivamente verificables.

QUE MIDE. No niveles completados (eso exige un modelo grande y horas). Mide el paso
anterior, que es donde está el cuello segun docs/DESIGN.md §8.9:

  1. effect_of_action  — visto un puñado de transiciones, ¿qué hace ACTION_N?
     respuesta verificada: "nada" | "mueve (dr,dc)" | "cambia el tablero"
  2. which_action      — ¿qué acción mueve al objeto hacia arriba/abajo/izq/der?
  3. predict_position  — dada la posición del objeto y una acción, ¿dónde queda?

Las tres se puntúan por coincidencia exacta contra la verdad calculada del propio
environment, así que no hace falta juez humano ni modelo evaluador.

PARA QUE SIRVE. Permite comparar ESTRATEGIAS DE PROMPT (con/sin lista de objetos,
con/sin tabla de efectos medidos) en minutos y con decenas de items, en vez de gastar
una noche por variante. La variante ganadora es la carga del seam C.

Uso:
  python scripts/build_micro_bench.py --games ls20 tu93 sc25 cd82 --out micro_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3.effects_model import shift_between as _nav_shift  # noqa: E402
from arc3.effects_model import MIN_CONF  # noqa: E402
from arc3.env import LocalGame, discover_environments  # noqa: E402

# NOTA (2026-08-19): la primera version del banco derivo su verdad con
# arc3.sandbox_nav._nav_shift, que alinea TODO el conjunto de celdas no-fondo.
# Sobre tableros densos (630-855 celdas no-fondo de 4096) eso ajusta ruido: llego a
# reportar el MISMO desplazamiento para cuatro acciones distintas. El banco quedaba
# con respuestas "move DR DC" inventadas. Ahora la verdad sale de
# arc3.effects_model.shift_between, que empareja huellas POR COLOR sobre las celdas
# que cambiaron y esta validado por prediccion fuera de muestra (96.6% con conf>=0.6,
# ver scripts/validate_effects_model.py).

SIMPLE_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
REPEATS = 8          # veces que se prueba cada acción (4 daba una confianza muy
                     # ruidosa: con n=4 un solo desplazamiento espurio ya es 25%)
DIRECTIONS = {"arriba": (-1, 0), "abajo": (1, 0), "izquierda": (0, -1), "derecha": (0, 1)}

# Metas para las preguntas de planificacion: ejes puros y diagonales a varias
# distancias, para que el brazo B no dependa de un puñado de casos.
PLAN_OFFSETS = [(dr, dc)
                for d in (3, 5, 7, 9, 12)
                for dr, dc in ((-d, 0), (d, 0), (0, -d), (0, d),
                               (-d, d), (d, -d), (-d, -d), (d, d))]


def dentro(grid, r: int, c: int) -> bool:
    """La meta tiene que caber en el tablero.

    Sin esto el 9% de los items de planificacion pedian llegar a casillas
    inexistentes (p.ej. columna 66 en una rejilla de 0..63). Un enunciado
    imposible no mide planificacion: mide como reacciona el modelo a un dato
    incoherente, que es otra cosa.
    """
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def grid_of(frame) -> list[list[int]]:
    """Ultima capa del frame como lista de listas de int (llega como numpy)."""
    if frame is None:
        return []
    f = getattr(frame, "frame", None)
    if f is None or len(f) == 0:
        return []
    return [[int(v) for v in row] for row in f[-1]]


def ascii_grid(grid, legend="WwgGcBMPRbSYOrNp") -> str:
    return "\n".join("".join(legend[v] if 0 <= v < len(legend) else "?" for v in row)
                     for row in grid)


def crop(grid, r, c, radius=6):
    """Recorte alrededor de (r,c): el tablero completo son 4096 celdas y no cabe."""
    r0, r1 = max(0, r - radius), min(len(grid), r + radius + 1)
    c0, c1 = max(0, c - radius), min(len(grid[0]), c + radius + 1)
    return [row[c0:c1] for row in grid[r0:r1]], (r0, c0)


def objects_of(grid, background):
    """Componentes conexas 4-conectadas distintas del fondo (para la variante con features)."""
    seen = [[False] * len(row) for row in grid]
    out = []
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if seen[r][c] or grid[r][c] == background:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < len(grid) and 0 <= nc < len(grid[nr])
                            and not seen[nr][nc] and grid[nr][nc] == color):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            out.append({"color": color, "size": len(cells),
                        "center": [round(sum(rs) / len(rs)), round(sum(cs) / len(cs))]})
    out.sort(key=lambda o: -o["size"])
    return out[:12]


def background_of(grid):
    freq = {}
    for row in grid:
        for v in row:
            freq[v] = freq.get(v, 0) + 1
    return max(freq, key=freq.get) if freq else 0


def probe_game(info) -> dict:
    """Sondea un juego: cada acción simple varias veces desde el estado inicial."""
    from arcengine import GameAction

    game = LocalGame(info)
    frame = game.reset()
    if frame is None:
        return {}
    base_grid = grid_of(frame)
    transitions = []
    for name in SIMPLE_ACTIONS:
        try:
            act = getattr(GameAction, name)
        except AttributeError:
            continue
        game.reset()
        prev = grid_of(game.env.latest_frame) if hasattr(game.env, "latest_frame") else base_grid
        for _ in range(REPEATS):
            nxt_frame = game.step(act)
            if nxt_frame is None:
                break
            nxt = grid_of(nxt_frame)
            if prev and nxt:
                transitions.append({"action": name, "before": prev, "after": nxt})
            prev = nxt
    return {"game": info.game_id.split("-")[0], "base": base_grid, "transitions": transitions}


def truth_for_action(trans_of_action) -> dict:
    """Verdad del efecto de una acción: nada / traslación consistente / otro cambio."""
    changed = 0
    shifts = {}
    for t in trans_of_action:
        if t["before"] != t["after"]:
            changed += 1
        s = _nav_shift(t["before"], t["after"])
        if s is not None:
            shifts[s] = shifts.get(s, 0) + 1
    n = len(trans_of_action)
    if n == 0:
        return {"kind": "unknown"}
    if changed == 0:
        return {"kind": "none"}
    if shifts:
        best = max(shifts, key=shifts.get)
        # Umbral de confianza sobre TODAS las observaciones (no solo las que
        # dieron desplazamiento). Es el mismo 0.6 validado por prediccion fuera
        # de muestra: por debajo, la traslacion no se sostiene y la respuesta
        # correcta es "change", no un vector inventado.
        if shifts[best] / n >= MIN_CONF:
            return {"kind": "move", "shift": [best[0], best[1]]}
    return {"kind": "change"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["ls20", "tu93", "sc25", "cd82"])
    ap.add_argument("--out", default="micro_bench.jsonl")
    ap.add_argument("--cap-per-class", type=int, default=20,
                    help="tope de items por clase de respuesta en effect_of_action (0=sin tope)")
    args = ap.parse_args()

    infos = {i.game_id.split("-")[0]: i for i in discover_environments(ROOT / "environment_files")}
    items = []
    for gname in args.games:
        if gname not in infos:
            print(f"  {gname}: no encontrado"); continue
        probe = probe_game(infos[gname])
        if not probe or not probe["transitions"]:
            print(f"  {gname}: sin transiciones"); continue

        by_action = {}
        for t in probe["transitions"]:
            by_action.setdefault(t["action"], []).append(t)
        truths = {a: truth_for_action(ts) for a, ts in by_action.items()}
        moves = {a: v["shift"] for a, v in truths.items() if v["kind"] == "move"}
        bg = background_of(probe["base"])
        print(f"  {gname}: {len(probe['transitions'])} transiciones, "
              f"efectos {[(a, v['kind']) for a, v in truths.items()]}")

        # --- pregunta 1: efecto de cada acción (una por acción con verdad conocida)
        for action, truth in truths.items():
            if truth["kind"] == "unknown":
                continue
            ts = by_action[action][:3]
            shots = []
            for t in ts:
                center = None
                s = _nav_shift(t["before"], t["after"])
                for r in range(len(t["before"])):
                    for c in range(len(t["before"][r])):
                        if t["before"][r][c] != t["after"][r][c]:
                            center = (r, c)
                            break
                    if center:
                        break
                if center is None:
                    center = (len(t["before"]) // 2, len(t["before"][0]) // 2)
                cb, org = crop(t["before"], center[0], center[1])
                ca, _ = crop(t["after"], center[0], center[1])
                # Objetos DEL RECORTE Y EN COORDENADAS DEL RECORTE.
                # Antes se adjuntaba la lista de objetos del tablero COMPLETO y del
                # frame inicial, con centros en coordenadas absolutas ([31,31],
                # [63,31]) mientras el enunciado mostraba un recorte de 13x13 de OTRA
                # transicion. Las features describian una vista distinta de la que se
                # veia, asi que el brazo A no medía "ayudan las features objetuales"
                # sino "ayuda una lista irrelevante" — y eso da que no por definicion.
                shots.append({"before": ascii_grid(cb), "after": ascii_grid(ca),
                              "origin": list(org), "moved": list(s) if s else None,
                              "objects": objects_of(cb, background_of(cb))[:6]})
            answer = (truth["kind"] if truth["kind"] != "move"
                      else f"move {truth['shift'][0]} {truth['shift'][1]}")
            items.append({
                "game": probe["game"], "type": "effect_of_action", "action": action,
                "shots": shots, "objects": objects_of(probe["base"], bg),
                "effects_table": {a: (v["kind"] if v["kind"] != "move"
                                      else f"move {v['shift'][0]} {v['shift'][1]}")
                                  for a, v in truths.items()},
                "answer": answer,
            })

        # --- pregunta 3: PLANIFICAR con la tabla de efectos (la que prueba el seam C)
        # Aquí la tabla NO filtra la respuesta: da las primitivas, pero el modelo debe
        # componerlas contra una meta. Es exactamente lo que queremos que mejore.
        if len(moves) >= 2:
            player = None
            for t in probe["transitions"]:
                s = _nav_shift(t["before"], t["after"])
                if s is not None:
                    for r in range(len(t["after"])):
                        for c in range(len(t["after"][r])):
                            if t["before"][r][c] != t["after"][r][c] and t["after"][r][c] != bg:
                                player = (r, c)
                                break
                        if player:
                            break
                if player:
                    break
            if player:
                # Rejilla amplia de metas: cada (dr,dc) es un problema de
                # planificacion distinto sobre la MISMA tabla medida. Con 6 offsets
                # el brazo B se quedaba en 18 items — demasiado poco para decidir
                # nada, que es justo el error de muestra pequena que venimos
                # arrastrando. Se filtran despues por ganador unico y estricto.
                for dr, dc in PLAN_OFFSETS:
                    target = (player[0] + dr, player[1] + dc)
                    if not dentro(probe["base"], *target):
                        continue
                    dist0 = abs(dr) + abs(dc)
                    gains = {}
                    for a, s in moves.items():
                        nd = abs(dr - s[0]) + abs(dc - s[1])
                        gains[a] = dist0 - nd
                    best = max(gains, key=gains.get)
                    # solo items con ganador ÚNICO y estrictamente positivo
                    if gains[best] <= 0:
                        continue
                    if sum(1 for a in gains if gains[a] == gains[best]) != 1:
                        continue
                    items.append({
                        "game": probe["game"], "type": "plan_action",
                        # tablero COMPLETO en ASCII: permite recrear el regimen de
                        # produccion (prompt largo con la rejilla entera) en el
                        # experimento G, en vez de la pregunta desnuda de ~86 tokens
                        "board": ascii_grid(probe["base"]),
                        "player": list(player), "target": list(target),
                        "shots": [{"action": a, "moved": list(by_action_shift)}
                                  for a, by_action_shift in moves.items()],
                        "objects": objects_of(probe["base"], bg),
                        "effects_table": {a: f"move {s[0]} {s[1]}" for a, s in moves.items()},
                        "answer": best,
                    })

        # --- pregunta 4: EVITAR ACCIONES INERTES (la otra mitad de la carga)
        # La nota que inyectamos tiene dos mitades y solo una estaba medida. Esta
        # es la segunda: marcar las acciones que NO hacen nada. Importa porque en
        # 5 de los 25 juegos locales NINGUNA accion simple tiene efecto, y ahi el
        # agente puede gastar la partida entera pulsando botones muertos.
        # La pregunta desplegable no es "¿ayuda saberlo?" (trivialmente si) sino
        # "¿vale sus tokens DECIRLO, o basta con omitir esas acciones?".
        inertes = [a for a, v in truths.items() if v["kind"] == "none"]
        if moves and inertes and player:
            for dr, dc in PLAN_OFFSETS:
                target = (player[0] + dr, player[1] + dc)
                if not dentro(probe["base"], *target):
                    continue
                dist0 = abs(dr) + abs(dc)
                gains = {a: dist0 - (abs(dr - s[0]) + abs(dc - s[1]))
                         for a, s in moves.items()}
                best = max(gains, key=gains.get)
                if gains[best] <= 0 or sum(1 for a in gains
                                           if gains[a] == gains[best]) != 1:
                    continue
                items.append({
                    "game": probe["game"], "type": "avoid_inert",
                    "player": list(player), "target": list(target),
                    "effects_table": {a: f"move {s[0]} {s[1]}" for a, s in moves.items()},
                    "inert_actions": sorted(inertes),
                    "answer": best,
                })

        # --- pregunta 2: qué acción mueve en cada dirección
        for label, vec in DIRECTIONS.items():
            # DESAMBIGUACION (2026-08-19): antes se exigia coincidencia EXACTA de
            # signos en los dos ejes, lo que dejaba pasar items imposibles. Ejemplo
            # real: ACTION2=[3,0] y ACTION3=[3,-3], se preguntaba "cual baja?" y se
            # esperaba ACTION2 — pero ACTION3 tambien baja. Dos modelos de tamanos
            # distintos fallaban exactamente los mismos items: la culpa era de la
            # pregunta, no del modelo.
            # Ahora "mueve hacia D" = tiene componente positiva en D, y el item solo
            # se conserva si EXACTAMENTE UNA accion la tiene.
            hits = [a for a, s in moves.items() if s[0] * vec[0] + s[1] * vec[1] > 0]
            if len(hits) == 1:
                ts = [t for a in moves for t in by_action[a][:2]]
                shots = []
                for t in ts[:6]:
                    s = _nav_shift(t["before"], t["after"])
                    shots.append({"action": t["action"], "moved": list(s) if s else None})
                items.append({
                    "game": probe["game"], "type": "which_action", "direction": label,
                    "shots": shots, "objects": objects_of(probe["base"], bg),
                    "effects_table": {a: f"move {s[0]} {s[1]}" for a, s in moves.items()},
                    "answer": hits[0],
                })

    # --- equilibrado de clases en effect_of_action
    # Con la verdad corregida, "change" se lleva 76 de 125: responder siempre
    # "change" acertaria el 60.8% y el brazo A no podria distinguir nada. Se topa
    # cada clase para que ninguna domine; no se inventa ni se altera ninguna
    # respuesta, solo se recorta el exceso de las mayoritarias.
    if args.cap_per_class:
        seen: dict[str, int] = {}
        kept = []
        for i in items:
            if i["type"] != "effect_of_action":
                kept.append(i)
                continue
            k = i["answer"]
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= args.cap_per_class:
                kept.append(i)
        drop = len(items) - len(kept)
        if drop:
            print(f"  equilibrado: {drop} items de effect_of_action recortados "
                  f"(tope {args.cap_per_class}/clase)")
        items = kept

    out = ROOT / args.out
    out.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    kinds = {}
    for i in items:
        kinds[i["type"]] = kinds.get(i["type"], 0) + 1
    print(f"\n{len(items)} items -> {out}  {kinds}")
    return 0 if items else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Manual del juego: la consolidacion al ganar nivel, ampliada a manual curado por el
anfitrion (meta inferida, controles con efecto probado, y donde esta AHORA el objeto que
gano el nivel anterior).

POR QUE
-------
- La consolidacion (src/arc3/level_carry.py) es la unica palanca que ha pasado la
  compuerta de 60 min sobre la base NVFP4 (26 niveles / 4.392 vs 23 / 4.114). Este
  modulo la extiende, no la sustituye.
- Research 2026: un agente sembrado con "manuales" por juego (meta, controles,
  mecanicas, condicion de victoria) escritos por un agente mas fuerte hizo 96.42 en
  los 25 juegos publicos con un tercio de los tokens. El manual es la representacion
  que transfiere. Aqui el manual lo escribe el ANFITRION a partir de la evidencia del
  propio juego, no un modelo externo.
- Objetualidad y analogia estructural (core knowledge de Chollet; structure mapping):
  el humano llega al nivel 2 buscando "el mismo objeto de antes, en otro sitio". El
  anfitrion puede localizarlo: componentes conexas del color que gano el nivel N en
  el tablero actual del nivel N+1.
- "Intuitive Gamer" (Tenenbaum): heuristicas simples orientadas a la meta, no busqueda
  profunda. La nota cierra con una instruccion de un paso, no con un plan.

QUE NO REPITE
-------------
- Cero escritura exigida al modelo; solo tokens de ENTRADA y solo desde el nivel 2.
- Solo se reportan afordancias POSITIVAS ("esto SI hizo algo"); nunca se etiqueta una
  accion como muerta por el fotograma final (el error de nav). Sobre el harness
  original la unica senal es el fotograma final, y reportar solo positivos es seguro.
- Todo degrada a cadena vacia: cualquier fallo -> el prompt del padre intacto.
"""

from __future__ import annotations

try:  # importable como paquete...
    from arc3.level_carry import LETRAS, _celda, _letra, describir_cambio, transiciones_ganadoras
except ImportError:  # ...o ejecutado en el mismo namespace que level_carry (celda del kernel)
    pass

MAX_OBJETOS = 3
MAX_CELDAS = 6


def _grid(frame):
    g = getattr(frame, "grid", None)
    return g if g else ()


def _nivel(frame) -> int:
    try:
        return int(getattr(frame, "level", 1) or 1)
    except (TypeError, ValueError):
        return 1


def componentes(grid, color_letra: str) -> list[dict]:
    """Componentes 4-conexas de un color (por letra de la leyenda) en un tablero.
    Devuelve [{'caja': (r0,c0,r1,c1), 'n': celdas}], mayores primero."""
    try:
        color = LETRAS.index(color_letra)
    except (ValueError, TypeError):
        return []
    filas = len(grid)
    if not filas:
        return []
    cols = max(len(r) for r in grid)
    visto = set()
    out = []
    for r in range(filas):
        for c in range(len(grid[r])):
            if grid[r][c] != color or (r, c) in visto:
                continue
            pila = [(r, c)]
            visto.add((r, c))
            celdas = []
            while pila:
                y, x = pila.pop()
                celdas.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < filas and 0 <= nx < len(grid[ny])
                            and grid[ny][nx] == color and (ny, nx) not in visto):
                        visto.add((ny, nx))
                        pila.append((ny, nx))
            rs = [y for y, _ in celdas]
            cs = [x for _, x in celdas]
            out.append({"caja": (min(rs), min(cs), max(rs), max(cs)), "n": len(celdas)})
    out.sort(key=lambda o: -o["n"])
    return out


def afordancias(history_entries, nivel_actual: int) -> dict:
    """Controles con efecto PROBADO en este juego (todos los niveles hasta ahora):
    acciones simples que alguna vez cambiaron el tablero, y celdas de click que lo
    hicieron. Solo positivos."""
    simples: dict[str, int] = {}
    clicks: list[tuple[int, int]] = []
    prev = None
    for e in history_entries or []:
        fr = getattr(e, "frame", None)
        if fr is None:
            continue
        acc = " ".join(str(getattr(e, "action", "") or "").split())
        if prev is not None and _grid(prev) and _grid(fr) != _grid(prev):
            nombre = acc.split("(")[0].upper()
            if nombre == "MOUSE":
                c = _celda(acc)
                if c and c not in clicks:
                    clicks.append(c)
            elif nombre:
                simples[nombre] = simples.get(nombre, 0) + 1
        prev = fr
    return {"simples": simples, "clicks": clicks}


def color_ganador(transicion: dict) -> str | None:
    """El color del objeto con el que se gano: el de bajo el click si lo hubo; si no,
    el color de origen mas frecuente en el censo de cambios."""
    if transicion.get("color_bajo_click"):
        return transicion["color_bajo_click"]
    censo = (transicion.get("cambio") or {}).get("censo") or {}
    if not censo:
        return None
    k = max(censo.items(), key=lambda kv: kv[1])[0]
    origen = k.split(">")[0]
    return origen if origen in LETRAS else None


def construir_manual(history_entries, current_frame) -> dict:
    nivel = _nivel(current_frame) if current_frame is not None else 1
    trans = transiciones_ganadoras(history_entries or [])
    ultima = trans[-1] if trans else None
    color = color_ganador(ultima) if ultima else None
    objetos = componentes(_grid(current_frame), color)[:MAX_OBJETOS] if (color and current_frame is not None) else []
    return {
        "nivel": nivel,
        "transiciones": trans,
        "afordancias": afordancias(history_entries, nivel),
        "color_ganador": color,
        "objetos_ahora": objetos,
    }


def render_manual_note(manual: dict) -> str:
    nivel = manual.get("nivel", 1)
    trans = manual.get("transiciones") or []
    if nivel <= 1 or not trans:
        return ""
    lineas = [f"MANUAL DEL JUEGO (lo escribe el anfitrion con lo ganado hasta ahora; vas por el "
              f"nivel {nivel}, llevas {len(trans)} ganados):"]
    # 1. como se gano el ultimo nivel (la consolidacion validada)
    t = trans[-1]
    partes = [f"- Ultimo nivel ganado ({t['nivel_ganado']}): con {t['accion'] or '?'} tras "
              f"{t['acciones_en_nivel']} acciones"]
    if t.get("color_bajo_click"):
        partes.append(f"(click sobre color {t['color_bajo_click']})")
    cam = t.get("cambio") or {}
    if cam.get("n"):
        caja = cam["caja"]
        censo = sorted(cam["censo"].items(), key=lambda kv: -kv[1])[:3]
        partes.append(f"; cambiaron {cam['n']} celdas en filas {caja[0]}-{caja[2]}, "
                      f"cols {caja[1]}-{caja[3]}: " + ", ".join(f"{k} x{v}" for k, v in censo))
    lineas.append(" ".join(partes))
    # 2. donde esta AHORA el objeto de ese color (analogia estructural)
    color = manual.get("color_ganador")
    objs = manual.get("objetos_ahora") or []
    if color and objs:
        sitios = ", ".join(f"filas {o['caja'][0]}-{o['caja'][2]} cols {o['caja'][1]}-{o['caja'][3]} "
                           f"({o['n']} celdas)" for o in objs)
        lineas.append(f"- El color {color} que gano el nivel anterior esta AHORA en: {sitios}")
    elif color:
        lineas.append(f"- El color {color} que gano el nivel anterior NO aparece en este tablero: "
                      f"la mecanica puede haber cambiado de objeto")
    # 3. controles con efecto probado en este juego (solo positivos)
    af = manual.get("afordancias") or {}
    simples = af.get("simples") or {}
    clicks = af.get("clicks") or []
    if simples:
        lineas.append("- Controles que SI han tenido efecto en este juego: " +
                      ", ".join(f"{k} x{v}" for k, v in sorted(simples.items(), key=lambda kv: -kv[1])[:6]))
    if clicks:
        if len(clicks) <= MAX_CELDAS:
            txt = ", ".join(f"({r},{c})" for r, c in clicks)
        else:
            rs = [r for r, _ in clicks]
            cs = [c for _, c in clicks]
            txt = f"{len(clicks)} celdas en filas {min(rs)}-{max(rs)}, cols {min(cs)}-{max(cs)}"
        lineas.append(f"- Clicks que SI han tenido efecto en este juego: {txt}")
    # 4. la heuristica de un paso (Intuitive Gamer), no un plan
    lineas.append("Primer paso sugerido: aplica la mecanica ganadora al objeto de arriba y mira "
                  "que cambia; los niveles componen la mecanica anterior con otra disposicion.")
    return "\n".join(lineas)

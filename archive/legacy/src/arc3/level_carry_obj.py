"""Consolidacion v4: la mecanica ganadora contada en OBJETOS, no en teclas.

POR QUE (DESIGN 8.61, medido sobre 722 turnos con nota)
-------------------------------------------------------
Las notas v1/v2/v3 nombran la ACCION que gano el nivel ("subio con LEFT"). Tres
medidas dicen que eso no sirve:

1. Conductual: tras ganar el nivel 1, el reuso de la accion ganadora es 55,2% y
   43,9% en dos replicas CON nota y 47,5% SIN nota. Las replicas encierran al
   control (pareado 5/3/4, p=0,727).
2. Atencional: en el 85-89% de los turnos el razonamiento NO menciona la nota.
3. La causa: en 8 de 19 juegos la accion "ganadora" ya es >=80% de TODAS las
   acciones del juego, seis de ellos al 100%. Decirle "ganaste con ACTION6" a un
   agente que solo usa ACTION6 es informacion cero.

Y lo que el modelo SI hace, literal, en el turno en que recibe "subio con LEFT":

    "Level 1 solved: charcoal piece overlapped the yellow target. Now level 2."

Ya consolida la mecanica, pero en vocabulario de OBJETOS Y RELACIONES. Que es,
ademas, la vista primaria que el anfitrion le da: `current_frame.segmentation`.

QUE CAMBIA RESPECTO A v1
------------------------
SOLO el vocabulario, no la cantidad. Mismo presupuesto (~90 tokens), misma
posicion, mismo disparador (nivel 2 en adelante). Si v4 mueve algo y v1/v2/v3 no,
la causa esta aislada.

COMO SE MANTIENE LA CORRESPONDENCIA
-----------------------------------
Se llama a `segment_layer` DEL PROPIO HARNESS (`inference.utils.segmentation`),
la misma funcion que alimenta `current_frame.segmentation`, con la misma tabla
`ARC_COLOR_CHARS`. Asi los `hash` que la nota cita son literalmente los que el
modelo puede buscar. Si esa importacion falla NO se inventa una segmentacion
propia (daria hashes distintos y la nota enganaria): se devuelve nota vacia.
"""

from __future__ import annotations

LETRAS = "WwgGcBMPRbSYOrNp"
HASH_CORTO = 8          # el propio modelo escribe n['hash'][:6] en sus sondas
MAX_CELDAS = 1 << 14    # cortafuegos de coste: no segmentar tableros absurdos
# El FONDO tambien es una componente conexa, y casi siempre la mayor: sin este filtro
# el protagonista de la nota acaba siendo "W de 247 celdas" en todos los juegos. El
# prompt del harness avisa de lo mismo ("background colors are often white or
# gray/black-ish large regions"). Se descarta por area, no por color, que es la unica
# forma fiable: hay juegos con fondo negro, gris o de color.
FRACCION_FONDO = 0.20


def _segmentar(grid):
    """La segmentacion DEL HARNESS, o None. Nunca una propia: los hash deben casar."""
    if not grid:
        return None
    filas = len(grid)
    cols = len(grid[0]) if filas else 0
    if filas * cols > MAX_CELDAS:
        return None
    try:
        from inference.utils.segmentation import segment_layer
        from inference.utils.grid_utils import ARC_COLOR_CHARS
        return segment_layer(grid, ARC_COLOR_CHARS)
    except Exception:
        return None


def _caja(nodo):
    b = nodo.get("boundary") or []
    if not b:
        return None
    rs = [p[0] for p in b]
    cs = [p[1] for p in b]
    return (min(rs), min(cs), max(rs), max(cs))


def _por_hash(seg) -> dict:
    out: dict = {}
    for n in seg["nodes"]:
        out.setdefault(n["hash"], []).append(n)
    return out


def _vecinos(seg, nodo_id) -> list:
    porid = {n["id"]: n for n in seg["nodes"]}
    out = []
    for i, j in seg.get("adjacency_list") or []:
        otro = j if i == nodo_id else (i if j == nodo_id else None)
        if otro is not None and otro in porid:
            out.append(porid[otro])
    return out


def _etiqueta(n) -> str:
    return f"{n['color']} de {n['pixels']} celdas"


def _celda_click(accion: str):
    t = accion or ""
    if "row=" not in t or "col=" not in t:
        return None
    try:
        return (int(t.split("row=", 1)[1].split(",", 1)[0].strip(") ")),
                int(t.split("col=", 1)[1].split(",", 1)[0].strip(") ")))
    except (ValueError, IndexError):
        return None


def _contiene(caja, celda) -> bool:
    if not caja or not celda:
        return False
    r0, c0, r1, c1 = caja
    return r0 <= celda[0] <= r1 and c0 <= celda[1] <= c1


def describir_objetos(antes_grid, despues_grid, accion: str = "") -> dict:
    """Que le paso a los OBJETOS en la transicion que gano el nivel.

    Devuelve {'movido', 'desaparecido', 'aparecido', 'nuevos_vecinos'}; cualquiera
    puede ser None/vacio. Se queda con UN protagonista para no inflar la nota.
    """
    vacio = {"movido": None, "desaparecido": None, "aparecido": None, "nuevos_vecinos": []}
    sa, sd = _segmentar(antes_grid), _segmentar(despues_grid)
    if sa is None or sd is None:
        return vacio
    celdas = sum(len(f) for f in antes_grid) or 1
    fondo = lambda n: n.get("pixels", 0) >= FRACCION_FONDO * celdas  # noqa: E731
    ha, hd = _por_hash(sa), _por_hash(sd)
    click = _celda_click(accion)

    # 1. Movidos: mismo hash, unico en ambos fotogramas, caja distinta.
    movidos = []
    for h, na in ha.items():
        nd = hd.get(h)
        if not nd or len(na) != 1 or len(nd) != 1 or fondo(nd[0]):
            continue
        ca, cd = _caja(na[0]), _caja(nd[0])
        if ca and cd and ca != cd:
            movidos.append((na[0], nd[0], (cd[0] - ca[0], cd[1] - ca[1])))

    # 2. Desaparecidos / aparecidos: por conteo de hash, sin el fondo.
    fuera = [ha[h][0] for h in ha
             if len(ha[h]) > len(hd.get(h, [])) and not fondo(ha[h][0])]
    dentro = [hd[h][0] for h in hd
              if len(hd[h]) > len(ha.get(h, [])) and not fondo(hd[h][0])]

    def preferencia(n):
        """El que esta bajo el click manda; si no, el mas grande."""
        return (1 if _contiene(_caja(n), click) else 0, n.get("pixels", 0))

    movido = max(movidos, key=lambda t: preferencia(t[1]), default=None)
    desaparecido = max(fuera, key=preferencia, default=None)
    aparecido = max(dentro, key=preferencia, default=None)

    # 3. Vecinos NUEVOS del protagonista movido (la relacion que cambio).
    nuevos = []
    if movido is not None:
        antes_v = {(n["color"], n["pixels"]) for n in _vecinos(sa, movido[0]["id"])}
        for n in _vecinos(sd, movido[1]["id"]):
            if (n["color"], n["pixels"]) not in antes_v and not fondo(n):
                nuevos.append(n)
        nuevos.sort(key=lambda n: -n.get("pixels", 0))
    return {"movido": movido, "desaparecido": desaparecido,
            "aparecido": aparecido, "nuevos_vecinos": nuevos[:2]}


def _linea_movido(movido) -> str:
    na, nd, (dr, dc) = movido
    partes = []
    if dr:
        partes.append(f"{abs(dr)} filas {'abajo' if dr > 0 else 'arriba'}")
    if dc:
        partes.append(f"{abs(dc)} cols {'derecha' if dc > 0 else 'izquierda'}")
    desp = " y ".join(partes) if partes else "de sitio"
    return (f"- se movio el objeto {_etiqueta(nd)} (hash {nd['hash'][:HASH_CORTO]}) {desp}")


def render_nota_objetos(nivel_actual, transiciones) -> str:
    """Nota v4. `transiciones` = salida de `transiciones_objeto`; vacia en el nivel 1.

    Se detalla SOLO el ultimo nivel ganado (es lo unico que se segmenta, y acota el
    coste a dos floodfills por turno). Los anteriores, una linea de recuento.
    """
    try:
        nivel_actual = int(nivel_actual)
    except (TypeError, ValueError):
        return ""
    if nivel_actual <= 1 or not transiciones:
        return ""
    ult = transiciones[-1]
    obj = ult.get("objetos") or {}
    movido, fuera, dentro = obj.get("movido"), obj.get("desaparecido"), obj.get("aparecido")
    vecinos = obj.get("nuevos_vecinos") or []
    if not any((movido, fuera, dentro)):
        return ""      # sin historia de objetos no se inventa nada: mejor nada que ruido

    lineas = [f"MECANICA GANADORA del nivel {ult['nivel_ganado']} (la calcula el anfitrion "
              f"sobre la MISMA segmentacion que ves en current_frame.segmentation):"]
    if movido is not None:
        lineas.append(_linea_movido(movido))
        if vecinos:
            lineas.append("  y paso a tocar: " + ", ".join(_etiqueta(n) for n in vecinos))
    if fuera is not None:
        lineas.append(f"- desaparecio el objeto {_etiqueta(fuera)} "
                      f"(hash {fuera['hash'][:HASH_CORTO]})")
    if dentro is not None and fuera is None:
        lineas.append(f"- aparecio el objeto {_etiqueta(dentro)} "
                      f"(hash {dentro['hash'][:HASH_CORTO]})")
    if len(transiciones) > 1:
        lineas.append(f"- (antes ganaste {len(transiciones) - 1} nivel(es) mas)")
    lineas.append("Busca esos objetos aqui por color+celdas o por prefijo de hash: "
                  "la relacion que los unio suele repetirse con otra disposicion.")
    return "\n".join(lineas)


def _nivel(frame) -> int:
    try:
        return int(getattr(frame, "level", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _grid(frame):
    g = getattr(frame, "grid", None)
    return g if g else ()


def transiciones_objeto(history_entries) -> list[dict]:
    """Un registro por nivel completado. Solo se SEGMENTA el ultimo: el resto solo
    se cuenta, que es lo que la nota necesita de ellos."""
    marcas: list[dict] = []
    if not history_entries:
        return marcas
    nivel_prev = 1
    frame_prev = None
    for i, e in enumerate(history_entries):
        fr = getattr(e, "frame", None)
        if fr is None:
            continue
        niv = _nivel(fr)
        if niv > nivel_prev:
            marcas.append({
                "nivel_ganado": nivel_prev,
                "accion": " ".join(str(getattr(e, "action", "") or "").split()),
                "antes": frame_prev,
                "despues": fr,
            })
            nivel_prev = niv
        frame_prev = fr
    if not marcas:
        return marcas
    ult = marcas[-1]
    ult["objetos"] = describir_objetos(_grid(ult["antes"]), _grid(ult["despues"]),
                                       ult.get("accion", ""))
    return marcas

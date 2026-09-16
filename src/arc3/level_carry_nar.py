"""Consolidacion v6: el contenido de v5, contado como v4 (narrativa causal).

POR QUE (DESIGN 8.63)
---------------------
v4 (efecto, contenido ERRONEO) subio la mencion de la nota en el razonamiento al 31,9%
frente al 11-15% de v1/v2/v3. v5 (precondicion, contenido CORRECTO: mediana 10 celdas en
vez de 44, y 6,6% de objetos >=100 celdas en vez de 38,9%) la hundio al 9,4%: pareado
1-13 de 14 juegos contra v4 (p=0,0018) e indistinguible de v1 (5/7, p=0,774).

Como v5 CONSERVA el vocabulario de objetos y aun asi pierde todo el efecto, lo que pagaba
en v4 no era el vocabulario. Quedan tres sospechosos, y v6 los prueba juntos porque son
exactamente lo que separa la redaccion de v4 de la de v5:

 1. la NARRATIVA CAUSAL: "se movio X y paso a tocar Y" (sujeto, verbo, consecuencia)
    frente al inventario de v5 ("respondia a tus jugadas: X; en contacto con: Y");
 2. la SALVEDAD EPISTEMICA que anadi en v5 por honestidad ("lo de justo despues no se
    puede observar"), unico elemento que v5 ANADE en vez de quitar;
 3. el cierre imperativo ("repite esa jugada") frente al descriptivo de v5.

QUE NO CAMBIA: absolutamente nada del contenido. Este modulo se GENERO a partir de
`level_carry_pre.py` sustituyendo UNICAMENTE la funcion que redacta; toda la extraccion
(`objeto_controlado`, `objeto_bajo_click`, `contactos`, `describir_posicion`,
`marcas_victoria` y sus auxiliares) es byte a byte la de v5, y hay un test que lo
comprueba comparando el codigo fuente y las salidas de los dos modulos.

HONESTIDAD DE LA REDACCION: v6 ordena en secuencia hechos observados -- el objeto respondia
a las jugadas, en el tablero previo a la victoria estaba en contacto con tal otro, y el
nivel se completo -- pero NO afirma ningun fotograma intermedio que el entorno no entregue.
Es la misma informacion de v5 con la sintaxis de v4.
"""

from __future__ import annotations

HASH_CORTO = 8
MAX_CELDAS = 1 << 14
# El fondo tambien es componente conexa y casi siempre la mayor: sin este filtro el
# protagonista sale "W de 247 celdas" en todos los juegos (bug cazado por los tests de
# v4). Se descarta por AREA, no por color: hay juegos con fondo negro, gris o de color.
FRACCION_FONDO = 0.20


def _segmentar(grid):
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


def _caja(n):
    b = n.get("boundary") or []
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


def _es_fondo(n, celdas) -> bool:
    return n.get("pixels", 0) >= FRACCION_FONDO * celdas


def objeto_controlado(anterior_grid, antes_grid):
    """El objeto que se movio entre los dos ultimos tableros DEL MISMO NIVEL.

    Ambos fotogramas son del nivel N, asi que aqui la diferencia SI es el efecto de una
    jugada real (a diferencia de v1-v4, que cruzaban la frontera de nivel).
    """
    sa, sb = _segmentar(anterior_grid), _segmentar(antes_grid)
    if sa is None or sb is None:
        return None
    celdas = sum(len(f) for f in antes_grid) or 1
    ha, hb = _por_hash(sa), _por_hash(sb)
    movidos = []
    for h, na in ha.items():
        nb = hb.get(h)
        if not nb or len(na) != 1 or len(nb) != 1 or _es_fondo(nb[0], celdas):
            continue
        ca, cb = _caja(na[0]), _caja(nb[0])
        if ca and cb and ca != cb:
            movidos.append((nb[0], (cb[0] - ca[0], cb[1] - ca[1])))
    if not movidos:
        return None
    return max(movidos, key=lambda t: t[0].get("pixels", 0))


def objeto_bajo_click(antes_grid, accion: str):
    """Sobre que objeto apuntaba la jugada ganadora, si fue un clic."""
    celda = _celda_click(accion)
    if celda is None:
        return None
    seg = _segmentar(antes_grid)
    if seg is None:
        return None
    celdas = sum(len(f) for f in antes_grid) or 1
    r, c = celda
    mejor = None
    for n in seg["nodes"]:
        caja = _caja(n)
        if not caja or _es_fondo(n, celdas):
            continue
        r0, c0, r1, c1 = caja
        if r0 <= r <= r1 and c0 <= c <= c1:
            if mejor is None or n["pixels"] < mejor["pixels"]:
                mejor = n          # el mas ajustado al punto, no el contenedor
    return mejor


def contactos(antes_grid, nodo):
    """Con que estaba en contacto el protagonista en el tablero de la victoria."""
    if nodo is None:
        return []
    seg = _segmentar(antes_grid)
    if seg is None:
        return []
    celdas = sum(len(f) for f in antes_grid) or 1
    # el nodo puede venir de otra segmentacion: se relocaliza por hash
    ids = [n["id"] for n in seg["nodes"] if n["hash"] == nodo["hash"]]
    if not ids:
        return []
    out = [n for n in _vecinos(seg, ids[0]) if not _es_fondo(n, celdas)]
    out.sort(key=lambda n: -n.get("pixels", 0))
    return out[:2]


def describir_posicion(anterior_grid, antes_grid, accion: str) -> dict:
    """Todo lo observable del momento de ganar, sin cruzar la frontera de nivel."""
    ctrl = objeto_controlado(anterior_grid, antes_grid) if anterior_grid else None
    clic = objeto_bajo_click(antes_grid, accion)
    protagonista = (ctrl[0] if ctrl else None) or clic
    return {"controlado": ctrl, "clic": clic, "contactos": contactos(antes_grid, protagonista)}


def render_nota_narrativa(nivel_actual, marcas) -> str:
    """Los datos de v5, con la sintaxis de v4: una secuencia, no un inventario."""
    try:
        nivel_actual = int(nivel_actual)
    except (TypeError, ValueError):
        return ""
    if nivel_actual <= 1 or not marcas:
        return ""
    ult = marcas[-1]
    pos = ult.get("posicion") or {}
    ctrl, clic, cont = pos.get("controlado"), pos.get("clic"), pos.get("contactos") or []
    if not ctrl and not clic:
        return ""

    lineas = [f"ASI GANASTE el nivel {ult['nivel_ganado']} "
              f"(leido en la misma segmentacion que ves):"]
    contra = cont[0] if cont else None
    if ctrl is not None:
        n, (dr, dc) = ctrl
        partes = []
        if dr:
            partes.append(f"{abs(dr)} filas")
        if dc:
            partes.append(f"{abs(dc)} cols")
        paso = " y ".join(partes) or "de sitio"
        linea = (f"- moviste el objeto {_etiqueta(n)} (hash {n['hash'][:HASH_CORTO]}), "
                 f"{paso} por jugada")
        if contra is not None:
            linea += f", hasta ponerlo en contacto con el objeto {_etiqueta(contra)}"
        lineas.append(linea + ", y el nivel cayo")
    elif clic is not None:
        linea = f"- clicaste sobre el objeto {_etiqueta(clic)} (hash {clic['hash'][:HASH_CORTO]})"
        if contra is not None:
            linea += f", que tocaba al objeto {_etiqueta(contra)}"
        lineas.append(linea + ", y el nivel cayo")
    if ctrl is not None and clic is not None and clic["hash"] != ctrl[0]["hash"]:
        lineas.append(f"- la jugada final apunto sobre el objeto {_etiqueta(clic)} "
                      f"(hash {clic['hash'][:HASH_CORTO]})")
    if len(marcas) > 1:
        lineas.append(f"- (antes ganaste {len(marcas) - 1} nivel(es) mas)")
    lineas.append("Repite esa jugada aqui: busca los mismos objetos por color+celdas o "
                  "prefijo de hash y llevalos a la misma relacion.")
    return '\n'.join(lineas)


def _nivel(frame) -> int:
    try:
        return int(getattr(frame, "level", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _grid(frame):
    g = getattr(frame, "grid", None)
    return g if g else ()


def marcas_victoria(history_entries) -> list[dict]:
    """Un registro por nivel completado. Solo se analiza el ultimo, y SOLO con
    fotogramas del nivel N: `antes` es el ultimo tablero del nivel y `anterior` el
    penultimo. Nunca se toca el primer tablero del nivel siguiente."""
    marcas: list[dict] = []
    if not history_entries:
        return marcas
    nivel_prev = 1
    frame_prev = None
    frame_prev2 = None
    for e in history_entries:
        fr = getattr(e, "frame", None)
        if fr is None:
            continue
        niv = _nivel(fr)
        if niv > nivel_prev:
            # frame_prev y frame_prev2 son del nivel que se acaba de ganar
            anterior = frame_prev2 if (frame_prev2 is not None and frame_prev is not None
                                       and _nivel(frame_prev2) == _nivel(frame_prev)) else None
            marcas.append({
                "nivel_ganado": nivel_prev,
                "accion": " ".join(str(getattr(e, "action", "") or "").split()),
                "anterior": anterior,
                "antes": frame_prev,
            })
            nivel_prev = niv
            frame_prev2 = None
            frame_prev = fr
            continue
        frame_prev2 = frame_prev
        frame_prev = fr
    if not marcas:
        return marcas
    ult = marcas[-1]
    ult["posicion"] = describir_posicion(_grid(ult["anterior"]) if ult["anterior"] else None,
                                         _grid(ult["antes"]) if ult["antes"] else (),
                                         ult.get("accion", ""))
    return marcas

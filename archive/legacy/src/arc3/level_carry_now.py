"""Consolidacion v7: la nota anclada en la VISTA ACTUAL.

POR QUE (DESIGN 8.64)
---------------------
Cuatro brazos han medido el enganche de la nota (cuanto la referencia el razonamiento):

    v1 16,8% | v2 18,8% | v3 19,5% | v4 37,2% | v5 14,9% | v6 15,6%

Solo v4 se movio (14-2 contra v1, p=0,0042), y las dos explicaciones que teniamos han
caido: no era el VOCABULARIO de objetos (v5 lo conserva y pierde el efecto) ni la FORMA
narrativa (v6 la restaura y sigue perdido). La NOVEDAD tambien queda descartada: v5 y v6
son tan nuevas como v4 y ninguna movio nada.

Lo que queda, y tiene mecanismo: v4 describia la diferencia **cruzando la frontera de
nivel**. Era un error factual, pero su "despues" **es el tablero que el modelo esta
mirando**: cuando la nota decia "desaparecio el objeto W de 624 celdas", el modelo podia
comprobarlo contra su propia vista. v5 y v6 citan piezas del nivel anterior (mediana 10
celdas) que **ya no estan en pantalla**, y le piden buscar por hash objetos inexistentes.

QUE HACE v7
-----------
Deja de hablar del pasado y **resuelve la referencia contra el tablero de ahora**: segmenta
`current_frame`, busca en el los objetos de la jugada ganadora y cita SOLO los que existen,
diciendo donde estan. Todo lo que la nota afirma es verificable por el modelo en la vista
que ya tiene delante.

Si no se resuelve ninguno, la nota va VACIA. Es la misma regla de siempre: mejor nada que
mandar al modelo a buscar lo que no hay -- que es justamente el fallo que v7 corrige.

QUE NO CAMBIA: la extraccion. El modulo se GENERO a partir de `level_carry_pre.py`
sustituyendo unicamente la funcion que redacta; las 14 funciones de extraccion son byte a
byte las de v5 y hay tests que lo comprueban. Los hechos son los mismos: cambia contra que
se contrastan.
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


TOLERANCIA_TAMANO = 0.25     # +-25% de celdas para dar dos objetos por "el mismo tipo"


def _posicion(n) -> str:
    c = _caja(n)
    return f"fila {c[0]}, col {c[1]}" if c else "?"


def _resolver(nodo, seg_actual, celdas_actuales):
    """Busca `nodo` en el tablero de AHORA. Devuelve (nodo_actual, exacto) o None.

    Dos niveles, y ninguno mas: hash identico (el objeto literal, misma forma y color) o
    mismo color con tamano parecido (la pieza analoga, que es lo que la composicion de
    niveles suele traer). Sin tercer nivel por color suelto: seria adivinar.
    """
    if nodo is None or seg_actual is None:
        return None
    vivos = [n for n in seg_actual["nodes"] if not _es_fondo(n, celdas_actuales)]
    for n in vivos:
        if n["hash"] == nodo["hash"]:
            return (n, True)
    objetivo = nodo.get("pixels", 0)
    if objetivo:
        cand = [n for n in vivos
                if n["color"] == nodo["color"]
                and abs(n["pixels"] - objetivo) <= TOLERANCIA_TAMANO * objetivo]
        if cand:
            return (min(cand, key=lambda n: abs(n["pixels"] - objetivo)), False)
    return None


def render_nota_actual(nivel_actual, marcas, grid_actual) -> str:
    """La jugada ganadora, pero contada sobre los objetos que el modelo tiene DELANTE."""
    try:
        nivel_actual = int(nivel_actual)
    except (TypeError, ValueError):
        return ""
    if nivel_actual <= 1 or not marcas or not grid_actual:
        return ""
    seg = _segmentar(grid_actual)
    if seg is None:
        return ""
    celdas = sum(len(f) for f in grid_actual) or 1

    ult = marcas[-1]
    pos = ult.get("posicion") or {}
    ctrl, clic, cont = pos.get("controlado"), pos.get("clic"), pos.get("contactos") or []

    filas = []
    movil = _resolver(ctrl[0] if ctrl else None, seg, celdas)
    if movil:
        n, exacto = movil
        filas.append((f"el que movias para ganar", n, exacto))
    objetivo = _resolver(clic, seg, celdas)
    if objetivo and not (movil and objetivo[0]["id"] == movil[0]["id"]):
        filas.append((f"sobre el que apuntaste al ganar", objetivo[0], objetivo[1]))
    if len(filas) < 2 and cont:
        tocado = _resolver(cont[0], seg, celdas)
        if tocado and not any(f[1]["id"] == tocado[0]["id"] for f in filas):
            filas.append((f"con el que estaba en contacto", tocado[0], tocado[1]))
    if not filas:
        return ""      # nada de la jugada anterior esta en pantalla: no se manda a buscar

    lineas = [f"EN ESTE TABLERO estan los objetos con los que ganaste el nivel "
              f"{ult['nivel_ganado']} (misma segmentacion que ves):"]
    for etiqueta, n, exacto in filas[:3]:
        marca = "el mismo" if exacto else "uno igual"
        lineas.append(f"- {etiqueta}: {marca}, objeto {_etiqueta(n)} "
                      f"(hash {n['hash'][:HASH_CORTO]}), ahora en {_posicion(n)}")
    if len(filas) >= 2:
        lineas.append("Llevalos a la misma relacion que tenian al ganar: eso subio el nivel.")
    else:
        lineas.append("Ese fue el protagonista de la jugada que subio el nivel anterior.")
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

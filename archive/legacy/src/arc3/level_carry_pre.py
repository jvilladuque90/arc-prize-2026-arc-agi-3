"""Consolidacion v5: la POSICION GANADORA (precondicion), no el efecto de la jugada.

POR QUE (DESIGN 8.62)
---------------------
v4 abrio el canal: escribir la nota en objetos subio la mencion en el razonamiento del
11-15% (v1/v2/v3) al **30,9%**, con vara de ruido de un punto y pareado 13-3 de 16
juegos (p=0,0213). Pero al auditar lo que la nota decia en produccion aparecio un fallo
que venia **desde v1**:

    desaparecio 208 | se movio 112 | mediana 44 celdas, max 650
    39% de los objetos citados tenian >=100 celdas (el tablero es 64x64 = 4096)

Ejemplos reales: "desaparecio el objeto W de 624 celdas" (vc33). Eso no es una pieza.

La causa es estructural: la transicion se marca cuando el fotograma POSTERIOR a la
accion ya tiene el nivel incrementado, asi que el "despues" es **el primer tablero del
nivel N+1**. La diferencia descrita no es el efecto de la jugada: es el **redibujado del
cambio de nivel**. Por eso domina "desaparecio".

**El efecto causal de la jugada ganadora es INOBSERVABLE**: el entorno no entrega un
fotograma intermedio entre aplicar la accion y estar en el nivel siguiente. Inventarlo es
lo que llevabamos haciendo cinco brazos.

QUE HACE v5
-----------
Describe lo que SI se observa y transfiere: la **configuracion en el momento de ganar**,
leida SOLO en fotogramas del nivel N (nunca se cruza la frontera de nivel):

- *que objeto controlas*: el que se movio entre los dos ultimos tableros DENTRO del nivel.
- *con que estaba en contacto* en el tablero justo anterior a la jugada ganadora.
- *sobre que objeto apuntaba la jugada*, si fue un clic.

Sirve para los dos tipos de juego que tenemos: los de clic (8 de 19 usan una sola accion
casi siempre; ahi lo que informa es el objetivo del clic) y los direccionales (ahi informa
que pieza responde a las teclas).

Los `hash` salen de `segment_layer` DEL PROPIO HARNESS, para que sean los mismos que el
modelo ve en `current_frame.segmentation`. Si esa importacion falla NO se inventa una
segmentacion propia: se devuelve nota vacia.
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


def render_nota_previa(nivel_actual, marcas) -> str:
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
        return ""      # sin nada observable, mejor ninguna nota que una inventada

    lineas = [f"POSICION GANADORA del nivel {ult['nivel_ganado']} (tablero justo ANTES de "
              f"la jugada ganadora, misma segmentacion que ves):"]
    if ctrl is not None:
        n, (dr, dc) = ctrl
        partes = []
        if dr:
            partes.append(f"{abs(dr)} filas")
        if dc:
            partes.append(f"{abs(dc)} cols")
        paso = " y ".join(partes) or "de sitio"
        lineas.append(f"- respondia a tus jugadas: objeto {_etiqueta(n)} "
                      f"(hash {n['hash'][:HASH_CORTO]}), {paso} por jugada")
    if clic is not None and (ctrl is None or clic["hash"] != ctrl[0]["hash"]):
        lineas.append(f"- apuntaste sobre: objeto {_etiqueta(clic)} "
                      f"(hash {clic['hash'][:HASH_CORTO]})")
    if cont:
        lineas.append("- en contacto con: " + ", ".join(_etiqueta(n) for n in cont))
    if len(marcas) > 1:
        lineas.append(f"- (antes ganaste {len(marcas) - 1} nivel(es) mas)")
    lineas.append("Reconstruye esa configuracion: busca los objetos por color+celdas o "
                  "prefijo de hash. Lo de justo despues no se puede observar.")
    return "\n".join(lineas)


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

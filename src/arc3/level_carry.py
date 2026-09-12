"""Consolidacion al ganar un nivel: la mecanica ganadora del nivel N, arrastrada al N+1.

POR QUE
-------
Con v24 (3.55) cerramos el nivel 1 casi en todas partes y el muro es el nivel 2. El
informe oficial de ARC-AGI-3 dice que la dificultad es *por composicion*: los niveles
tardios integran los conceptos de los tempranos. Un humano no llega al nivel 2 con
un transcripto de clicks; llega con una regla ("clicar el cuadrado amarillo abre la
puerta"). Es lo que hace el hipocampo al consolidar: reproduce la trayectoria
exitosa y la comprime en un mecanismo.

Nuestro agente, en cambio, llega al nivel 2 con el historial crudo y re-deriva. Este
modulo calcula en el ANFITRION, a partir del historial que el harness ya persiste,
que accion subio cada nivel y que cambio en el tablero, y lo inyecta como texto.

QUE NO REPITE
-------------
- No exige escritura al modelo (las ranuras obligatorias costaron acciones y no
  pagaron): solo tokens de ENTRADA, ~100 por turno y solo a partir del nivel 2.
- No razona sobre acciones "inertes" por el fotograma final (el error de nav): un
  nivel completado es un evento inequivoco, no una inferencia. Si el tablero no
  muestra cambio en la transicion ganadora, la nota lo dice y no inventa.
- No depende de la conciencia de animacion: se monta sobre el harness original.
"""

from __future__ import annotations

# Orden 0..15 de la leyenda del harness: W=white, w=light gray, g=gray, G=dark gray,
# c=charcoal, B=black, M=magenta, P=pink, R=red, b=blue, S=sky blue, Y=yellow,
# O=orange, r=dark red, N=light green, p=purple.
LETRAS = "WwgGcBMPRbSYOrNp"
MAX_NIVELES_EN_NOTA = 4
MAX_TRANSICIONES_CENSO = 4


def _letra(v) -> str:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return "?"
    return LETRAS[i] if 0 <= i < len(LETRAS) else "?"


def _nivel(frame) -> int:
    try:
        return int(getattr(frame, "level", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _grid(frame):
    g = getattr(frame, "grid", None)
    return g if g else ()


def _celda(accion: str):
    txt = accion or ""
    if "row=" not in txt or "col=" not in txt:
        return None
    try:
        f = int(txt.split("row=", 1)[1].split(",", 1)[0].strip(") "))
        c = int(txt.split("col=", 1)[1].split(",", 1)[0].strip(") "))
        return f, c
    except (ValueError, IndexError):
        return None


def describir_cambio(antes, despues) -> dict:
    """Que cambio entre dos tableros: celdas, caja y censo de transiciones de color."""
    cambios = []
    censo: dict[str, int] = {}
    filas = min(len(antes), len(despues))
    for r in range(filas):
        fa, fd = antes[r], despues[r]
        for c in range(min(len(fa), len(fd))):
            if fa[c] != fd[c]:
                cambios.append((r, c))
                k = f"{_letra(fa[c])}>{_letra(fd[c])}"
                censo[k] = censo.get(k, 0) + 1
    if not cambios:
        return {"n": 0, "caja": None, "censo": {}}
    rs = [r for r, _ in cambios]
    cs = [c for _, c in cambios]
    return {"n": len(cambios), "caja": (min(rs), min(cs), max(rs), max(cs)), "censo": censo}


def transiciones_ganadoras(history_entries) -> list[dict]:
    """Una entrada por nivel completado: la accion que lo subio y que cambio.

    El historial son pares (accion, fotograma DESPUES de la accion). El nivel sube
    en la entrada i cuando su fotograma esta en un nivel mayor que el de la entrada
    i-1 (o que 1, para la primera). Esa accion gano el nivel anterior.
    """
    out: list[dict] = []
    if not history_entries:
        return out
    nivel_prev = 1
    inicio_nivel = 0          # indice de la primera accion del nivel en curso
    frame_prev = None
    for i, e in enumerate(history_entries):
        fr = getattr(e, "frame", None)
        if fr is None:
            continue
        niv = _nivel(fr)
        if niv > nivel_prev:
            antes = _grid(frame_prev) if frame_prev is not None else ()
            out.append({
                "nivel_ganado": nivel_prev,
                "accion": " ".join(str(getattr(e, "action", "") or "").split()),
                "acciones_en_nivel": i - inicio_nivel + 1,
                "cambio": describir_cambio(antes, _grid(fr)),
                "color_bajo_click": _color_bajo_click(getattr(e, "action", ""), antes),
            })
            inicio_nivel = i + 1
            nivel_prev = niv
        frame_prev = fr
    return out


def _color_bajo_click(accion: str, antes):
    c = _celda(accion or "")
    if c is None or not antes:
        return None
    r, col = c
    try:
        return _letra(antes[r][col])
    except (IndexError, TypeError):
        return None


def render_carry_note(nivel_actual: int, transiciones: list[dict]) -> str:
    """Nota para el prompt. Vacia en el nivel 1 o sin niveles ganados."""
    try:
        nivel_actual = int(nivel_actual)
    except (TypeError, ValueError):
        return ""
    if nivel_actual <= 1 or not transiciones:
        return ""
    lineas = [f"MECANICAS GANADORAS (calculadas por el anfitrion; vas por el nivel "
              f"{nivel_actual}, llevas {len(transiciones)} ganados):"]
    for t in transiciones[-MAX_NIVELES_EN_NOTA:]:
        partes = [f"- nivel {t['nivel_ganado']}: subio con {t['accion'] or '?'} "
                  f"tras {t['acciones_en_nivel']} acciones en ese nivel"]
        if t.get("color_bajo_click"):
            partes.append(f"(click sobre color {t['color_bajo_click']})")
        cam = t.get("cambio") or {}
        if cam.get("n"):
            caja = cam["caja"]
            censo = sorted(cam["censo"].items(), key=lambda kv: -kv[1])[:MAX_TRANSICIONES_CENSO]
            partes.append(f"; cambiaron {cam['n']} celdas en filas {caja[0]}-{caja[2]}, "
                          f"cols {caja[1]}-{caja[3]}: " +
                          ", ".join(f"{k} x{v}" for k, v in censo))
        else:
            partes.append("; el tablero final no muestra el cambio (pudo ser un evento "
                          "de nivel), no lo interpretes como accion inerte")
        lineas.append(" ".join(partes))
    lineas.append("Los niveles COMPONEN: la mecanica que gano el anterior suele seguir "
                  "valiendo aqui con otra disposicion. Pruebala primero, en pocas acciones.")
    return "\n".join(lineas)

"""Helpers de NAVEGACION para inyectar en el sandbox del agente (palanca de amplificacion).

POR QUE EXISTE. Medido en el rerun oculto: cada juego dispone de ~52.000 tokens y
ejecuta ~94 acciones (556 tokens por accion), mientras el primer nivel cuesta entre
7 y 55 acciones jugando perfecto. El agente delibera una accion a la vez. Pero el
entorno YA acepta `action([...])` con listas y bucles: un solo turno de pensamiento
podria ejecutar veinte acciones. Estos helpers convierten "que tecla me mueve y como
llego alli" -- algo que el modelo redescubre a mano en cada juego, gastando turnos --
en una llamada que devuelve una SECUENCIA lista para ejecutar.

Es la navegacion guiada de nuestra Fase 3 (modelo de movimiento aprendido + busqueda),
portada al dialecto del sandbox y montada sobre el harness fuerte.

DIALECTO OBLIGATORIO (verificado contra python_tool_sandbox.py):
  - sin imports, sin anotaciones de tipo, sin constantes de modulo
  - solo nombres de SAFE_BUILTINS: abs all any dict enumerate float getattr hasattr
    int isinstance len list max min range round set sorted str sum tuple zip print
    Exception ValueError TypeError RuntimeError
  - NO existen KeyError ni IndexError -> usar `except Exception` y comprobaciones
    explicitas de pertenencia
  - funciones puras: sin E/S, sin llamadas al modelo

El prelude se ensambla con inspect.getsource (misma tecnica que schema_helpers), asi
que estas mismas funciones son las que se testean y las que se inyectan.
"""

from __future__ import annotations

import inspect
import textwrap

# --- funciones inyectadas (dialecto sandbox de aqui en adelante) --------------


def _nav_as_grid(x):
    """Convierte frame o lista de listas en grid; None si no se puede."""
    if x is None:
        return None
    grid = None
    for name in ("_grid", "grid"):
        value = getattr(x, name, None)
        if isinstance(value, (list, tuple)):
            grid = value
            break
    if grid is None and isinstance(x, (list, tuple)):
        grid = x
    if grid is None:
        return None
    rows = []
    for row in grid:
        if not isinstance(row, (list, tuple)):
            return None
        rows.append(list(row))
    return rows


def _nav_shift(before, after):
    """Desplazamiento (drow, dcol) del objeto que se movio entre dos grids.

    Metodo: por cada color, centroide de las celdas que lo GANARON menos centroide
    de las que lo PERDIERON. Se toma el color con mas celdas emparejadas. Devuelve
    None si nada se movio o si el cambio no es una traslacion limpia.
    """
    ga = _nav_as_grid(before)
    gb = _nav_as_grid(after)
    if ga is None or gb is None:
        return None
    # El FONDO se mueve al reves que el objeto (donde el objeto llega, el fondo se
    # pierde). Sin excluirlo, el signo del desplazamiento sale invertido cuando el
    # fondo gana el desempate -- bug real cazado por scripts/test_sandbox_nav.py.
    freq = {}
    for row in ga:
        for v in row:
            if v in freq:
                freq[v] = freq[v] + 1
            else:
                freq[v] = 1
    background = None
    background_n = 0
    for v in freq:
        if freq[v] > background_n:
            background_n = freq[v]
            background = v
    gained = {}
    lost = {}
    rows = min(len(ga), len(gb))
    for r in range(rows):
        ra = ga[r]
        rb = gb[r]
        cols = min(len(ra), len(rb))
        for c in range(cols):
            va = ra[c]
            vb = rb[c]
            if va == vb:
                continue
            if vb in gained:
                gained[vb].append((r, c))
            else:
                gained[vb] = [(r, c)]
            if va in lost:
                lost[va].append((r, c))
            else:
                lost[va] = [(r, c)]
    best = None
    best_n = 0
    for color in gained:
        if color == background:
            continue
        if color not in lost:
            continue
        n = min(len(gained[color]), len(lost[color]))
        if n > best_n:
            best_n = n
            best = color
    if best is None or best_n == 0:
        return None
    g = gained[best]
    l = lost[best]
    gr = sum(p[0] for p in g) / len(g)
    gc = sum(p[1] for p in g) / len(g)
    lr = sum(p[0] for p in l) / len(l)
    lc = sum(p[1] for p in l) / len(l)
    dr = int(round(gr - lr))
    dc = int(round(gc - lc))
    if dr == 0 and dc == 0:
        return None
    return (dr, dc)


def _nav_transitions():
    """Alcanza el global `transitions` del sandbox (su nombre queda sombreado
    dentro de las funciones que lo reciben por parametro)."""
    return transitions  # noqa: F821 -- lo define el bootstrap del sandbox


def _nav_action_name(t):
    a = getattr(t, "action", None)
    if a is None:
        return None
    name = getattr(a, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(a)


def motion_model():
    """Modelo de movimiento MEDIDO de tus propias transiciones.

    Devuelve {nombre_de_accion: [drow, dcol]} solo para las acciones que produjeron
    un desplazamiento CONSISTENTE (la mayoria de sus observaciones coinciden). Las
    acciones que no mueven nada, o que mueven de forma erratica, quedan fuera.
    """
    counts = {}
    try:
        items = _nav_transitions()
    except Exception:
        return {}
    for t in items:
        name = _nav_action_name(t)
        if not name:
            continue
        before = getattr(t, "before_frame", None)
        after = getattr(t, "after_frame", None)
        shift = _nav_shift(before, after)
        if shift is None:
            continue
        key = str(shift[0]) + "," + str(shift[1])
        if name not in counts:
            counts[name] = {}
        bucket = counts[name]
        if key in bucket:
            bucket[key] = bucket[key] + 1
        else:
            bucket[key] = 1
    model = {}
    for name in counts:
        bucket = counts[name]
        total = sum(bucket[k] for k in bucket)
        best_key = None
        best_n = 0
        for k in bucket:
            if bucket[k] > best_n:
                best_n = bucket[k]
                best_key = k
        if best_key is None or total == 0:
            continue
        if best_n * 2 < total:  # sin mayoria => movimiento erratico, se descarta
            continue
        parts = best_key.split(",")
        model[name] = [int(parts[0]), int(parts[1])]
    return model


def plan_moves(drow, dcol, limit=40):
    """Secuencia de acciones que logra el desplazamiento (drow, dcol) pedido.

    Usa el modelo de movimiento medido y avanza con el paso que mas reduce la
    distancia restante. Devuelve [] si no hay modelo util o si no se puede acercar.
    Pasala directo a action(...) para ejecutar MUCHOS pasos en UN turno.
    """
    model = motion_model()
    if not model:
        return []
    moves = []
    rr = int(drow)
    cc = int(dcol)
    steps = 0
    while steps < int(limit) and (rr != 0 or cc != 0):
        best_name = None
        best_gain = 0
        best_rr = rr
        best_cc = cc
        for name in model:
            v = model[name]
            nr = rr - v[0]
            nc = cc - v[1]
            gain = (abs(rr) + abs(cc)) - (abs(nr) + abs(nc))
            if gain > best_gain:
                best_gain = gain
                best_name = name
                best_rr = nr
                best_cc = nc
        if best_name is None:
            break
        moves.append(best_name)
        rr = best_rr
        cc = best_cc
        steps = steps + 1
    return moves


def player_pos():
    """Posicion [row, col] del objeto que se movio en la ultima transicion util,
    o None si aun no se ha observado ningun movimiento."""
    try:
        items = _nav_transitions()
    except Exception:
        return None
    idx = len(items) - 1
    while idx >= 0:
        t = items[idx]
        before = getattr(t, "before_frame", None)
        after = getattr(t, "after_frame", None)
        ga = _nav_as_grid(before)
        gb = _nav_as_grid(after)
        shift = _nav_shift(before, after)
        if shift is not None and ga is not None and gb is not None:
            cells = []
            rows = min(len(ga), len(gb))
            for r in range(rows):
                cols = min(len(ga[r]), len(gb[r]))
                for c in range(cols):
                    if ga[r][c] != gb[r][c] and gb[r][c] != ga[r][c]:
                        cells.append((r, c))
            if cells:
                colors = {}
                for cell in cells:
                    v = gb[cell[0]][cell[1]]
                    if v in colors:
                        colors[v].append(cell)
                    else:
                        colors[v] = [cell]
                best = None
                best_n = 0
                for v in colors:
                    if len(colors[v]) > best_n:
                        best_n = len(colors[v])
                        best = v
                if best is not None:
                    pts = colors[best]
                    r = int(round(sum(p[0] for p in pts) / len(pts)))
                    c = int(round(sum(p[1] for p in pts) / len(pts)))
                    return [r, c]
        idx = idx - 1
    return None


# --- ensamblado del prelude (fuera del dialecto sandbox) ---------------------

_PRELUDE_FUNCTIONS = (
    _nav_as_grid,
    _nav_shift,
    _nav_transitions,
    _nav_action_name,
    motion_model,
    plan_moves,
    player_pos,
)

NAV_PROMPT_NOTE = (
    "NAVIGATION HELPERS also preloaded: motion_model() returns the MEASURED "
    "{action: [drow, dcol]} learned from your own transitions; plan_moves(drow, dcol) "
    "returns an ordered action list achieving that displacement; player_pos() returns "
    "the [row, col] of the object that moves. Prefer action(plan_moves(dr, dc)) to "
    "execute a whole route in ONE turn instead of one action per turn."
)


def build_nav_prelude():
    """Fuente inyectable, verificada compilable (misma tecnica que schema_helpers)."""
    parts = [textwrap.dedent(inspect.getsource(fn)) for fn in _PRELUDE_FUNCTIONS]
    source = "\n".join(parts)
    compile(source, "<nav_prelude>", "exec")
    return source

"""S1 (plan.md): predictor de efectos como DATOS en el sandbox del agente, no como nota.

POR QUE ASI
-----------
La leccion de sheetu12b coincide con la nuestra: "frames yes, narration no". F13 (fotogramas
consultables desde el sandbox) movio el oculto 3,20 -> 3,71; el mismo contenido narrado por el
anfitrion bajo a 2,57. Nuestras notas de texto (mapa cognitivo 8.84, siete brazos de memoria)
tampoco pagaron. Asi que el predictor NO escribe nada en el prompt de cada turno: es una funcion
mas del sandbox, documentada en una sola linea junto a `segmentation`, que el modelo consulta si
quiere.

QUE HACE
--------
`action_effects()` recorre `transitions` del NIVEL ACTUAL y, para cada accion, cuenta cuantas
veces cambio el tablero. Para los clics agrupa por el COLOR de la celda clicada, que es lo que
generaliza a celdas nunca clicadas: si clicar fondo nunca hizo nada, un clic nuevo sobre fondo
tampoco. "Cambio" ignora el HUD: una fila o columna de BORDE que cambia en la mayoria de
transiciones del nivel (la barra de presupuesto que casi todos los juegos dibujan; crece, asi
que cada celda suya cambia una sola vez y hay que detectarla por linea, no por celda).

`action_effects(accion)` devuelve solo la probabilidad suavizada (Laplace) de que esa accion
cambie el tablero.

COMO SE MONTA
-------------
Dos ediciones sobre la plantilla del sandbox (`python_tool_sandbox._SANDBOX_BOOTSTRAP`), con
anclas que F13 de agentfix no toca, y una linea en `STRUCTURED_RUNTIME_STATE_ADDENDUM`.
Cada ancla debe aparecer exactamente una vez; si no, no se toca nada.
"""

from __future__ import annotations

# --- codigo que vive DENTRO del sandbox (nivel de modulo del bootstrap) --------------------
HELPER_SRC = r'''
def _fx_rows(frame):
    try:
        g = getattr(frame, "_grid", None)
        if g:
            return [list(r) for r in g]
        return [list(line) for line in str(getattr(frame, "ascii", "")).split("\n") if line]
    except Exception:
        return []


def _fx_ascii_rows(frame):
    try:
        return [line for line in str(getattr(frame, "ascii", "")).split("\n") if line]
    except Exception:
        return []


_FX_ENGINE = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT",
              "ACTION5": "SPACE", "ACTION6": "MOUSE", "ACTION7": "UNDO"}


def _fx_parse(action):
    """'MOUSE(row=4, col=7)' | {'action': 'MOUSE', 'row': 4, 'col': 7} | 'UP' -> (name, row, col)."""
    if isinstance(action, dict):
        name = str(action.get("action", "")).strip().upper()
        name = _FX_ENGINE.get(name, name)
        try:
            return name, int(action.get("row")), int(action.get("col"))
        except (TypeError, ValueError):
            return name, None, None
    text = str(action or "").strip()
    name = text.split("(", 1)[0].strip().upper()
    name = _FX_ENGINE.get(name, name)
    row = col = None
    if "row=" in text and "col=" in text:
        try:
            row = int(text.split("row=", 1)[1].split(",", 1)[0].strip(") "))
            col = int(text.split("col=", 1)[1].split(",", 1)[0].strip(") "))
        except (ValueError, IndexError):
            row = col = None
    return name, row, col


def _fx_diff(a, b):
    cells = set()
    if not a or not b or len(a) != len(b):
        return None
    for r, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            return None
        for c, (x, y) in enumerate(zip(ra, rb)):
            if x != y:
                cells.add((r, c))
    return cells


def _fx_rate(tried, changed):
    return round((changed + 1.0) / (tried + 2.0), 3)


def _fx_compute(transitions, valid_actions, current_frame):
    level = getattr(current_frame, "level", None)
    rows = []
    for t in transitions or []:
        before, after = getattr(t, "before_frame", None), getattr(t, "after_frame", None)
        if before is None or after is None or getattr(before, "level", None) != level:
            continue
        name, r, c = _fx_parse(getattr(t, "action", ""))
        if not name:
            continue
        won = getattr(after, "level", level) != level
        diff = _fx_diff(_fx_rows(before), _fx_rows(after))
        color = None
        if r is not None and c is not None:
            asc = _fx_ascii_rows(before)
            if 0 <= r < len(asc) and 0 <= c < len(asc[r]):
                color = asc[r][c]
        rows.append((name, color, diff, won))

    # HUD: una LINEA de borde (fila o columna extrema) que cambia en la mayoria de transiciones
    # del nivel. Por lineas y no por celdas: la barra de presupuesto crece, y cada celda suya
    # cambia una sola vez.
    cur = _fx_rows(current_frame)
    alto, ancho = len(cur), (len(cur[0]) if cur else 0)

    def _lineas(cell):
        r, c = cell
        out = []
        if r == 0:
            out.append(("r", 0))
        if r == alto - 1:
            out.append(("r", alto - 1))
        if c == 0:
            out.append(("c", 0))
        if c == ancho - 1:
            out.append(("c", ancho - 1))
        return out

    hud = set()
    diffs = [d for (_n, _c, d, w) in rows if d is not None and not w]
    if len(diffs) >= 4 and alto and ancho:
        counts = {}
        for d in diffs:
            for linea in {l for cell in d for l in _lineas(cell)}:
                counts[linea] = counts.get(linea, 0) + 1
        hud = {l for l, k in counts.items() if k * 2 > len(diffs)}

    def _real(diff):
        return any(not (set(_lineas(cell)) & hud) for cell in diff)

    actions, clicks = {}, {}
    for name, color, diff, won in rows:
        changed = bool(won) or (diff is None) or _real(diff)
        slot = actions.setdefault(name, [0, 0])
        slot[0] += 1
        slot[1] += int(changed)
        if name == "MOUSE" and color is not None:
            cs = clicks.setdefault(color, [0, 0])
            cs[0] += 1
            cs[1] += int(changed)

    names = []
    for a in valid_actions or []:
        n = _FX_ENGINE.get(str(a).strip().upper(), str(a).strip().upper())
        if n and n not in names:
            names.append(n)
    present = []
    for line in _fx_ascii_rows(current_frame):
        for ch in line:
            if ch not in present:
                present.append(ch)
    return {
        "level": level,
        "transitions_this_level": len(rows),
        "actions": {n: {"tried": t, "changed": k, "p_change": _fx_rate(t, k)}
                    for n, (t, k) in actions.items()},
        "untested": [n for n in names if n not in actions and n not in ("MOUSE", "RESET")],
        "clicks_by_color": {col: {"tried": t, "changed": k, "p_change": _fx_rate(t, k)}
                            for col, (t, k) in clicks.items()},
        "untested_click_colors": ([ch for ch in present if ch not in clicks]
                                  if "MOUSE" in names else []),
        "hud_edge_lines_ignored": sorted("%s%d" % l for l in hud),
    }


def _fx_action_effects(transitions, valid_actions, current_frame, action=None):
    info = _fx_compute(transitions, valid_actions, current_frame)
    if action is None:
        return info
    name, r, c = _fx_parse(action)
    if name == "MOUSE" and r is not None and c is not None:
        asc = _fx_ascii_rows(current_frame)
        if 0 <= r < len(asc) and 0 <= c < len(asc[r]):
            slot = info["clicks_by_color"].get(asc[r][c])
            return slot["p_change"] if slot else _fx_rate(0, 0)
    slot = info["actions"].get(name)
    return slot["p_change"] if slot else _fx_rate(0, 0)


'''

REGISTER_SRC = '''
        runtime_globals["action_effects"] = (
            lambda action=None: _fx_action_effects(
                runtime_globals.get("transitions") or [],
                runtime_globals.get("valid_actions") or [],
                runtime_globals.get("current_frame"),
                action,
            )
        )'''

# Anclas sobre la plantilla YA desindentada (textwrap.dedent se aplica al importar).
ANCLA_HELPER = "def _json_safe(value):"
ANCLA_REGISTRO = '        runtime_globals["last_action_result"] = action_result'

DOC_LINE = (
    "- `action_effects()` returns what each action has actually done on the CURRENT level, "
    "computed from `transitions` (an edge row/col that changes on most actions is treated as HUD and ignored): "
    "`{'level', 'actions': {name: {'tried', 'changed', 'p_change'}}, 'untested': [...], "
    "'clicks_by_color': {color: {'tried', 'changed', 'p_change'}}, 'untested_click_colors': [...]}`. "
    "`action_effects(action)` returns only `p_change` for one action; a MOUSE click is judged by the "
    "color under it, e.g. `action_effects({'action': 'MOUSE', 'row': 4, 'col': 7})`.\n"
)


def patch_sandbox_source(source: str) -> str:
    """Devuelve la plantilla con el helper y su registro. Lanza si un ancla no es unica."""
    text = str(source)
    if "_fx_action_effects" in text:
        return text
    for ancla in (ANCLA_HELPER, ANCLA_REGISTRO):
        n = text.count(ancla)
        if n != 1:
            raise RuntimeError(f"ancla encontrada {n} veces, se esperaba 1: {ancla.strip()!r}")
    # el bootstrap es un textwrap.dedent de un bloque indentado 4 espacios
    text = text.replace(ANCLA_HELPER, HELPER_SRC.lstrip("\n") + "\n" + ANCLA_HELPER, 1)
    text = text.replace(ANCLA_REGISTRO, ANCLA_REGISTRO + REGISTER_SRC, 1)
    return text


def patch_addendum(addendum: str) -> str:
    """Anade la linea de documentacion tras la de `valid_actions`, o al final."""
    text = str(addendum)
    if "action_effects()" in text:
        return text
    ancla = "- `valid_actions` is the current list of valid action names.\n"
    if ancla in text:
        return text.replace(ancla, ancla + DOC_LINE, 1)
    return text + DOC_LINE

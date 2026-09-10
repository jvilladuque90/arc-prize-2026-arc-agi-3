"""Mapa cognitivo: grafo de estados calculado por el ANFITRION e inyectado en el prompt.

POR QUE
-------
El informe oficial de ARC-AGI-3 mide cuatro capacidades: exploracion, modelado,
fijacion de metas y planificacion. Los dos primeros puestos del *preview* no eran
modelos de lenguaje: el 2o (Blind Squirrel, 6.71%) construia un **grafo dirigido de
estados** a partir de los fotogramas y podaba las acciones que hacian bucle o no
cambiaban nada. Los agentes de lenguaje con vision se quedaron en 3.70-4.37% porque
"carecen del seguimiento de estado fotograma a fotograma que estos entornos
requieren".

Es tambien lo que hace un cerebro: la teoria de segmentacion de eventos dice que no
se codifica cada fotograma sino la TRANSICION en el limite de evento, y los mapas
cognitivos almacenan "que lleva a donde". Nuestro agente, en cambio, recibe un
transcripto y re-deriva el estado cada turno con 27 mil millones de parametros.

EL ERROR QUE ESTE MODULO NO REPITE
----------------------------------
``src/arc3/sandbox_nav.py::_nav_shift`` comparaba UNICAMENTE el fotograma final. En
los juegos de tipo 1 (ft09, sb26) el primero y el ultimo son identicos aunque la
accion SI hizo algo, asi que aquel helper le ensenaba al modelo que las acciones
informativas estaban muertas.

Aqui las aristas NO se etiquetan comparando fotogramas. Se toman del guardia de
no-ops del harness anim, que recibe por accion la tupla
``(nivel, firma_antes, firma_accion, board_changed, animated)`` — con ``animated``
puesto cuando el entorno devolvio varios fotogramas. Una accion animada NUNCA se
reporta como inerte, aunque el tablero final sea identico. La correccion se hereda
del harness en vez de re-implementarse.

COSTE
-----
Solo tokens de ENTRADA (la nota se anade al prompt del usuario). CERO escritura
exigida al modelo: los dos parches anteriores que exigian que el modelo escribiera
(ranuras, modelo de mundo) costaron acciones y no pagaron.
"""

from __future__ import annotations

MAX_LISTA = 6          # cuantos elementos como mucho por linea de la nota
MAX_CELDAS_CLICK = 8   # celdas de click que se enumeran antes de resumir a caja


class MapRecorder:
    """Envoltorio del ``NoopGuard`` del harness que ademas registra el grafo.

    Se instala en lugar del guardia real: delega TODO en el (para no alterar su
    comportamiento ni una coma) y de paso apunta cada transicion observada.

    La arista i va del estado ``before_sig`` de la observacion i al estado
    ``before_sig`` de la observacion i+1 del mismo nivel: el harness reasigna
    ``noop_guard_board_sig`` con el tablero refrescado justo despues de observar,
    asi que las observaciones consecutivas encadenan el recorrido real.
    """

    def __init__(self, inner):
        self._inner = inner
        self.registros: list[dict] = []

    # -- delegacion total al guardia real ----------------------------------

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def is_known_noop(self, level, board_sig, action_sig):
        return self._inner.is_known_noop(level, board_sig, action_sig)

    def observe(self, *, level, board_before_sig, action_sig, board_changed,
                animated=False, **kw):
        try:
            self.registros.append({
                "level": _entero(level),
                "antes": str(board_before_sig),
                "accion": " ".join(str(action_sig or "").split()),
                # "hizo algo" = cambio el tablero O devolvio animacion. La segunda
                # mitad es la que evita el error de nav.
                "efecto": bool(board_changed) or bool(animated),
                "solo_animacion": bool(animated) and not bool(board_changed),
            })
        except Exception:  # noqa: BLE001 — registrar jamas puede tumbar el guardia
            pass
        return self._inner.observe(
            level=level, board_before_sig=board_before_sig, action_sig=action_sig,
            board_changed=board_changed, animated=animated, **kw)

    def reset(self) -> None:
        self.registros = []


def _entero(v, defecto=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return defecto


def _nombre(accion: str) -> str:
    """'MOUSE(row=20, col=13)' -> 'MOUSE';  'UP' -> 'UP'."""
    return (accion or "").split("(")[0].strip().upper()


def _celda(accion: str):
    """Extrae (fila, col) de una accion de raton, o None."""
    txt = accion or ""
    if "row=" not in txt or "col=" not in txt:
        return None
    try:
        fila = int(txt.split("row=", 1)[1].split(",", 1)[0].strip(") "))
        col = int(txt.split("col=", 1)[1].split(",", 1)[0].strip(") "))
        return fila, col
    except (ValueError, IndexError):
        return None


def construir_grafo(registros: list[dict], nivel: int) -> dict:
    """Grafo del nivel indicado: aristas, estados y afordancias."""
    delnivel = [r for r in registros if r.get("level") == nivel]
    aristas: list[tuple[str, str, str, bool]] = []   # (origen, accion, destino, efecto)
    for i, r in enumerate(delnivel):
        destino = delnivel[i + 1]["antes"] if i + 1 < len(delnivel) else None
        aristas.append((r["antes"], r["accion"], destino, r["efecto"]))

    estados: dict[str, int] = {}
    for r in delnivel:
        estados[r["antes"]] = estados.get(r["antes"], 0) + 1

    utiles: set[str] = set()
    inertes: dict[str, int] = {}
    celdas_utiles: list[tuple[int, int]] = []
    solo_anim = 0
    for r in delnivel:
        n = _nombre(r["accion"])
        if r["efecto"]:
            utiles.add(n)
            c = _celda(r["accion"])
            if c and c not in celdas_utiles:
                celdas_utiles.append(c)
        else:
            inertes[n] = inertes.get(n, 0) + 1
        if r.get("solo_animacion"):
            solo_anim += 1

    return {
        "aristas": aristas,
        "estados": estados,
        "utiles": utiles,
        # inerte de verdad = nunca tuvo efecto en NINGUN estado de este nivel
        "nunca_utiles": {n for n in inertes if n not in utiles},
        "celdas_utiles": celdas_utiles,
        "solo_animacion": solo_anim,
        "n_acciones": len(delnivel),
    }


def render_map_note(registros: list[dict], nivel: int, firma_actual: str | None,
                    acciones_validas: list[str] | None) -> str:
    """Nota compacta para el prompt. Devuelve '' si no hay nada que aportar."""
    if not registros:
        return ""
    g = construir_grafo(registros, nivel)
    if not g["aristas"]:
        return ""

    lineas: list[str] = []
    n_estados = len(g["estados"])
    visitas = g["estados"].get(firma_actual or "", 0)

    cab = (f"MAPA DEL ANFITRION (nivel {nivel}, {g['n_acciones']} acciones aqui, "
           f"{n_estados} estados distintos)")
    lineas.append(cab)

    # 1. Que se ha probado DESDE el estado actual, y con que resultado.
    if firma_actual:
        desde = [(a, dst, ef) for (org, a, dst, ef) in g["aristas"] if org == firma_actual]
        if desde:
            vistos: dict[str, bool] = {}
            for a, _dst, ef in desde:
                vistos[a] = vistos.get(a, False) or ef
            hechas = [f"{a}{'' if ef else ' (sin efecto)'}"
                      for a, ef in list(vistos.items())[:MAX_LISTA]]
            lineas.append(f"- desde este estado ya probaste: {', '.join(hechas)}")
        if visitas > 1:
            lineas.append(f"- ATENCION: estas en un estado ya visitado {visitas} veces "
                          f"(posible bucle)")

        # 2. La frontera: acciones validas no probadas AQUI y que ademas no estan
        #    ya demostradas inertes en todo el nivel. Sin este segundo filtro la
        #    nota se contradice sola (listaria una accion como "por probar" y dos
        #    lineas mas abajo como "nunca hizo nada").
        if acciones_validas:
            probadas = {_nombre(a) for (org, a, _d, _e) in g["aristas"] if org == firma_actual}
            frontera = [a for a in acciones_validas
                        if a.upper() not in probadas
                        and a.upper() not in g["nunca_utiles"]
                        and a.upper() != "MOUSE"]
            if frontera:
                lineas.append(f"- desde este estado NO has probado: "
                              f"{', '.join(frontera[:MAX_LISTA])}")

    # 3. Afordancias del nivel: que ha servido y que no, nunca.
    if g["nunca_utiles"]:
        lineas.append(f"- nunca han hecho nada en este nivel: "
                      f"{', '.join(sorted(g['nunca_utiles'])[:MAX_LISTA])}")
    if g["celdas_utiles"]:
        cs = g["celdas_utiles"]
        if len(cs) <= MAX_CELDAS_CLICK:
            texto = ", ".join(f"({f},{c})" for f, c in cs)
        else:
            fs = [f for f, _ in cs]
            cols = [c for _, c in cs]
            texto = (f"{len(cs)} celdas en filas {min(fs)}-{max(fs)}, "
                     f"cols {min(cols)}-{max(cols)}")
        lineas.append(f"- clicks que SI hicieron algo: {texto}")

    # 4. La correccion de animacion, dicha explicitamente.
    if g["solo_animacion"]:
        lineas.append(f"- {g['solo_animacion']} accion(es) tuvieron efecto SOLO en la "
                      f"animacion (tablero final identico): NO son inertes")

    if len(lineas) == 1:
        return ""
    return "\n".join(lineas)

"""H1 (plan.md paso 3): endurecimiento SIN riesgo de modelo sobre el harness parcheado de Franzen.

Dos defectos verificados por ejecucion en la auditoria (DESIGN 8.85, informes R03/R05):

R1. Un timeout o una caida del sandbox devuelve un resultado SIN la clave `keepable_functions`;
    `_record_retained_functions` entonces construye `kept = {}` y VACIA la biblioteca de funciones
    persistentes del juego entero (y le dice al modelo "no longer retained"). El tipo de fragmento que
    agota los 30 s (busqueda larga, bucle sobre muchos fotogramas) es justo el que el prompt promueve.
    Arreglo: si la clave no esta, la ejecucion no dijo nada sobre la biblioteca; se conserva tal cual.
    Un resultado exitoso SIEMPRE trae la clave (python_tool_sandbox.py: lista, vacia si hace falta), y
    el borrado deliberado ("all" rechazado) tambien la trae, asi que esos caminos no cambian.

R2. `_render_tool_payload` solo recorta valores `str`; un `result` list/dict (180 KB en una prueba)
    se vuelca entero al contexto. Arreglo: si el volcado JSON supera 16.000 caracteres (~4x el
    presupuesto de salida de 3.072 tokens) se convierte en texto y pasa por el recorte normal.

Nada de esto cambia lo que el modelo ve en una ejecucion normal.

Cada ancla debe aparecer EXACTAMENTE una vez; si no, no se toca nada. Independiente del fin de linea.
"""

from __future__ import annotations

A1 = (
    "        try:\n"
    "            previous = self._kept_functions\n"
    "            kept = {}\n"
)
B1 = (
    "        try:\n"
    "            if \"keepable_functions\" not in sandbox_result:\n"
    "                # H1/R1: timeout, caida o respuesta invalida del sandbox: no dice nada de la\n"
    "                # biblioteca, asi que se conserva (antes se vaciaba entera)\n"
    "                return\n"
    "            previous = self._kept_functions\n"
    "            kept = {}\n"
)
A2 = (
    "            value = result.get(field)\n"
    "            if isinstance(value, str):\n"
    "                value, suppressed_rows = _suppress_board_dumps(value)\n"
)
B2 = (
    "            value = result.get(field)\n"
    "            if value is not None and not isinstance(value, str):\n"
    "                # H1/R2: un result list/dict enorme se volcaba entero al contexto\n"
    "                try:\n"
    "                    dumped = json.dumps(value, ensure_ascii=False, default=str)\n"
    "                except Exception:\n"
    "                    dumped = str(value)\n"
    "                if len(dumped) > 16000:\n"
    "                    value = dumped\n"
    "                    result[field] = dumped\n"
    "            if isinstance(value, str):\n"
    "                value, suppressed_rows = _suppress_board_dumps(value)\n"
)

PAIRS = ((A1, B1), (A2, B2))
MARCA = "H1/R1"


def apply_to_source(src: str) -> str:
    """Devuelve el codigo de tool_agent.py endurecido. Lanza si algun ancla no es unica."""
    if MARCA in src:
        return src
    nl = "\r\n" if "\r\n" in src else "\n"
    out = src
    for a, b in PAIRS:
        a_n, b_n = a.replace("\n", nl), b.replace("\n", nl)
        n = out.count(a_n)
        if n != 1:
            raise RuntimeError(f"ancla encontrada {n} veces, se esperaba 1: {a.strip().splitlines()[0]!r}")
        out = out.replace(a_n, b_n, 1)
    return out

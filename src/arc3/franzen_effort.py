"""E1: esfuerzo de razonamiento FIJO desde el inicio sobre el harness parcheado de Franzen.

POR QUE (DESIGN 8.88): la plantilla de chat de Qwen3.8-Flash-Next usa `reasoning_effort` con valor por
defecto `xhigh`, que inyecta en el prompt del sistema "piensa con cuidado, valida supuestos clave,
considera alternativas, prioriza la correccion". `low` inyecta "piensa breve, ve directo a la
conclusion" y `medium` no inyecta ninguna instruccion. Franzen corre todo en xhigh (su escalera de
esfuerzo solo baja DESPUES de una truncacion y esta apagada). En los 15 juegos dificiles el modelo gasta
~2.200 tokens por accion (31 acciones en 37 min en cd82) y 3,5 % de las respuestas (>= 11K tokens)
consumen el 19 % de los tokens: el cuello de botella es la velocidad de comprension. Este brazo prueba
el MECANISMO de la opcion B (modelo que razona con menos tokens) con el modelo actual y sin ingenieria.

El valor llega al servidor como chat_template_kwargs["reasoning_effort"] por el mismo camino que ya usa
la escalera de Franzen. Si ARC3_STATIC_REASONING_EFFORT esta vacio no cambia nada.
Cada ancla debe aparecer EXACTAMENTE una vez. Independiente del fin de linea.
"""

from __future__ import annotations

A = (
    "        ladder = _reasoning_effort_ladder()\n"
    "        rung = getattr(self, \"_reasoning_effort_rung\", -1)\n"
    "        if ladder and 0 <= rung < len(ladder):\n"
    "            kwargs[\"reasoning_effort\"] = ladder[rung]\n"
    "        return kwargs\n"
)
B = (
    "        # E1: esfuerzo fijo desde el inicio (la escalera, si esta activa, lo sobrescribe despues)\n"
    "        static_effort = os.environ.get(\"ARC3_STATIC_REASONING_EFFORT\", \"\").strip()\n"
    "        if static_effort:\n"
    "            kwargs[\"reasoning_effort\"] = static_effort\n"
    "        ladder = _reasoning_effort_ladder()\n"
    "        rung = getattr(self, \"_reasoning_effort_rung\", -1)\n"
    "        if ladder and 0 <= rung < len(ladder):\n"
    "            kwargs[\"reasoning_effort\"] = ladder[rung]\n"
    "        return kwargs\n"
)
MARCA = "ARC3_STATIC_REASONING_EFFORT"
VALIDOS = ("xhigh", "medium", "low")


def apply_to_source(src: str) -> str:
    """Devuelve tool_agent.py con el esfuerzo fijo. Lanza si el ancla no es unica."""
    if MARCA in src:
        return src
    nl = "\r\n" if "\r\n" in src else "\n"
    a, b = A.replace("\n", nl), B.replace("\n", nl)
    n = src.count(a)
    if n != 1:
        raise RuntimeError(f"ancla encontrada {n} veces, se esperaba 1")
    return src.replace(a, b, 1)

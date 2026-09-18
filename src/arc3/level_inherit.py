"""Herencia entre niveles: no borrar lo que el modelo aprendio justo al ganar.

EL FALLO (medido sobre tres corridas de 60 min)
-----------------------------------------------
El harness mantiene siete campos de "modelo del mundo" que el modelo puede escribir con
prefijos (`World model:`, `Goal model:`, ..., `Cross-level notes:`). En cada **transicion
de nivel** borra seis de los siete y conserva **solo** `cross_level_notes`
(`tool_agent.py::_update_summarized_knowledge_from_step_summary`).

Y el modelo escribe justo al reves:

    campo                 lo escribe el modelo    sobrevive al cambio de nivel
    World model:          153-255 veces           NO (se borra)
    Cross-level notes:    0, 0 y 20 veces         SI (unico que persiste)

Es decir: **vuelca todo su conocimiento en el campo que se borra y no usa el unico que
persiste**. Cada vez que gana un nivel —el momento en que ese conocimiento vale mas— se
tira entero.

Y lo que se tira no es relleno. Textual de nuestras corridas:

    "A didn't complete; toggle works (clicked cells now orange). Switch to hypothesis B"
    "ACTION7 = no-op. New hypothesis: checkers-like. Clicking a token highlights a legal
     jump-landing square"
    "All interior clicks so far produce ZERO interior change; only the bottom HUD bar shrinks"

Mecanicas descubiertas, hipotesis refutadas y autocorrecciones: exactamente el contenido
del manual del agente que hizo el 100% del set publico (DESIGN 8.48 punto 3).

QUE HACE ESTE MODULO
--------------------
En la transicion de nivel, **antes** de que el harness borre, dobla los campos sustantivos
dentro de `cross_level_notes`, sellados por nivel. Nada mas.

POR QUE ENCAJA CON TODO LO MEDIDO
---------------------------------
- **Cero tokens de salida**: el modelo ya escribio ese texto; solo se impide que se pierda.
  El coste de salida es la objecion que freno esta familia (v13/v14), y aqui no existe.
- **En el vocabulario del modelo**: son sus propias palabras, no una descripcion nuestra.
  8.61-8.65 midieron que el modelo ignora lo que le escribimos nosotros.
- **Minimo, ganado y raro** (regla de 8.41): se dispara UNA vez por nivel completado, que
  es la unica senal inequivoca que da el entorno (8.53).
"""

from __future__ import annotations

CAMPOS = (("world_model", "mundo"), ("goal_model", "meta"), ("action_model", "acciones"))
MAX_NIVELES = 3          # cuantos sellos se arrastran; acota el crecimiento
MAX_PALABRAS_CAMPO = 60  # la mediana medida es 41; 60 deja margen sin inflar
SEP = " || "


def _recortar(texto: str, maximo: int = MAX_PALABRAS_CAMPO) -> str:
    palabras = (texto or "").split()
    if not palabras:
        return ""
    return " ".join(palabras[:maximo]) + ("..." if len(palabras) > maximo else "")


def sello_de_nivel(nivel_ganado: int, conocimiento: dict) -> str:
    """Un sello compacto con lo que el modelo sabia al ganar, EN SUS PALABRAS."""
    partes = []
    for clave, etiqueta in CAMPOS:
        v = _recortar((conocimiento or {}).get(clave, ""))
        if v:
            partes.append(f"{etiqueta}: {v}")
    if not partes:
        return ""
    return f"[al ganar el nivel {nivel_ganado}] " + "; ".join(partes)


def doblar(previo: str, nuevo_sello: str) -> str:
    """Acumula el sello nuevo conservando como mucho MAX_NIVELES, los mas recientes."""
    if not nuevo_sello:
        return previo or ""
    sellos = [s for s in (previo or "").split(SEP) if s.strip()]
    if nuevo_sello in sellos:
        return previo or ""
    sellos.append(nuevo_sello)
    return SEP.join(sellos[-MAX_NIVELES:])


def heredar(conocimiento: dict, nivel_ganado: int) -> str:
    """Devuelve el nuevo valor de `cross_level_notes`, o el previo si no hay nada que salvar."""
    if not isinstance(conocimiento, dict):
        return ""
    sello = sello_de_nivel(nivel_ganado, conocimiento)
    return doblar(conocimiento.get("cross_level_notes", ""), sello)


def es_transicion(resumen) -> bool:
    try:
        return bool((resumen or {}).get("level_transition"))
    except Exception:
        return False

"""Formatos de prompt del banco micro: UNICA fuente de verdad.

Lo importan tanto el runner local (CPU) como el de Colab (GPU). El de Colab lo
descarga del repo publico junto con el banco, asi que ambos brazos comparan
EXACTAMENTE los mismos textos: si los prompts divergieran, el A/B no mediria el
formato sino la divergencia.

Las variantes son pares pareados sobre los mismos items:

  A. effect_of_action  V0_crudo  vs  V1_objetos   -> ¿las features objetuales
     ayudan a INFERIR la mecanica? (la tesis de Fase 3, por fin comprobable)
  B. plan_action       V0_sin_tabla vs V2_con_tabla -> ¿inyectar los efectos
     medidos basta para PLANIFICAR? (esto decide la carga del seam C)
  C. which_action      lookup                     -> control de lectura: la
     respuesta esta literalmente en el enunciado. Si el modelo falla aqui, sus
     fallos en A y B son de formato/parseo, no de razonamiento.
"""

from __future__ import annotations

import re

LEGEND = ("Colores: W=blanco w=gris-claro g=gris G=gris-oscuro c=carbon B=negro M=magenta "
          "P=rosa R=rojo b=azul S=celeste Y=amarillo O=naranja r=rojo-oscuro N=verde p=morado")


def prompt_effect(item, with_objects: bool) -> str:
    p = ["Estas analizando un juego de rejilla. Cada ejemplo muestra el tablero ANTES y "
         "DESPUES de ejecutar la misma accion.", LEGEND, ""]
    for i, s in enumerate(item["shots"], 1):
        p += [f"Ejemplo {i} - ANTES:", s["before"]]
        # Las features van PEGADAS a su ejemplo y en coordenadas del recorte que se
        # esta mostrando. Una lista de objetos del tablero completo, en coordenadas
        # absolutas, describe una vista que el modelo no tiene delante: no es la
        # tesis de Fase 3, es ruido con formato de dato.
        if with_objects and s.get("objects"):
            objs = ", ".join(f"color {o['color']} tam {o['size']} centro {o['center']}"
                             for o in s["objects"])
            p += [f"  (objetos en este recorte, fila/columna dentro del recorte: {objs})"]
        p += [f"Ejemplo {i} - DESPUES:", s["after"], ""]
    # OJO CON EL FORMATO: la primera version ofrecia la plantilla literal
    # "move DR DC" y el modelo de 0.6B la copiaba tal cual en el 100% de los items
    # (0/54 en ambos brazos). Un hueco copiable se copia: aqui solo se muestran
    # respuestas ya concretas, para que no haya nada que parrotear.
    p += [f"Que hace la accion {item['action']}?",
          "Responde con UNA sola linea, sin explicar. Ejemplos de respuestas validas:",
          "  none        <- si el tablero queda igual",
          "  change      <- si cambia pero nada se traslada",
          "  move -1 0   <- si algo se traslada 1 fila hacia arriba",
          "  move 0 3    <- si algo se traslada 3 columnas hacia la derecha",
          "  move 2 -5   <- si algo baja 2 filas y va 5 columnas a la izquierda",
          "Respuesta:"]
    return "\n".join(p)


def prompt_which(item, _unused: bool = False) -> str:
    obs = "\n".join(f"  {s['action']} desplaza {s['moved']}"
                    for s in item["shots"] if s.get("moved"))
    return "\n".join(["Observaciones medidas en este juego (fila, columna):", obs, "",
                      f"Que accion mueve el objeto hacia {item['direction']}?",
                      "Responde solo el nombre de la accion (ej: ACTION1). Respuesta:"])


def dir_words(sr: int, sc: int) -> str:
    """(0,+1) -> '1 a la derecha'. Nombrar la direccion evita que el modelo tenga
    que interpretar un vector antes de razonar con el."""
    parts = []
    if sr < 0:
        parts.append(f"{-sr} arriba")
    if sr > 0:
        parts.append(f"{sr} abajo")
    if sc < 0:
        parts.append(f"{-sc} a la izquierda")
    if sc > 0:
        parts.append(f"{sc} a la derecha")
    return " y ".join(parts) if parts else "nada"


def _parse_shift(v: str) -> tuple[int, int]:
    a, b = v.replace("move", "").split()
    return int(a), int(b)


def inverse_map(effects: dict) -> str:
    """Mapa DIRECCION -> ACCION, en vez de ACCION -> vector.

    Medido: con la tabla vectorial el 4B acierta 81.1% cuando la meta esta en un
    eje puro pero solo 51.8% en diagonal, y 58.4% cuando solo hay 2 acciones
    disponibles (ninguna apunta a la meta, hay que comparar reducciones de
    distancia). El mapa inverso quita el paso de inversion mental: el modelo
    pregunta 'quiero ir a la derecha' y lee la accion.
    """
    cardinales = (("ARRIBA", (-1, 0)), ("ABAJO", (1, 0)),
                  ("IZQUIERDA", (0, -1)), ("DERECHA", (0, 1)))
    out = []
    for nombre, vec in cardinales:
        hits = [a for a, v in effects.items()
                if sum(x * y for x, y in zip(_parse_shift(v), vec)) > 0]
        if hits:
            out.append(f"    para ir {nombre}: {', '.join(sorted(hits))}")
    return "\n".join(out)


def prompt_plan(item, with_table: bool) -> str:
    p = ["Un objeto debe llegar a una casilla objetivo en una rejilla.",
         f"Posicion actual del objeto (fila, columna): {item['player']}",
         f"Posicion objetivo (fila, columna): {item['target']}", ""]
    if with_table:
        tab = "\n".join(f"  {a}: {v}" for a, v in item["effects_table"].items())
        p += ["Efecto MEDIDO de cada accion (filas, columnas):", tab, ""]
    else:
        p += ["Acciones disponibles: " + ", ".join(item["effects_table"].keys()), ""]
    p += ["Que accion acerca mas el objeto al objetivo?",
          "Responde solo el nombre de la accion (ej: ACTION1). Respuesta:"]
    return "\n".join(p)


def normalize(text: str, kind: str) -> str:
    t = (text or "").strip().lower().replace("**", "")
    # OJO: todo tipo cuya respuesta sea un nombre de accion tiene que estar aqui.
    # avoid_inert falto al anadirlo y sus dos brazos dieron 0/31 IDENTICO — no era
    # un resultado, era que las respuestas caian a la rama de effect_of_action y
    # nunca podian casar. Mismo sintoma que ya delato otros dos fallos: dos
    # condiciones que deberian diferir dando exactamente el mismo numero.
    if kind in ("which_action", "plan_action", "avoid_inert"):
        m = re.search(r"action\s*([1-7])", t)
        return f"ACTION{m.group(1)}" if m else t[:20]
    if "none" in t:
        return "none"
    m = re.search(r"move\s*(-?\d+)\s*[, ]\s*(-?\d+)", t)
    if m:
        return f"move {int(m.group(1))} {int(m.group(2))}"
    if "change" in t:
        return "change"
    return t[:20]


# (nombre, tipo de item, constructor)  -- los pares A y B comparten items
def prompt_plan_words(item) -> str:
    """V3: la misma tabla pero con las direcciones NOMBRADAS."""
    tab = "\n".join(f"  {a}: mueve {dir_words(*_parse_shift(v))}"
                    for a, v in item["effects_table"].items())
    return "\n".join([
        "Un objeto debe llegar a una casilla objetivo en una rejilla.",
        f"Posicion actual del objeto (fila, columna): {item['player']}",
        f"Posicion objetivo (fila, columna): {item['target']}", "",
        "Efecto MEDIDO de cada accion:", tab, "",
        "Que accion acerca mas el objeto al objetivo?",
        "Responde solo el nombre de la accion (ej: ACTION1). Respuesta:"])


def prompt_plan_inverse(item) -> str:
    """V4: tabla vectorial + mapa inverso direccion -> accion."""
    tab = "\n".join(f"  {a}: {v}" for a, v in item["effects_table"].items())
    return "\n".join([
        "Un objeto debe llegar a una casilla objetivo en una rejilla.",
        f"Posicion actual del objeto (fila, columna): {item['player']}",
        f"Posicion objetivo (fila, columna): {item['target']}", "",
        "Efecto MEDIDO de cada accion (filas, columnas):", tab, "",
        "Resumen por direccion:", inverse_map(item["effects_table"]), "",
        # La pista tiene que ser CIERTA: algunas acciones combinan los dos ejes y
        # si llegan solas. Afirmar lo contrario seria meter un hecho falso, que es
        # el mismo error que ya cazamos en el detector.
        "Si el objetivo esta en diagonal, mira si alguna accion mueve en los dos "
        "ejes a la vez; si ninguna lo hace, elige la que reduzca mas la distancia "
        "total.", "",
        "Que accion acerca mas el objeto al objetivo?",
        "Responde solo el nombre de la accion (ej: ACTION1). Respuesta:"])


def prompt_inert(item, marcar: bool) -> str:
    """D: ¿vale la pena DECIR que una accion es inerte, o basta con omitirla?

    En los dos brazos las acciones inertes estan DISPONIBLES para elegir (si no,
    no serian distractores y la pregunta no mediria nada). Lo que cambia es si la
    nota las declara muertas o simplemente no habla de ellas.
    """
    todas = sorted(set(item["effects_table"]) | set(item["inert_actions"]))
    p = ["Un objeto debe llegar a una casilla objetivo en una rejilla.",
         f"Posicion actual del objeto (fila, columna): {item['player']}",
         f"Posicion objetivo (fila, columna): {item['target']}", "",
         "Acciones disponibles: " + ", ".join(todas), "",
         "Efecto MEDIDO de cada accion:"]
    for a in todas:
        if a in item["effects_table"]:
            p.append(f"  {a}: mueve {dir_words(*_parse_shift(item['effects_table'][a]))}")
        elif marcar:
            p.append(f"  {a}: SIN EFECTO, no cambia nada — no gastes turnos en ella")
    p += ["", "Que accion acerca mas el objeto al objetivo?",
          "Responde solo el nombre de la accion (ej: ACTION1). Respuesta:"]
    return "\n".join(p)


def dir_words_en(sr: int, sc: int) -> str:
    parts = []
    if sr < 0:
        parts.append(f"{-sr} up")
    if sr > 0:
        parts.append(f"{sr} down")
    if sc < 0:
        parts.append(f"{-sc} left")
    if sc > 0:
        parts.append(f"{sc} right")
    return " and ".join(parts) if parts else "nothing"


def prompt_plan_lang(item, idioma: str) -> str:
    """E: ¿en que IDIOMA conviene inyectar la nota?

    DESAJUSTE QUE EL BANCO NO ESTABA PROBANDO. El prompt del harness esta en
    ingles ("You are a coding agent solving a grid-based puzzle game") y nuestra
    nota se inyecta en espanol. Todas las medidas anteriores usaron prompt espanol
    + nota espanola, un regimen que en produccion NO ocurre. Aqui se comparan:
      en    : marco y nota en ingles (lo que haria un harness coherente)
      mixto : marco en ingles + nota en espanol (lo que hariamos hoy al desplegar)
      es    : todo en espanol (el regimen del banco hasta ahora, de control)
    """
    if idioma == "es":
        tab = "\n".join(f"  {a}: mueve {dir_words(*_parse_shift(v))}"
                        for a, v in item["effects_table"].items())
        return "\n".join([
            "Un objeto debe llegar a una casilla objetivo en una rejilla.",
            f"Posicion actual del objeto (fila, columna): {item['player']}",
            f"Posicion objetivo (fila, columna): {item['target']}", "",
            "Efecto MEDIDO de cada accion:", tab, "",
            "Que accion acerca mas el objeto al objetivo?",
            "Responde solo el nombre de la accion (ej: ACTION1). Respuesta:"])

    # marco en ingles; la nota cambia de idioma segun la variante
    if idioma == "en":
        tab = "\n".join(f"  {a}: moves {dir_words_en(*_parse_shift(v))}"
                        for a, v in item["effects_table"].items())
        cab = "MEASURED effect of each action:"
    else:  # mixto: nota en espanol dentro de un prompt en ingles
        tab = "\n".join(f"  {a}: mueve {dir_words(*_parse_shift(v))}"
                        for a, v in item["effects_table"].items())
        cab = "Efecto MEDIDO de cada accion:"
    return "\n".join([
        "An object must reach a target cell on a grid.",
        f"Current object position (row, column): {item['player']}",
        f"Target position (row, column): {item['target']}", "",
        cab, tab, "",
        "Which action brings the object closest to the target?",
        "Answer with the action name only (e.g. ACTION1). Answer:"])


def _fake_action(item) -> str:
    usadas = {int(a.replace("ACTION", "")) for a in item["effects_table"]}
    for i in (1, 2, 3, 4, 5):
        if i not in usadas:
            return f"ACTION{i}"
    return "ACTION5"


def prompt_wrong_entry(item, marcar_dudosa: bool) -> str:
    """F: ¿la marca de incertidumbre nos protege de NUESTROS PROPIOS errores?

    El detector acierta 96.6% con confianza >= 0.6, asi que ~1 de cada 30
    afirmaciones inyectadas es FALSA. Aqui se simula ese caso peor: se anade una
    entrada inventada que, de ser cierta, llevaria exactamente al objetivo — o sea
    que parece la mejor opcion con diferencia. La respuesta correcta sigue siendo
    la mejor accion REAL.
      V0: la entrada falsa se presenta como cualquier otra
      V1: se presenta con la degradacion honesta que usa render_effects_note
    Si la marca sirve, V1 cae menos en la trampa.
    """
    pr, pc = item["player"]
    tr, tc = item["target"]
    falsa = _fake_action(item)
    lineas = [f"  {a}: mueve {dir_words(*_parse_shift(v))}"
              for a, v in item["effects_table"].items()]
    cebo = f"mueve {dir_words(tr - pr, tc - pc)}"
    if marcar_dudosa:
        lineas.append(f"  {falsa}: {cebo}, pero su efecto NO es constante "
                      f"(33% de 6 obs) — verifica antes de fiarte")
    else:
        lineas.append(f"  {falsa}: {cebo}")
    return "\n".join([
        "Un objeto debe llegar a una casilla objetivo en una rejilla.",
        f"Posicion actual del objeto (fila, columna): {item['player']}",
        f"Posicion objetivo (fila, columna): {item['target']}", "",
        "Efecto MEDIDO de cada accion:", "\n".join(lineas), "",
        "Que accion acerca mas el objeto al objetivo, de forma FIABLE?",
        "Responde solo el nombre de la accion (ej: ACTION1). Respuesta:"])


VARIANTS = [
    ("A.V0_crudo",     "effect_of_action", lambda it: prompt_effect(it, False)),
    ("A.V1_objetos",   "effect_of_action", lambda it: prompt_effect(it, True)),
    ("B.V0_sin_tabla", "plan_action",      lambda it: prompt_plan(it, False)),
    ("B.V2_con_tabla", "plan_action",      lambda it: prompt_plan(it, True)),
    ("B.V3_palabras",  "plan_action",      prompt_plan_words),
    ("B.V4_inverso",   "plan_action",      prompt_plan_inverse),
    ("C.lookup",       "which_action",     lambda it: prompt_which(it)),
    ("D.V0_omitir",    "avoid_inert",      lambda it: prompt_inert(it, False)),
    ("D.V1_marcar",    "avoid_inert",      lambda it: prompt_inert(it, True)),
    ("E.es_en_es",     "plan_action",      lambda it: prompt_plan_lang(it, "es")),
    ("E.mixto_en_es",  "plan_action",      lambda it: prompt_plan_lang(it, "mixto")),
    ("E.en_en_en",     "plan_action",      lambda it: prompt_plan_lang(it, "en")),
    ("F.V0_sin_marca", "plan_action",      lambda it: prompt_wrong_entry(it, False)),
    ("F.V1_con_marca", "plan_action",      lambda it: prompt_wrong_entry(it, True)),
]

PAIRS = [("A objetos", "A.V0_crudo", "A.V1_objetos"),
         ("B tabla", "B.V0_sin_tabla", "B.V2_con_tabla"),
         ("B palabras vs vector", "B.V2_con_tabla", "B.V3_palabras"),
         ("B inverso vs vector", "B.V2_con_tabla", "B.V4_inverso"),
         ("D marcar inertes", "D.V0_omitir", "D.V1_marcar"),
         ("E nota es vs en (marco ingles)", "E.mixto_en_es", "E.en_en_en"),
         ("E marco es vs marco en", "E.es_en_es", "E.en_en_en"),
         ("F marca de incertidumbre", "F.V0_sin_marca", "F.V1_con_marca")]


def trivial_baselines(items) -> dict:
    """Clase mayoritaria por tipo: el suelo que hay que superar para no celebrar ruido."""
    out = {}
    for kind in sorted({i["type"] for i in items}):
        sub = [i["answer"] for i in items if i["type"] == kind]
        freq = {}
        for a in sub:
            freq[a] = freq.get(a, 0) + 1
        top = max(freq, key=freq.get)
        out[kind] = {"clase": top, "precision": round(freq[top] / len(sub), 3), "n": len(sub)}
    return out


def paired_contrast(res: dict) -> dict:
    """Contraste PAREADO: cuenta los discordantes, no dos porcentajes sueltos.

    Con n=125 y brazos correlacionados, comparar 0.42 vs 0.46 no dice nada; contar
    los items que un brazo acierta y el otro falla si. `p_signo` es la binomial de
    dos colas bajo H0 (los discordantes se reparten 50/50).
    """
    from math import comb
    out = {}
    for label, a, b in PAIRS:
        if a not in res or b not in res:
            continue
        va, vb = res[a]["por_item"], res[b]["por_item"]
        solo_a = sum(1 for x, y in zip(va, vb) if x and not y)
        solo_b = sum(1 for x, y in zip(va, vb) if y and not x)
        n = solo_a + solo_b
        k = min(solo_a, solo_b)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)) if n else 1.0
        out[label] = {f"solo_{a}": solo_a, f"solo_{b}": solo_b,
                      "discordantes": n, "p_signo": round(p, 4)}
    return out

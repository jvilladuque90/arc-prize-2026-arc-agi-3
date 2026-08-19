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
        p += [f"Ejemplo {i} - ANTES:", s["before"], f"Ejemplo {i} - DESPUES:", s["after"], ""]
    if with_objects and item.get("objects"):
        objs = ", ".join(f"color {o['color']} tam {o['size']} centro {o['center']}"
                         for o in item["objects"][:6])
        p += [f"Objetos detectados en el tablero: {objs}", ""]
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
    if kind in ("which_action", "plan_action"):
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
VARIANTS = [
    ("A.V0_crudo",     "effect_of_action", lambda it: prompt_effect(it, False)),
    ("A.V1_objetos",   "effect_of_action", lambda it: prompt_effect(it, True)),
    ("B.V0_sin_tabla", "plan_action",      lambda it: prompt_plan(it, False)),
    ("B.V2_con_tabla", "plan_action",      lambda it: prompt_plan(it, True)),
    ("C.lookup",       "which_action",     lambda it: prompt_which(it)),
]

PAIRS = [("A objetos", "A.V0_crudo", "A.V1_objetos"),
         ("B tabla", "B.V0_sin_tabla", "B.V2_con_tabla")]


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

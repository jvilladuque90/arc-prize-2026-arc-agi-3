"""Tests de la herencia entre niveles, sin GPU.

El que manda es `test_simula_el_borrado_del_harness`: reproduce la secuencia real
—el modelo escribe World model, gana nivel, el harness borra— y exige que el
conocimiento sobreviva en el unico campo que el harness conserva.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc3.level_inherit import (  # noqa: E402
    MAX_NIVELES, SEP, doblar, es_transicion, heredar, sello_de_nivel,
)

# textual de nuestras corridas (_tmp_nvfp4longlog)
REAL = ("A didn't complete; toggle works (clicked cells now orange). "
        "Switch to hypothesis B (W->blue, g->orange)")


def conocimiento(**kw):
    base = {"world_model": "", "goal_model": "", "action_model": "",
            "recent_findings": "", "open_questions": "", "current_plan": "",
            "cross_level_notes": ""}
    base.update(kw)
    return base


# ------------------------------------------------------------------ el sello

def test_el_sello_lleva_las_palabras_del_modelo():
    s = sello_de_nivel(1, conocimiento(world_model=REAL))
    assert "hypothesis B" in s, "debe conservar el texto del modelo, no parafrasearlo"
    assert "nivel 1" in s


def test_el_sello_junta_los_tres_campos_sustantivos():
    s = sello_de_nivel(2, conocimiento(world_model="W", goal_model="G", action_model="A"))
    assert "mundo: W" in s and "meta: G" in s and "acciones: A" in s


def test_sin_contenido_no_hay_sello():
    assert sello_de_nivel(1, conocimiento()) == ""
    assert sello_de_nivel(1, {}) == ""


def test_recorta_lo_muy_largo_sin_perder_el_principio():
    largo = " ".join(f"p{i}" for i in range(200))
    s = sello_de_nivel(1, conocimiento(world_model=largo))
    assert "p0 p1 p2" in s, "el principio es lo que importa"
    assert s.endswith("...")
    assert len(s.split()) < 80


# ---------------------------------------------------------------- acumulacion

def test_acumula_varios_niveles():
    v = ""
    for n in (1, 2):
        v = doblar(v, sello_de_nivel(n, conocimiento(world_model=f"saber{n}")))
    assert "saber1" in v and "saber2" in v
    assert v.count(SEP) == 1


def test_acota_el_crecimiento():
    v = ""
    for n in range(1, 8):
        v = doblar(v, sello_de_nivel(n, conocimiento(world_model=f"s{n}")))
    assert len(v.split(SEP)) == MAX_NIVELES, "no puede crecer sin limite"
    assert "s7" in v, "lo mas reciente debe quedarse"
    assert "s1" not in v, "lo mas viejo debe caer"


def test_no_duplica_el_mismo_sello():
    s = sello_de_nivel(1, conocimiento(world_model="X"))
    assert doblar(doblar("", s), s).count("nivel 1") == 1


# --------------------------------------------------------- LO QUE MANDA

def test_simula_el_borrado_del_harness():
    """La secuencia real: el modelo escribe World model, gana nivel, el harness borra."""
    k = conocimiento(world_model=REAL, goal_model="llevar el bloque al objetivo")

    # 1) llega la transicion: heredamos ANTES de que el harness borre
    k["cross_level_notes"] = heredar(k, nivel_ganado=1)

    # 2) el harness borra los seis campos, conserva cross_level_notes (codigo real)
    for campo in ("world_model", "goal_model", "action_model",
                  "recent_findings", "open_questions", "current_plan"):
        k[campo] = ""

    # 3) en el nivel 2 el conocimiento sigue ahi
    assert k["world_model"] == "", "el harness sigue borrando, no lo hemos tocado"
    assert "hypothesis B" in k["cross_level_notes"], "se perdio lo aprendido"
    assert "llevar el bloque" in k["cross_level_notes"]


def test_sin_nada_que_salvar_no_ensucia_lo_previo():
    k = conocimiento(cross_level_notes="algo previo")
    assert heredar(k, 1) == "algo previo"


def test_tolera_entradas_absurdas():
    assert heredar(None, 1) == ""
    assert heredar("no soy dict", 1) == ""
    assert doblar(None, "") == ""


def test_deteccion_de_transicion():
    assert es_transicion({"level_transition": True})
    assert not es_transicion({"level_transition": False})
    assert not es_transicion({})
    assert not es_transicion(None)

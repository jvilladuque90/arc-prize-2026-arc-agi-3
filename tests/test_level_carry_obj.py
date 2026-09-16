"""Tests de la nota v4 (consolidacion en vocabulario de OBJETOS), sin GPU.

Se prueba contra la segmentacion REAL del harness (`inference.utils.segmentation`),
que es la misma que alimenta `current_frame.segmentation` en el prompt del modelo:
si los hash no casaran, la nota citaria objetos que el modelo no puede encontrar.
"""

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
_HARNESS = RAIZ / "_tmp_nvfp4_bundle" / "src" / "ARC3-Inference"
if _HARNESS.is_dir():
    sys.path.insert(0, str(_HARNESS))

from arc3.level_carry_obj import (  # noqa: E402
    describir_objetos, render_nota_objetos, transiciones_objeto,
)

pytestmark = pytest.mark.skipif(
    not _HARNESS.is_dir(), reason="hace falta la segmentacion del harness"
)


def tablero(n=16):
    return [[0] * n for _ in range(n)]


def pinta(g, r0, c0, alto, ancho, v):
    for r in range(r0, r0 + alto):
        for c in range(c0, c0 + ancho):
            g[r][c] = v


class Frame:
    def __init__(self, grid, level):
        self.grid = [row[:] for row in grid]
        self.level = level


class Entry:
    def __init__(self, action, frame):
        self.action = action
        self.frame = frame


# --------------------------------------------------------------------------- objetos

def test_detecta_objeto_que_se_mueve():
    antes = tablero()
    pinta(antes, 2, 2, 2, 2, 4)        # pieza 'c' (charcoal) arriba a la izquierda
    pinta(antes, 10, 10, 3, 3, 11)     # objetivo 'Y' (yellow)
    despues = tablero()
    pinta(despues, 6, 2, 2, 2, 4)      # la misma pieza, 4 filas mas abajo
    pinta(despues, 10, 10, 3, 3, 11)

    o = describir_objetos(antes, despues)
    assert o["movido"] is not None, "no detecto el desplazamiento"
    _na, nd, (dr, dc) = o["movido"]
    assert (dr, dc) == (4, 0)
    assert nd["pixels"] == 4
    assert nd["color"] == "c"


def test_detecta_objeto_que_desaparece():
    antes = tablero()
    pinta(antes, 10, 10, 3, 3, 11)     # el objetivo amarillo
    despues = tablero()                # se lo comieron

    o = describir_objetos(antes, despues)
    assert o["desaparecido"] is not None
    assert o["desaparecido"]["color"] == "Y"
    assert o["desaparecido"]["pixels"] == 9


def test_prefiere_el_objeto_bajo_el_click():
    antes = tablero()
    pinta(antes, 2, 2, 4, 4, 8)        # objeto grande 'R'
    pinta(antes, 12, 12, 1, 2, 6)      # objeto pequeno 'M'
    despues = tablero()
    pinta(despues, 3, 2, 4, 4, 8)      # ambos se mueven
    pinta(despues, 13, 12, 1, 2, 6)

    sin_click = describir_objetos(antes, despues)
    assert sin_click["movido"][1]["pixels"] == 16, "sin click deberia mandar el grande"

    con_click = describir_objetos(antes, despues, "MOUSE(row=13, col=12)")
    assert con_click["movido"][1]["pixels"] == 2, "el click debe imponer el protagonista"


def test_vecino_nuevo_es_la_relacion_que_cambio():
    antes = tablero()
    pinta(antes, 2, 2, 2, 2, 4)
    pinta(antes, 10, 2, 3, 3, 11)
    despues = tablero()
    # filas 8-9: la de abajo (9) comparte arista con la fila 10 del amarillo
    pinta(despues, 8, 2, 2, 2, 4)
    pinta(despues, 10, 2, 3, 3, 11)

    o = describir_objetos(antes, despues)
    assert o["movido"] is not None
    colores = {n["color"] for n in o["nuevos_vecinos"]}
    assert "Y" in colores, f"esperaba tocar el amarillo, vecinos: {colores}"


def test_hash_es_el_mismo_que_ve_el_modelo():
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    from inference.utils.segmentation import segment_layer

    antes = tablero()
    pinta(antes, 2, 2, 2, 2, 4)
    despues = tablero()
    pinta(despues, 6, 2, 2, 2, 4)

    o = describir_objetos(antes, despues)
    nd = o["movido"][1]
    del_modelo = {n["hash"] for n in segment_layer(despues, ARC_COLOR_CHARS)["nodes"]}
    assert nd["hash"] in del_modelo, "la nota citaria un hash que el modelo no ve"


# ----------------------------------------------------------------------------- nota

def _transiciones_de_ejemplo():
    antes = tablero()
    pinta(antes, 2, 2, 2, 2, 4)
    pinta(antes, 10, 2, 3, 3, 11)
    despues = tablero()
    pinta(despues, 8, 2, 2, 2, 4)     # queda pegada al amarillo: hay relacion que contar
    pinta(despues, 10, 2, 3, 3, 11)
    return transiciones_objeto([
        Entry("DOWN", Frame(antes, 1)),
        Entry("DOWN", Frame(despues, 2)),
    ])


def test_nota_vacia_en_nivel_1_y_sin_transiciones():
    assert render_nota_objetos(1, _transiciones_de_ejemplo()) == ""
    assert render_nota_objetos(2, []) == ""
    assert render_nota_objetos("x", _transiciones_de_ejemplo()) == ""


def test_nota_habla_de_objetos_y_no_de_teclas():
    nota = render_nota_objetos(2, _transiciones_de_ejemplo())
    assert nota, "deberia producir nota en el nivel 2"
    assert "objeto" in nota and "hash" in nota
    assert "segmentation" in nota, "debe anclar al vocabulario que el modelo ya usa"
    # el punto de v4: NO nombrar la tecla
    for tecla in ("DOWN", "LEFT", "RIGHT", "SPACE", "ACTION1", "ACTION6"):
        assert tecla not in nota, f"v4 no debe nombrar la tecla, aparecio {tecla}"


def test_nota_cabe_en_el_presupuesto_de_v1():
    nota = render_nota_objetos(2, _transiciones_de_ejemplo())
    # v1 medida en produccion: ~93 tokens. Se aproxima por palabras con holgura.
    assert len(nota.split()) <= 70, f"nota demasiado larga: {len(nota.split())} palabras"
    assert len(nota) <= 600, f"nota demasiado larga: {len(nota)} caracteres"


def test_sin_historia_de_objetos_no_inventa():
    """Si el tablero no cambia, mejor ninguna nota que una nota vacia de contenido."""
    g = tablero()
    pinta(g, 2, 2, 2, 2, 4)
    tr = transiciones_objeto([Entry("DOWN", Frame(g, 1)), Entry("DOWN", Frame(g, 2))])
    assert render_nota_objetos(2, tr) == ""


def test_degrada_a_vacio_si_no_hay_segmentacion(monkeypatch):
    import arc3.level_carry_obj as m
    monkeypatch.setattr(m, "_segmentar", lambda g: None)
    assert describir_objetos([[0]], [[1]])["movido"] is None

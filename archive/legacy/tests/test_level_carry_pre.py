"""Tests de la nota v5 (POSICION GANADORA / precondicion), sin GPU.

El test que manda es `test_nunca_cruza_la_frontera_de_nivel`: el fallo que v5 arregla
(DESIGN 8.62) era describir el redibujado del cambio de nivel creyendo que era la jugada.
"""

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
_HARNESS = RAIZ / "_tmp_nvfp4_bundle" / "src" / "ARC3-Inference"
if _HARNESS.is_dir():
    sys.path.insert(0, str(_HARNESS))

from arc3.level_carry_pre import (  # noqa: E402
    contactos, marcas_victoria, objeto_bajo_click, objeto_controlado, render_nota_previa,
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


# ------------------------------------------------------------------ objeto controlado

def test_detecta_la_pieza_que_responde_a_las_jugadas():
    anterior = tablero()
    pinta(anterior, 4, 2, 2, 2, 4)     # la pieza
    pinta(anterior, 10, 2, 3, 3, 11)   # el objetivo, quieto
    antes = tablero()
    pinta(antes, 5, 2, 2, 2, 4)        # bajo una fila
    pinta(antes, 10, 2, 3, 3, 11)

    ctrl = objeto_controlado(anterior, antes)
    assert ctrl is not None
    nodo, (dr, dc) = ctrl
    assert (dr, dc) == (1, 0)
    assert nodo["color"] == "c" and nodo["pixels"] == 4


def test_el_fondo_no_es_la_pieza():
    anterior = tablero()
    pinta(anterior, 4, 2, 2, 2, 4)
    antes = tablero()
    pinta(antes, 5, 2, 2, 2, 4)
    ctrl = objeto_controlado(anterior, antes)
    assert ctrl is not None
    assert ctrl[0]["color"] != "W", "el fondo no puede ser el protagonista"


# ----------------------------------------------------------------------- clic

def test_clic_elige_el_objeto_mas_ajustado_no_el_contenedor():
    g = tablero()
    pinta(g, 4, 4, 8, 8, 8)            # marco grande 'R'
    pinta(g, 7, 7, 2, 2, 11)           # pieza pequena 'Y' dentro

    n = objeto_bajo_click(g, "MOUSE(row=7, col=7)")
    assert n is not None
    assert n["color"] == "Y" and n["pixels"] == 4, "debe elegir el interior, no el marco"


def test_sin_clic_no_hay_objetivo():
    g = tablero()
    pinta(g, 4, 4, 2, 2, 8)
    assert objeto_bajo_click(g, "LEFT") is None


def test_contactos_son_los_vecinos_reales():
    g = tablero()
    pinta(g, 8, 2, 2, 2, 4)            # filas 8-9
    pinta(g, 10, 2, 3, 3, 11)          # filas 10-12: comparten arista
    pinta(g, 2, 12, 2, 2, 6)           # lejos, no toca
    seg_nodo = objeto_bajo_click(g, "MOUSE(row=8, col=2)")
    cont = contactos(g, seg_nodo)
    colores = {n["color"] for n in cont}
    assert "Y" in colores
    assert "M" not in colores, "no deberia listar objetos que no tocan"


# ------------------------------------------------------- LA FRONTERA DE NIVEL

def test_nunca_cruza_la_frontera_de_nivel():
    """El fallo de v1-v4: el tablero del nivel siguiente se colaba en la descripcion."""
    n1a = tablero()
    pinta(n1a, 4, 2, 2, 2, 4)
    pinta(n1a, 10, 2, 3, 3, 11)
    n1b = tablero()
    pinta(n1b, 8, 2, 2, 2, 4)          # la pieza baja y queda pegada al objetivo
    pinta(n1b, 10, 2, 3, 3, 11)
    # el nivel 2 es un tablero COMPLETAMENTE distinto y enorme
    n2 = tablero()
    pinta(n2, 0, 0, 14, 14, 9)         # 196 celdas de 'b': si se colara, dominaria

    marcas = marcas_victoria([
        Entry("DOWN", Frame(n1a, 1)),
        Entry("DOWN", Frame(n1b, 1)),
        Entry("DOWN", Frame(n2, 2)),
    ])
    assert len(marcas) == 1
    pos = marcas[0]["posicion"]
    citados = []
    if pos["controlado"]:
        citados.append(pos["controlado"][0])
    if pos["clic"]:
        citados.append(pos["clic"])
    citados += pos["contactos"]
    assert citados, "deberia describir algo del nivel 1"
    for n in citados:
        assert n["color"] != "b", "se colo un objeto del nivel siguiente"
        assert n["pixels"] <= 9, f"objeto demasiado grande: {n['pixels']} celdas"
    # y la pieza correcta es la del nivel 1
    assert pos["controlado"] is not None
    assert pos["controlado"][0]["pixels"] == 4


def test_nota_no_menciona_desapariciones_masivas():
    """v4 escribia 'desaparecio el objeto W de 624 celdas'. v5 no habla de desapariciones."""
    n1a = tablero()
    pinta(n1a, 4, 2, 2, 2, 4)
    pinta(n1a, 10, 2, 3, 3, 11)
    n1b = tablero()
    pinta(n1b, 8, 2, 2, 2, 4)
    pinta(n1b, 10, 2, 3, 3, 11)
    n2 = tablero()
    marcas = marcas_victoria([
        Entry("DOWN", Frame(n1a, 1)),
        Entry("DOWN", Frame(n1b, 1)),
        Entry("DOWN", Frame(n2, 2)),
    ])
    nota = render_nota_previa(2, marcas)
    assert "desaparecio" not in nota


# ----------------------------------------------------------------------- nota

def _marcas_ejemplo():
    a = tablero()
    pinta(a, 4, 2, 2, 2, 4)
    pinta(a, 10, 2, 3, 3, 11)
    b = tablero()
    pinta(b, 8, 2, 2, 2, 4)
    pinta(b, 10, 2, 3, 3, 11)
    c = tablero()
    pinta(c, 1, 1, 3, 3, 6)
    return marcas_victoria([
        Entry("DOWN", Frame(a, 1)), Entry("DOWN", Frame(b, 1)), Entry("DOWN", Frame(c, 2)),
    ])


def test_nota_vacia_en_nivel_1_y_sin_marcas():
    assert render_nota_previa(1, _marcas_ejemplo()) == ""
    assert render_nota_previa(2, []) == ""
    assert render_nota_previa(None, _marcas_ejemplo()) == ""


def test_nota_habla_de_objetos_y_no_de_teclas():
    nota = render_nota_previa(2, _marcas_ejemplo())
    assert nota
    assert "objeto" in nota and "hash" in nota and "segmentacion" in nota
    for tecla in ("DOWN", "LEFT", "RIGHT", "UP", "SPACE", "ACTION1", "ACTION6"):
        assert tecla not in nota, f"v5 no debe nombrar la tecla, aparecio {tecla}"


def test_nota_no_afirma_lo_inobservable():
    nota = render_nota_previa(2, _marcas_ejemplo())
    assert "no se puede observar" in nota, "debe declarar que el efecto no es observable"


def test_nota_cabe_en_el_presupuesto():
    nota = render_nota_previa(2, _marcas_ejemplo())
    assert len(nota.split()) <= 75, f"{len(nota.split())} palabras"
    assert len(nota) <= 650, f"{len(nota)} caracteres"


def test_degrada_a_vacio_sin_segmentacion(monkeypatch):
    import arc3.level_carry_pre as m
    monkeypatch.setattr(m, "_segmentar", lambda g: None)
    assert objeto_controlado([[0]], [[1]]) is None
    assert objeto_bajo_click([[0]], "MOUSE(row=0, col=0)") is None

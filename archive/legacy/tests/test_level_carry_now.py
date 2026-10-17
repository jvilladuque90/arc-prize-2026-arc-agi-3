"""Tests de la nota v7 (anclada en la VISTA ACTUAL), sin GPU.

Los dos que mandan (DESIGN 8.64):
  - `test_solo_cita_objetos_que_existen_ahora`: el fallo de v5/v6 era mandar a buscar
    piezas del nivel anterior que ya no estan en pantalla.
  - `test_nota_vacia_si_no_queda_nada_en_pantalla`: si no se resuelve nada, mejor nada.
"""

import inspect
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
_HARNESS = RAIZ / "_tmp_nvfp4_bundle" / "src" / "ARC3-Inference"
if _HARNESS.is_dir():
    sys.path.insert(0, str(_HARNESS))

from arc3 import level_carry_now as now  # noqa: E402
from arc3 import level_carry_pre as pre  # noqa: E402

pytestmark = pytest.mark.skipif(
    not _HARNESS.is_dir(), reason="hace falta la segmentacion del harness"
)

FUNCIONES = ("objeto_controlado", "objeto_bajo_click", "contactos", "describir_posicion",
             "marcas_victoria", "_segmentar", "_caja", "_por_hash", "_vecinos",
             "_etiqueta", "_celda_click", "_es_fondo", "_nivel", "_grid")


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


def _historia_y_nivel2(nivel2):
    """Nivel 1: pieza 'c' de 4 celdas que baja hasta tocar un 'Y' de 9. Luego, nivel 2."""
    a = tablero()
    pinta(a, 4, 2, 2, 2, 4)
    pinta(a, 10, 2, 3, 3, 11)
    b = tablero()
    pinta(b, 8, 2, 2, 2, 4)
    pinta(b, 10, 2, 3, 3, 11)
    h = [Entry("DOWN", Frame(a, 1)), Entry("DOWN", Frame(b, 1)),
         Entry("DOWN", Frame(nivel2, 2))]
    return now.marcas_victoria(h)


def test_extraccion_identica_a_v5():
    for f in FUNCIONES:
        assert inspect.getsource(getattr(now, f)) == inspect.getsource(getattr(pre, f)), f


# ------------------------------------------------------------- LO QUE MANDA

def test_solo_cita_objetos_que_existen_ahora():
    """El nivel 2 trae la misma pieza en otro sitio, mas objetos nuevos que no vienen a cuento."""
    n2 = tablero()
    pinta(n2, 1, 12, 2, 2, 4)      # la MISMA pieza 'c' de 4 celdas, ahora arriba a la derecha
    pinta(n2, 12, 12, 1, 3, 6)     # un 'M' nuevo, irrelevante
    nota = now.render_nota_actual(2, _historia_y_nivel2(n2), n2)
    assert nota, "la pieza sigue en pantalla: deberia haber nota"
    assert "el mismo" in nota, "es el objeto literal: debe decirlo"
    assert "fila 1, col 12" in nota, "debe decir donde esta AHORA"
    assert "de 3 celdas" not in nota, "no debe citar objetos ajenos a la jugada"


def test_nota_vacia_si_no_queda_nada_en_pantalla():
    n2 = tablero()
    pinta(n2, 1, 1, 4, 4, 9)       # tablero completamente distinto: ni 'c' ni 'Y'
    assert now.render_nota_actual(2, _historia_y_nivel2(n2), n2) == ""


def test_acepta_la_pieza_analoga_por_color_y_tamano():
    n2 = tablero()
    pinta(n2, 3, 3, 1, 5, 4)       # 'c' de 5 celdas: mismo color, tamano parecido (4 -> 5)
    nota = now.render_nota_actual(2, _historia_y_nivel2(n2), n2)
    assert nota
    assert "uno igual" in nota, "no es el hash exacto: no debe decir 'el mismo'"


def test_rechaza_el_mismo_color_con_tamano_muy_distinto():
    n2 = tablero()
    pinta(n2, 3, 3, 5, 5, 4)       # 'c' de 25 celdas frente a 4: demasiado lejos
    assert now.render_nota_actual(2, _historia_y_nivel2(n2), n2) == ""


def test_relaciona_los_dos_objetos_cuando_ambos_estan():
    n2 = tablero()
    pinta(n2, 1, 1, 2, 2, 4)       # la pieza
    pinta(n2, 12, 12, 3, 3, 11)    # y el objetivo
    nota = now.render_nota_actual(2, _historia_y_nivel2(n2), n2)
    assert "Llevalos a la misma relacion" in nota
    assert nota.count("- ") >= 2, "deberia citar los dos"


# ---------------------------------------------------------------- lo basico

def test_vacia_en_nivel_1_sin_marcas_y_sin_tablero():
    n2 = tablero()
    pinta(n2, 1, 12, 2, 2, 4)
    m = _historia_y_nivel2(n2)
    assert now.render_nota_actual(1, m, n2) == ""
    assert now.render_nota_actual(2, [], n2) == ""
    assert now.render_nota_actual(2, m, None) == ""
    assert now.render_nota_actual(None, m, n2) == ""


def test_no_nombra_teclas():
    n2 = tablero()
    pinta(n2, 1, 12, 2, 2, 4)
    nota = now.render_nota_actual(2, _historia_y_nivel2(n2), n2)
    for tecla in ("DOWN", "LEFT", "RIGHT", "UP", "SPACE", "ACTION1", "ACTION6"):
        assert tecla not in nota, f"aparecio {tecla}"


def test_cabe_en_el_presupuesto():
    n2 = tablero()
    pinta(n2, 1, 1, 2, 2, 4)
    pinta(n2, 12, 12, 3, 3, 11)
    nota = now.render_nota_actual(2, _historia_y_nivel2(n2), n2)
    assert len(nota.split()) <= 75, f"{len(nota.split())} palabras"
    assert len(nota) <= 650, f"{len(nota)} caracteres"


def test_degrada_a_vacio_sin_segmentacion(monkeypatch):
    n2 = tablero()
    pinta(n2, 1, 12, 2, 2, 4)
    m = _historia_y_nivel2(n2)
    monkeypatch.setattr(now, "_segmentar", lambda g: None)
    assert now.render_nota_actual(2, m, n2) == ""

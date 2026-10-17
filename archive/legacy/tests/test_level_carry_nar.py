"""Tests de la nota v6 (contenido de v5 con forma de v4), sin GPU.

El test que manda es `test_extraccion_identica_a_v5`: si el contenido no fuera
exactamente el de v5, el brazo no probaria la FORMA y el resultado no diria nada
(DESIGN 8.63).
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

from arc3 import level_carry_nar as nar  # noqa: E402
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


def _historia(accion="DOWN"):
    a = tablero()
    pinta(a, 4, 2, 2, 2, 4)
    pinta(a, 10, 2, 3, 3, 11)
    b = tablero()
    pinta(b, 8, 2, 2, 2, 4)
    pinta(b, 10, 2, 3, 3, 11)
    c = tablero()
    pinta(c, 1, 1, 3, 3, 6)
    return [Entry(accion, Frame(a, 1)), Entry(accion, Frame(b, 1)), Entry(accion, Frame(c, 2))]


# --------------------------------------------------------- LO QUE MANDA

def test_extraccion_identica_a_v5_en_codigo():
    for f in FUNCIONES:
        assert inspect.getsource(getattr(nar, f)) == inspect.getsource(getattr(pre, f)), f


def test_extraccion_identica_a_v5_en_salida():
    h = _historia()
    ma, mb = nar.marcas_victoria(h), pre.marcas_victoria(h)
    assert len(ma) == len(mb) == 1
    pa, pb = ma[0]["posicion"], mb[0]["posicion"]
    assert (pa["controlado"] is None) == (pb["controlado"] is None)
    if pa["controlado"]:
        assert pa["controlado"][0]["hash"] == pb["controlado"][0]["hash"]
        assert pa["controlado"][1] == pb["controlado"][1]
    assert [n["hash"] for n in pa["contactos"]] == [n["hash"] for n in pb["contactos"]]


# ------------------------------------------------------------ la forma

def test_la_nota_es_una_secuencia_no_un_inventario():
    nota = nar.render_nota_narrativa(2, nar.marcas_victoria(_historia()))
    assert nota
    assert "y el nivel cayo" in nota, "falta la consecuencia: es lo que la hace narrativa"
    assert "moviste" in nota or "clicaste" in nota, "falta el verbo en segunda persona"
    assert "Repite esa jugada" in nota, "falta el cierre imperativo"


def test_no_lleva_la_salvedad_epistemica_de_v5():
    nota = nar.render_nota_narrativa(2, nar.marcas_victoria(_historia()))
    assert "no se puede observar" not in nota
    v5 = pre.render_nota_previa(2, pre.marcas_victoria(_historia()))
    assert "no se puede observar" in v5, "v5 si debe llevarla (control del experimento)"


def test_no_nombra_teclas():
    nota = nar.render_nota_narrativa(2, nar.marcas_victoria(_historia()))
    for tecla in ("DOWN", "LEFT", "RIGHT", "UP", "SPACE", "ACTION1", "ACTION6"):
        assert tecla not in nota, f"aparecio {tecla}"


def test_variante_de_clic():
    a = tablero()
    pinta(a, 4, 4, 8, 8, 8)
    pinta(a, 7, 7, 2, 2, 11)
    h = [Entry("MOUSE(row=7, col=7)", Frame(a, 1)),
         Entry("MOUSE(row=7, col=7)", Frame(a, 1)),
         Entry("MOUSE(row=7, col=7)", Frame(tablero(), 2))]
    nota = nar.render_nota_narrativa(2, nar.marcas_victoria(h))
    assert "clicaste sobre el objeto" in nota
    assert "y el nivel cayo" in nota


# --------------------------------------------------- lo heredado de v5

def test_sigue_sin_cruzar_la_frontera_de_nivel():
    a = tablero()
    pinta(a, 4, 2, 2, 2, 4)
    pinta(a, 10, 2, 3, 3, 11)
    b = tablero()
    pinta(b, 8, 2, 2, 2, 4)
    pinta(b, 10, 2, 3, 3, 11)
    n2 = tablero()
    pinta(n2, 0, 0, 14, 14, 9)      # 196 celdas en el nivel siguiente
    marcas = nar.marcas_victoria([Entry("DOWN", Frame(a, 1)), Entry("DOWN", Frame(b, 1)),
                                  Entry("DOWN", Frame(n2, 2))])
    nota = nar.render_nota_narrativa(2, marcas)
    assert "196 celdas" not in nota
    assert "de 9 celdas" in nota or "de 4 celdas" in nota


def test_vacia_en_nivel_1_y_sin_marcas():
    assert nar.render_nota_narrativa(1, nar.marcas_victoria(_historia())) == ""
    assert nar.render_nota_narrativa(2, []) == ""
    assert nar.render_nota_narrativa(None, nar.marcas_victoria(_historia())) == ""


def test_cabe_en_el_presupuesto():
    nota = nar.render_nota_narrativa(2, nar.marcas_victoria(_historia()))
    assert len(nota.split()) <= 75, f"{len(nota.split())} palabras"
    assert len(nota) <= 650, f"{len(nota)} caracteres"

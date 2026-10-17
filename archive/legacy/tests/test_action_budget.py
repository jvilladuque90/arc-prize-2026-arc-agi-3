"""Tests del empuje al agrupamiento de acciones (DESIGN 8.66), sin GPU.

Lo que mandan: que la nota CALLE donde el agente ya agrupa bien (es la condicion que la
distingue del texto de cada turno, que nunca ha pagado) y que hable donde hay hueco.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from arc3.action_budget import (  # noqa: E402
    MINIMO_TURNOS, UMBRAL, ratio, render_nota_presupuesto, turnos_de,
)


class Agente:
    pass


# ------------------------------------------------------- cuando CALLA

def test_calla_si_ya_agrupa_bien():
    """re86 hacia 10,4 acciones por turno; tn36 9,5. Ahi no hay nada que decir."""
    assert render_nota_presupuesto(104, 10) == ""
    assert render_nota_presupuesto(95, 10) == ""


def test_calla_justo_en_el_umbral():
    assert render_nota_presupuesto(30, 10) == "", "3,0 por turno es la mediana: no es hueco"


def test_calla_al_principio_sin_ratio_fiable():
    for turnos in range(MINIMO_TURNOS):
        assert render_nota_presupuesto(1, turnos) == ""


def test_calla_con_entradas_absurdas():
    assert render_nota_presupuesto(0, 10) == ""
    assert render_nota_presupuesto("x", 10) == ""
    assert render_nota_presupuesto(10, None) == ""


# ------------------------------------------------------- cuando HABLA

def test_habla_donde_hay_hueco():
    """su15 hacia 1,4 por turno; tr87 y r11l 1,2. Son los que necesitan el empujon."""
    nota = render_nota_presupuesto(14, 10)
    assert nota
    assert "1.4 por turno" in nota
    assert "14 acciones en 10 turnos" in nota


def test_la_nota_dice_el_hecho_que_el_agente_no_ve():
    nota = render_nota_presupuesto(12, 10)
    assert "RELOJ" in nota, "el hecho clave: la partida se corta por tiempo"
    assert "cuesta lo mismo" in nota, "y que el turno cuesta igual lleve 1 accion o 5"
    assert "action([" in nota, "debe mostrar la forma concreta de agrupar"


def test_no_dicta_cuantas_acciones():
    nota = render_nota_presupuesto(12, 10)
    for imperativo in ("debes mandar", "manda siempre", "usa 5", "al menos"):
        assert imperativo not in nota


def test_es_breve():
    nota = render_nota_presupuesto(12, 10)
    assert len(nota.split()) <= 60, f"{len(nota.split())} palabras"


# ------------------------------------------------------------ contador

def test_el_contador_va_por_agente():
    a, b = Agente(), Agente()
    assert [turnos_de(a) for _ in range(3)] == [1, 2, 3]
    assert turnos_de(b) == 1, "cada partida cuenta sus propios turnos"
    assert turnos_de(a) == 4


def test_ratio():
    assert ratio(14, 10) == 1.4
    assert ratio(5, 0) == 0.0


def test_el_umbral_es_la_mediana_medida():
    assert UMBRAL == 3.0, "la mediana del banco es 3,5; el umbral se fija justo por debajo"

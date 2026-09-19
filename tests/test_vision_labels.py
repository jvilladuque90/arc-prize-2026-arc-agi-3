"""Tests de la imagen etiquetada, sin GPU.

Los que mandan: que la rejilla NO pise ninguna celda de juego (va en el margen y en
lineas de 1 px), que los colores coincidan con la paleta del harness, y que el coste
en pixeles no supere al del harness (1024x1024).
"""

import base64
import io
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
_HARNESS = RAIZ / "_tmp_nvfp4_bundle" / "src" / "ARC3-Inference"
if _HARNESS.is_dir():
    sys.path.insert(0, str(_HARNESS))

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from arc3.vision_labels import (  # noqa: E402
    MARGEN, PALETA, PASO, REJILLA, componer, panel, parte_imagen, url_datos,
)

PIXELES_HARNESS = 1024 * 1024   # lo que manda hoy: 64x64 escalado x16


def tablero(n=64, v=0):
    return [[v] * n for _ in range(n)]


# ------------------------------------------------------------------ colores

@pytest.mark.skipif(not _HARNESS.is_dir(), reason="hace falta el bundle del harness")
def test_los_colores_son_EXACTAMENTE_los_del_harness():
    """Si no coincidieran, la imagen contradiria al ascii que ve en el mismo turno."""
    from inference.agent.vision_context import ARC_COLOR_MAP
    assert PALETA == ARC_COLOR_MAP, "la paleta se ha desviado de la del harness"


def test_la_rejilla_no_esta_en_la_paleta():
    assert REJILLA not in PALETA.values(), "la rejilla se confundiria con una celda"


def test_una_celda_conserva_su_color():
    g = tablero()
    g[20][30] = 11                      # amarillo
    im = panel(g, 10, "t")
    # centro de la celda (20,30), desplazado por el margen
    px = im.getpixel((MARGEN + 30 * 10 + 5, MARGEN + 20 * 10 + 5))
    assert px == PALETA[11], f"la celda cambio de color: {px}"


# ------------------------------------------------------------- la rejilla

def test_la_rejilla_cae_en_los_bordes_no_en_el_centro():
    g = tablero()
    im = panel(g, 10, "t")
    borde = im.getpixel((MARGEN + PASO * 10, MARGEN + 5))
    centro = im.getpixel((MARGEN + PASO * 10 + 5, MARGEN + 5))
    assert borde == REJILLA, "no hay linea donde deberia"
    assert centro != REJILLA, "la linea invade la celda"


def test_los_rotulos_van_en_el_margen():
    g = tablero()
    im = panel(g, 10, "t")
    # toda la banda superior por encima del margen es fondo o tinta, nunca tablero
    for x in range(0, im.width, 37):
        assert im.getpixel((x, 1)) in ((255, 255, 255), (0, 0, 0))


# ------------------------------------------------------------- composicion

def test_dos_paneles_cuando_hay_anterior():
    a, b = tablero(), tablero()
    solo = componer(None, b)
    dos = componer(a, b)
    assert dos.width > solo.width, "deberia haber dos paneles"


def test_sin_anterior_funciona_igual():
    im = componer(None, tablero())
    assert im.width > 0 and im.height > 0


def test_no_gasta_mas_pixeles_que_el_harness():
    dos = componer(tablero(), tablero())
    solo = componer(None, tablero())
    assert dos.width * dos.height <= PIXELES_HARNESS, (
        f"{dos.width}x{dos.height} supera al harness")
    assert solo.width * solo.height <= PIXELES_HARNESS


# ----------------------------------------------------------------- salida

def test_la_parte_es_un_png_valido():
    p = parte_imagen(tablero(), tablero())
    assert p["type"] == "image_url"
    url = p["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert im.format == "PNG"


def test_degrada_a_none_sin_tablero():
    assert parte_imagen(None, None) is None
    assert parte_imagen(tablero(), []) is None


def test_tolera_filas_irregulares_y_valores_raros():
    g = [[0, 1, 2], [3, 4], [5, "x", None, 99]]
    im = panel(g, 10, "t")
    assert im.width > 0


def test_url_datos_es_reversible():
    im = panel(tablero(8), 10, "t")
    u = url_datos(im)
    assert Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).size == im.size

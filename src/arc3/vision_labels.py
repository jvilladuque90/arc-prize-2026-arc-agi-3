"""Imagen etiquetada: rejilla con coordenadas, y el fotograma anterior al lado.

EL HUECO (DESIGN 8.75)
----------------------
El modelo SI recibe una imagen del tablero — 3.687 consultas a la cache multimodal lo
confirman — pero el harness la renderiza asi (`vision_context.py`):

    image = Image.new("RGB", (cols, rows), ...)          # un pixel por celda
    image = image.resize((cols*16, rows*16), NEAREST)    # 64x64 -> 1024x1024

**Un mapa de colores plano: sin etiquetas, sin rejilla, sin coordenadas, y solo el
fotograma actual.** Y al modelo se le pide emitir `MOUSE(row=X, col=Y)`: para acertar una
celda tiene que contar pixeles en 1024x1024 sin una sola referencia.

Tufa Labs, primero de la tabla con 18,81, hace justo lo contrario: *"renders recent frames
as labeled images"* — etiquetadas, y varios fotogramas.

Llevamos nueve brazos escribiendo texto al final del prompt y **cero** tocando la imagen.

QUE HACE ESTE MODULO
--------------------
1. Dibuja **rejilla cada 8 celdas** en cian puro (0,255,255), que **no esta en la paleta
   ARC** — la mas cercana es el celeste (136,216,241) y se distingue sin ambiguedad.
2. Rotula filas y columnas cada 8 en un **margen** blanco, fuera del tablero, para no
   tapar ni una celda de juego.
3. Pone el **fotograma anterior a la izquierda y el actual a la derecha**, rotulados, para
   que el cambio se vea de un vistazo.

RESOLUCION (corregido tras DESIGN 8.76)
---------------------------------------
El panel unico va a **escala x16 EXACTA, la misma del harness**: asi, al comparar contra
la base, la unica variable son las etiquetas. En 8.76 baje la escala a x10 para meter dos
paneles dentro del presupuesto de pixeles, y eso degradaba justo la capacidad que queria
medir — resolver una celda para leer su coordenada. Un panel: 1.058x1.058 = 1,12 Mpx,
un 6,6% sobre el harness, y ese sobrecoste es solo el margen de rotulos.
Dos paneles (escala x10) siguen disponibles y suman 0,90 Mpx, pero son un experimento
DISTINTO y no deben mezclarse con el de etiquetado.

POR QUE NO CONTRADICE 8.73
--------------------------
8.73 concluyo "dejar de anadir" porque las herramientas artesanales perjudican, y nuestras
siete muestras lo confirman. Esto **no es de esa familia**: no le dice al modelo que pensar
ni le inyecta una conclusion nuestra. **Aumenta la fidelidad de lo que observa**, que es
justo la diferencia con el lider.
"""

from __future__ import annotations

import base64
import io

PASO = 8                 # cada cuantas celdas van rejilla y rotulo
ESCALA_DOBLE = 10        # dos paneles
# x16 EXACTO: es la escala que usa el harness, para que al comparar la unica variable
# sean las etiquetas y no la resolucion. Ese fue el confuso de DESIGN 8.76.
ESCALA_SIMPLE = 16       # un panel
MARGEN = 34              # banda blanca para los rotulos (cabe la fuente grande)
TAM_FUENTE = 22          # la de por defecto es diminuta en un lienzo de 1050 px
REJILLA = (0, 255, 255)  # cian puro: NO esta en la paleta ARC
FONDO = (255, 255, 255)
TINTA = (0, 0, 0)

# La paleta del harness, para que los colores coincidan exactamente.
PALETA = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}


def _fuente():
    """Fuente legible. `load_default(size=)` existe desde Pillow 10.1; si no, la diminuta."""
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=TAM_FUENTE)
    except TypeError:
        return ImageFont.load_default()


def panel(grid, escala: int, titulo: str):
    """Un tablero con rejilla cada PASO celdas y coordenadas en el margen."""
    from PIL import Image, ImageDraw

    filas = len(grid)
    cols = max((len(f) for f in grid), default=0)
    if filas <= 0 or cols <= 0:
        raise ValueError("tablero vacio")

    tablero = Image.new("RGB", (cols, filas), PALETA[0])
    px = tablero.load()
    for r, fila in enumerate(grid):
        for c in range(cols):
            v = fila[c] if c < len(fila) else 0
            try:
                px[c, r] = PALETA.get(int(v), PALETA[0])
            except (TypeError, ValueError):
                px[c, r] = PALETA[0]
    tablero = tablero.resize((cols * escala, filas * escala), Image.Resampling.NEAREST)

    lienzo = Image.new("RGB", (cols * escala + MARGEN, filas * escala + MARGEN), FONDO)
    lienzo.paste(tablero, (MARGEN, MARGEN))
    d = ImageDraw.Draw(lienzo)
    f = _fuente()

    # `min(..., ancho-1)`: sin esto la linea del borde derecho/inferior cae en la
    # coordenada = ancho del lienzo y PIL la recorta, perdiendose (fallo de 8.76).
    x_max, y_max = lienzo.width - 1, lienzo.height - 1
    for c in range(0, cols + 1, PASO):
        x = min(MARGEN + c * escala, x_max)
        d.line([(x, MARGEN), (x, y_max)], fill=REJILLA, width=1)
        if c < cols:
            d.text((x + 3, 3), str(c), fill=TINTA, font=f)
    for r in range(0, filas + 1, PASO):
        y = min(MARGEN + r * escala, y_max)
        d.line([(MARGEN, y), (x_max, y)], fill=REJILLA, width=1)
        if r < filas:
            d.text((3, y + 3), str(r), fill=TINTA, font=f)
    return lienzo


def componer(grid_antes, grid_ahora):
    """Anterior a la izquierda, actual a la derecha. Si no hay anterior, solo el actual."""
    from PIL import Image

    if not grid_antes:
        return panel(grid_ahora, ESCALA_SIMPLE, "")
    a = panel(grid_antes, ESCALA_DOBLE, "")
    b = panel(grid_ahora, ESCALA_DOBLE, "")
    sep = 8
    out = Image.new("RGB", (a.width + sep + b.width, max(a.height, b.height)), FONDO)
    out.paste(a, (0, 0))
    out.paste(b, (a.width + sep, 0))
    return out


def url_datos(imagen) -> str:
    buf = io.BytesIO()
    imagen.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parte_imagen(grid_antes, grid_ahora):
    """La parte `image_url` lista para el mensaje, o None si no se puede componer."""
    if not grid_ahora:
        return None
    try:
        return {"type": "image_url", "image_url": {"url": url_datos(componer(grid_antes, grid_ahora))}}
    except Exception:
        return None

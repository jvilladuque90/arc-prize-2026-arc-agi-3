"""Genera figuras ILUSTRATIVAS para docs/DESIGN.md con grids SINTETICOS.

No usa datos de la competencia (los environment_files estan gitignored por privacidad):
todas las figuras se dibujan sobre un grid 64x64 inventado, para explicar el pipeline de
features y la mecanica sin reproducir juegos reales.

Uso:  python scripts/make_doc_figures.py   ->  docs/img/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arc3.features import connected_components  # noqa: E402
from arc3.llm_prompt import ARC_PALETTE  # noqa: E402

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 10


def synth_frame() -> np.ndarray:
    """Grid 64x64 sintetico: fondo, un 'avatar', un 'boton' pequeno raro, una 'pared'."""
    g = np.zeros((64, 64), dtype=np.int8)          # fondo negro (0)
    g[8:56, 6:58] = 5                               # tablero gris (5)
    g[20:36, 20:44] = 1                             # region azul (1)
    g[46:50, 46:50] = 2                             # boton rojo pequeno (2) -> button_score alto
    g[10:12, 10:30] = 8                             # barra de estado arriba (contador)
    g[27:31, 28:32] = 4                             # avatar amarillo (4)
    return g


def render(grid: np.ndarray) -> Image.Image:
    rgb = ARC_PALETTE[np.clip(grid, 0, 15)]
    return Image.fromarray(rgb, "RGB").resize((64 * SCALE, 64 * SCALE), Image.NEAREST)


def fig_frame(grid: np.ndarray) -> None:
    render(grid).save(OUT / "01_frame.png")


def fig_features(grid: np.ndarray) -> None:
    """Overlay: bounding boxes de objetos + button_score + borde enmascarado del hash."""
    img = render(grid).convert("RGB")
    d = ImageDraw.Draw(img)
    counts = np.bincount(grid.ravel(), minlength=16)
    background = int(counts.argmax())
    total = grid.size
    # borde de 3px que el hash ignora (HUD/contadores)
    b = 3 * SCALE
    d.rectangle([b, b, 64 * SCALE - b, 64 * SCALE - b], outline=(255, 255, 255), width=2)
    for o in connected_components(grid, background):
        y0, x0, y1, x1 = o["bbox"]
        rarity = 1.0 - counts[o["color"]] / total
        area = (y1 - y0 + 1) * (x1 - x0 + 1)
        bscore = (0.5 * rarity + 0.5 * (o["size"] / area)) * (1.0 if o["size"] <= 64 else 0.3)
        col = (0, 255, 0) if bscore > 0.5 else (255, 200, 0)
        d.rectangle([x0 * SCALE, y0 * SCALE, (x1 + 1) * SCALE, (y1 + 1) * SCALE],
                    outline=col, width=2)
        d.text((x0 * SCALE + 2, y0 * SCALE + 2), f"{bscore:.2f}", fill=col)
    img.save(OUT / "02_features.png")


def main() -> int:
    g = synth_frame()
    fig_frame(g)
    fig_features(g)
    print("figuras en", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

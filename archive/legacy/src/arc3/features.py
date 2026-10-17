"""Feature engineering para frames de ARC-AGI-3.

Los frames son grids 64x64 con colores 0..15. Aquí se computan:
  - features por frame (histograma de color, objetos, simetrías, bordes, entropía)
  - features de transición (s, a, s'): píxeles cambiados, bbox del cambio,
    deltas por color y detección de traslación (vector de movimiento)

Todo en numpy puro (sin scipy) para poder correr offline en Kaggle sin deps extra.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional, Sequence

import numpy as np

N_COLORS = 16
GRID = 64
# Desplazamientos máximos a testear al detectar traslación de objetos entre frames.
MAX_SHIFT = 8


def frame_to_grid(frame: Any) -> np.ndarray:
    """Convierte FrameData.frame (lista de grids; puede traer varios por animación)
    al último grid como np.ndarray (64, 64) int8."""
    if frame is None or len(frame) == 0:
        return np.zeros((GRID, GRID), dtype=np.int8)
    last = frame[-1]
    return np.asarray(last, dtype=np.int8)


def connected_components(
    grid: np.ndarray, background: Optional[int] = None
) -> list[dict[str, Any]]:
    """Componentes conexas 4-conectadas de celdas del mismo color (ignora el fondo).

    Devuelve una lista de objetos: color, size, bbox (y0, x0, y1, x1), centroid.
    BFS puro en python: el grid es 64x64, es barato.
    """
    h, w = grid.shape
    if background is None:
        background = int(np.bincount(grid.ravel(), minlength=N_COLORS).argmax())
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or grid[y, x] == background:
                continue
            color = int(grid[y, x])
            q = deque([(y, x)])
            seen[y, x] = True
            cells = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            objects.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": (min(ys), min(xs), max(ys), max(xs)),
                    "centroid": (float(np.mean(ys)), float(np.mean(xs))),
                }
            )
    objects.sort(key=lambda o: -o["size"])
    return objects


def _edge_density(grid: np.ndarray) -> float:
    """Fracción de pares vecinos (4-conn) con colores distintos: mide 'estructura'."""
    dh = grid[:, 1:] != grid[:, :-1]
    dv = grid[1:, :] != grid[:-1, :]
    return float((dh.sum() + dv.sum()) / (dh.size + dv.size))


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def grid_features(grid: np.ndarray, max_objects: int = 8) -> dict[str, Any]:
    """Features escalares de un grid 64x64."""
    counts = np.bincount(grid.ravel(), minlength=N_COLORS)[:N_COLORS]
    background = int(counts.argmax())
    objects = connected_components(grid, background)
    feats: dict[str, Any] = {
        "background": background,
        "n_colors": int((counts > 0).sum()),
        "color_entropy": _entropy(counts),
        "edge_density": _edge_density(grid),
        "sym_h": float((grid == grid[:, ::-1]).mean()),  # simetría izquierda-derecha
        "sym_v": float((grid == grid[::-1, :]).mean()),  # simetría arriba-abajo
        "n_objects": len(objects),
    }
    for c in range(N_COLORS):
        feats[f"color_{c}"] = int(counts[c])
    for i in range(max_objects):
        if i < len(objects):
            o = objects[i]
            y0, x0, y1, x1 = o["bbox"]
            feats[f"obj{i}_color"] = o["color"]
            feats[f"obj{i}_size"] = o["size"]
            feats[f"obj{i}_cy"], feats[f"obj{i}_cx"] = o["centroid"]
            feats[f"obj{i}_h"], feats[f"obj{i}_w"] = y1 - y0 + 1, x1 - x0 + 1
        else:
            feats[f"obj{i}_color"] = -1
            feats[f"obj{i}_size"] = 0
            feats[f"obj{i}_cy"] = feats[f"obj{i}_cx"] = -1.0
            feats[f"obj{i}_h"] = feats[f"obj{i}_w"] = 0
    return feats


def _detect_translation(prev: np.ndarray, nxt: np.ndarray, diff: np.ndarray) -> tuple[int, int, float]:
    """Busca el shift (dy, dx) que mejor explica el cambio como traslación.

    Solo mira la región cambiada: si nxt == shift(prev) sobre esa región, hay
    movimiento de un objeto. Devuelve (dy, dx, score) con score en [0, 1].
    """
    ys, xs = np.nonzero(diff)
    if len(ys) == 0:
        return 0, 0, 0.0
    best = (0, 0, 0.0)
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            if dy == 0 and dx == 0:
                continue
            sy, sx = ys - dy, xs - dx
            ok = (sy >= 0) & (sy < GRID) & (sx >= 0) & (sx < GRID)
            if not ok.any():
                continue
            match = float((nxt[ys[ok], xs[ok]] == prev[sy[ok], sx[ok]]).mean())
            if match > best[2]:
                best = (dy, dx, match)
    return best


def transition_features(prev_grid: np.ndarray, next_grid: np.ndarray) -> dict[str, Any]:
    """Features del cambio entre dos frames consecutivos."""
    diff = prev_grid != next_grid
    n_changed = int(diff.sum())
    feats: dict[str, Any] = {"n_changed": n_changed}
    if n_changed == 0:
        feats.update(
            {"chg_y0": -1, "chg_x0": -1, "chg_h": 0, "chg_w": 0,
             "chg_area_frac": 0.0, "move_dy": 0, "move_dx": 0, "move_score": 0.0,
             "colors_gained": 0, "colors_lost": 0}
        )
        return feats
    ys, xs = np.nonzero(diff)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    feats["chg_y0"], feats["chg_x0"] = int(y0), int(x0)
    feats["chg_h"], feats["chg_w"] = int(y1 - y0 + 1), int(x1 - x0 + 1)
    feats["chg_area_frac"] = float(n_changed / diff.size)
    prev_counts = np.bincount(prev_grid.ravel(), minlength=N_COLORS)[:N_COLORS]
    next_counts = np.bincount(next_grid.ravel(), minlength=N_COLORS)[:N_COLORS]
    delta = next_counts.astype(int) - prev_counts.astype(int)
    feats["colors_gained"] = int((delta > 0).sum())
    feats["colors_lost"] = int((delta < 0).sum())
    dy, dx, score = _detect_translation(prev_grid, next_grid, diff)
    feats["move_dy"], feats["move_dx"], feats["move_score"] = dy, dx, score
    return feats


def action_effect_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resumen por acción a partir de filas de transición: ¿qué acciones 'hacen algo'?

    Cada fila debe traer: action_id, n_changed, level_up (bool), game_over (bool).
    """
    out: list[dict[str, Any]] = []
    by_action: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_action.setdefault(int(r["action_id"]), []).append(r)
    for action_id, rs in sorted(by_action.items()):
        n = len(rs)
        out.append(
            {
                "action_id": action_id,
                "n_uses": n,
                "p_change": float(np.mean([r["n_changed"] > 0 for r in rs])),
                "avg_pixels_changed": float(np.mean([r["n_changed"] for r in rs])),
                "p_level_up": float(np.mean([bool(r.get("level_up")) for r in rs])),
                "p_game_over": float(np.mean([bool(r.get("game_over")) for r in rs])),
            }
        )
    return out

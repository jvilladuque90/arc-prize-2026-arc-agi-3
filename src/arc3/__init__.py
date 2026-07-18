"""arc3: utilidades para ARC-AGI-3 (Kaggle arc-prize-2026-arc-agi-3).

Módulos:
  env      -> descubrimiento y ejecución local de environments (arcengine/arc_agi)
  features -> feature engineering sobre frames 64x64 y transiciones (s, a, s')
  probe    -> política de sondeo que genera el dataset de features por juego
"""

from .features import (
    connected_components,
    grid_features,
    frame_to_grid,
    transition_features,
)

__all__ = [
    "connected_components",
    "grid_features",
    "frame_to_grid",
    "transition_features",
]

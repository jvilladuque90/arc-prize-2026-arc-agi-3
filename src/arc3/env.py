"""Descubrimiento y ejecución local de environments ARC-AGI-3.

Envuelve arc_agi.LocalEnvironmentWrapper para jugar los juegos de
environment_files/ sin API remota (igual que hará el rerun de Kaggle offline).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from arcengine import FrameDataRaw, GameAction

logger = logging.getLogger("arc3")


def discover_environments(env_root: Path) -> list[EnvironmentInfo]:
    """Lista los environments locales a partir de environment_files/<game>/<hash>/metadata.json."""
    infos: list[EnvironmentInfo] = []
    for meta_path in sorted(env_root.glob("*/*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # local_dir del metadata es relativo al root del dataset; usamos el real.
        meta["local_dir"] = str(meta_path.parent)
        infos.append(EnvironmentInfo.model_validate(meta))
    return infos


class LocalGame:
    """Sesión de un juego local: reset/step con FrameDataRaw."""

    def __init__(self, info: EnvironmentInfo, seed: int = 0) -> None:
        self.info = info
        self.env = LocalEnvironmentWrapper(
            environment_info=info,
            logger=logger,
            scorecard_id="local-probe",
            seed=seed,
            save_recording=False,
        )

    def reset(self) -> Optional[FrameDataRaw]:
        return self.env.reset()

    def step(
        self, action: GameAction, x: Optional[int] = None, y: Optional[int] = None
    ) -> Optional[FrameDataRaw]:
        data: dict[str, Any] = {"game_id": self.info.game_id}
        if action.is_complex():
            data["x"] = int(x or 0)
            data["y"] = int(y or 0)
        return self.env.step(action, data=data)

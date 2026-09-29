"""Genera notebooks/sheetu_long.ipynb: el kernel publico `scottlegrand/taaf-flashnext-sheetu12b-0922`
(5,19 en el set oculto, el mejor publico al 2026-09-29) con UNA sola edicion nuestra: el recorte
de la ventana offline de build_animfast_long.py, que solo actua fuera de un rerun real.

QUE ES (plan.md): el stack NVFP4 de keithtyser —la misma base que nuestra v24— mas AGENTFIX, seis
reparaciones del bucle del agente medidas en transcripts de Kaggle. En la version publicada solo
van encendidas F1 IMAGES (el historial deja de reenviar tableros viejos, ~26% del presupuesto de
contexto), F3 MEMORY (parseo tolerante del modelo del mundo y sin borrado en game over) y TIMING.

ATRIBUCION: Scott Le Grand (agentfix, servicio), keithtyser (stack NVFP4), Tufa Labs (duck
harness: Bessis, Cottaar, Pressman, Smit, Tesnar, Viel). Nuestro derivado se publica abierto.

Uso:  python scripts/build_sheetu_long.py [minutos]     (por defecto 60)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_animfast_long as recorte  # noqa: E402

PUBLICO = ROOT / "_tmp_sheetu" / "taaf-flashnext-sheetu12b-0922.ipynb"
recorte.SRC = ROOT / "notebooks" / "sheetu.ipynb"
recorte.OUT = ROOT / "notebooks" / "sheetu_long.ipynb"

if __name__ == "__main__":
    if not recorte.SRC.exists():
        shutil.copyfile(PUBLICO, recorte.SRC)
    sys.exit(recorte.main())

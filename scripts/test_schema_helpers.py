"""Test local (CPU, gratis) de los helpers que v4 inyecta en el sandbox.

Por qué importa: el sandbox del agente corta a 30 s por llamada. Si
`connected_components` tarda demasiado en un grid 64x64 real (Python puro, sin
numpy, bajo SAFE_BUILTINS), el graft schema_helpers ESTORBA en vez de ayudar —
y v4 estaría perdiendo turnos en el rerun oculto. Esto lo mide sin GPU.

Uso: python scripts/test_schema_helpers.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "_tmp_fork_bundle" / "src"
for _repo in ("ARC3-Inference", "tufa-arc-agi-framework/src", "taaf-grafts"):
    sys.path.insert(0, str(BUNDLE / _repo))

# Importar el MÓDULO directo, sin pasar por taaf_grafts/__init__ (que arrastra
# los solvers y con ellos arcengine/vllm). Los helpers son puros por diseño.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sh_standalone", BUNDLE / "taaf-grafts" / "taaf_grafts" / "schema_helpers.py")
_sh = importlib.util.module_from_spec(_spec)
sys.modules["sh_standalone"] = _sh
_spec.loader.exec_module(_sh)
SANDBOX_HELPERS_PRELUDE = _sh.SANDBOX_HELPERS_PRELUDE
connected_components = _sh.connected_components
grid_diff = _sh.grid_diff


def synthetic_frame(seed: int, n_objects: int = 40):
    """64x64 con fondo 0 y objetos rectangulares de colores 1-15."""
    rnd = random.Random(seed)
    grid = [[0] * 64 for _ in range(64)]
    for _ in range(n_objects):
        h, w = rnd.randint(1, 6), rnd.randint(1, 6)
        r0, c0 = rnd.randint(0, 63 - h), rnd.randint(0, 63 - w)
        color = rnd.randint(1, 15)
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                grid[r][c] = color
    return grid


def worst_case_frame():
    """Peor caso para components: cada celda un color distinto (4096 objetos)."""
    return [[(r * 64 + c) % 16 for c in range(64)] for r in range(64)]


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1000, out


def main() -> int:
    print(f"prelude: {len(SANDBOX_HELPERS_PRELUDE)} chars\n")
    fails = []

    a, b = synthetic_frame(1), synthetic_frame(2)
    ms, comps = timed(connected_components, a)
    print(f"connected_components (típico, {len(comps)} objetos): {ms:.1f} ms")
    if ms > 3000:
        fails.append(f"connected_components típico lento: {ms:.0f} ms")

    ms_w, comps_w = timed(connected_components, worst_case_frame())
    print(f"connected_components (peor caso, {len(comps_w)} objetos): {ms_w:.1f} ms")
    if ms_w > 10000:
        fails.append(f"connected_components peor caso >10 s: {ms_w:.0f} ms")

    ms_d, diff = timed(grid_diff, a, b)
    print(f"grid_diff 64x64: {ms_d:.1f} ms, n_cells={diff['n_cells']}, bbox={diff['bbox']}")
    if ms_d > 2000:
        fails.append(f"grid_diff lento: {ms_d:.0f} ms")

    # correctitud: 4-conectividad (diagonal = objetos separados)
    diag = [[1, 0], [0, 1]]
    comps_diag = connected_components(diag, colors=1)
    print(f"4-conectividad (diagonal separada): {len(comps_diag)} objetos "
          f"{'OK' if len(comps_diag) == 2 else 'FALLO'}")
    if len(comps_diag) != 2:
        fails.append("4-conectividad rota")

    # correctitud: grid_diff detecta un solo pixel
    g1 = [[0, 0], [0, 0]]
    g2 = [[0, 0], [0, 5]]
    d = grid_diff(g1, g2)
    ok = d["n_cells"] == 1 and d["bbox"] == [1, 1, 1, 1] and d["by_color"] == {"0->5": 1}
    print(f"grid_diff 1 pixel: {'OK' if ok else 'FALLO ' + str(d)}")
    if not ok:
        fails.append("grid_diff incorrecto")

    print("\n" + ("FALLOS: " + "; ".join(fails) if fails else "TODO OK — helpers aptos para el sandbox de 30 s"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

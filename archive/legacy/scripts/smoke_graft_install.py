"""Smoke test LOCAL (CPU, cero cuota) del install de grafts para el duck v4.

Carga el bundle del fork (descargado en _tmp_fork_bundle), añade sus repos a
sys.path como hace el notebook, desempaqueta benchmark_initial.pkl y corre
composite.install con el flag set candidato. Verifica:
  1. banner TAAF_GRAFTS con los flags esperados armados,
  2. que el solver quedó con analyzer_factory injertado (no stock),
  3. que el prelude de schema_helpers se construyó (si el flag va activo).

Uso:
  python scripts/smoke_graft_install.py                       # flags del v4
  python scripts/smoke_graft_install.py efficiency retry_guard shortcircuit
"""

from __future__ import annotations

import pathlib
import pickle
import sys
from pathlib import Path

# El benchmark se pickleó en Linux; en Windows PosixPath no se puede instanciar.
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "_tmp_fork_bundle"

V4_FLAGS = {"efficiency": True, "retry_guard": True, "shortcircuit": True,
            "schema_helpers": True}


def main() -> int:
    flags = ({f: True for f in sys.argv[1:]} if len(sys.argv) > 1 else V4_FLAGS)

    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src", "taaf-grafts"):
        p = BUNDLE / "src" / repo
        if not p.is_dir():
            print(f"FALTA {p} — descargar el bundle primero"); return 1
        sys.path.insert(0, str(p))

    with open(BUNDLE / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    stock_solver = bm.solver
    stock_factory = getattr(stock_solver, "analyzer_factory", None)
    print(f"benchmark: {bm.label!r}, solver stock: {type(stock_solver).__name__}, "
          f"analyzer_factory stock: {stock_factory!r}")

    from taaf_grafts.composite import install
    install(bm, flags=flags)

    solver = bm.solver
    grafted_factory = getattr(solver, "analyzer_factory", None)
    print(f"solver post-install: {type(solver).__name__}, "
          f"analyzer_factory: {'INJERTADO' if grafted_factory is not stock_factory else 'stock (¡NO cambió!)'}")

    if flags.get("schema_helpers"):
        from taaf_grafts import schema_helpers as sh
        prelude = sh.SANDBOX_HELPERS_PRELUDE
        names = [n for n in ("grid_diff", "connected_components",
                             "action_effect_summary", "recent_history")
                 if prelude and f"def {n}" in prelude]
        print(f"prelude schema_helpers ({len(prelude)} chars): "
              f"{'OK (' + ', '.join(names) + ')' if len(names) == 4 else 'INCOMPLETO: ' + repr(names)}")

    ok = grafted_factory is not stock_factory
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

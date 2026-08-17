"""Verifica que el notebook con --nav quedo bien armado ANTES de gastar GPU."""

import json
import sys
from pathlib import Path

nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
src = "".join("".join(c["source"]) for c in nb["cells"])

checks = {
    "marca de inyeccion": "NAV_HELPERS injected" in src,
    "install de grafts": "taaf_grafts.composite import install" in src,
    "schema_helpers activo": '"schema_helpers": True' in src,
    "ventana 70 min": 'TAAF_OFFLINE_SOFT_MIN", "70"' in src,
    "parche protegido": "[nav_helpers] injection failed" in src,
}
try:
    checks["orden: install < parche < juegos"] = (
        src.index("_graft_install") < src.index("_sh.SANDBOX_HELPERS_PRELUDE")
        < src.index("bm.games =")
    )
except ValueError:
    checks["orden: install < parche < juegos"] = False

for k, v in checks.items():
    print(f"  {'OK ' if v else 'FALLO'} {k}")
raise SystemExit(0 if all(checks.values()) else 1)

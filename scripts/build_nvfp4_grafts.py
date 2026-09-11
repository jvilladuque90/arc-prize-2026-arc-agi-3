"""Genera notebooks/nvfp4_grafts.ipynb: el kernel NVFP4 verbatim + UNA celda nuestra.

Brazo "base de mas puntaje + lo nuestro", una sola variable: los tres injertos de v23
(efficiency, retry_guard, schema_helpers; shortcircuit OFF igual que en v23) montados
sobre el stack de servicio NVFP4+MTP de keithtyser. Kernel de EXPERIMENTO: nunca el
que envia el trigger.

POR QUE LOS INJERTOS VAN EMBEBIDOS Y NO COMO DATASET
----------------------------------------------------
El bundle NVFP4 no trae taaf-grafts. Adjuntar el fork de thtennant para traerlos
colisiona con `_find_bundle_dir()` del notebook: ambos bundles llevan la misma
etiqueta `duck-harness-kaggle` y `rglob` devuelve el primero que encuentra, asi que
el notebook podria elegir el bundle equivocado y el stack NVFP4 no cargaria. Elegir
por etiqueta (el truco de --anim) no sirve porque las etiquetas son identicas.

El paquete taaf_grafts se embebe COMPLETO y byte a byte (los 15 modulos): __init__.py
importa banking_solver en caliente, asi que un subconjunto romperia el import.
`composite.install` restaura stock ante cualquier error; el peor caso es la corrida
NVFP4 plana.

El harness del bundle NVFP4 es byte-identico (inference/) al fork de thtennant contra
el que se escribieron los injertos — verificado con `diff -rq`; y
`scripts/smoke_graft_install.py` prueba el install contra ese mismo arbol.

COMPUERTA: exactamente UNA celda anadida, en el hueco documentado por el autor
("Customization hook ... the safe place for one-off experiments"), y ninguna otra
celda tocada. Cualquier otra diferencia rompe la inferencia "este archivo exacto".
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_grafts.ipynb"
GRAFTS = ROOT / "_tmp_fork_bundle" / "src" / "taaf-grafts" / "taaf_grafts"

# Se inserta DESPUES de esta celda (la del "customization hook", indice 13).
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# INJERTOS taaf_grafts (nuestros) SOBRE EL STACK NVFP4 — kernel de EXPERIMENTO.
# Brazo "base de mas puntaje + lo nuestro", UNA variable: los tres injertos de v23
# (efficiency, retry_guard, schema_helpers). shortcircuit OFF como en v23.
#
# Van EMBEBIDOS (15 modulos, byte a byte, tal como corren en v23) y no como dataset:
# el bundle NVFP4 no trae taaf-grafts, y adjuntar el fork de thtennant colisiona con
# _find_bundle_dir() (misma etiqueta duck-harness-kaggle, rglob devuelve el primero).
# composite.install restaura stock ante cualquier error: el peor caso es NVFP4 plano.
try:
    import base64 as _b64g
    import sys as _sysg
    from pathlib import Path as _PathG
    _root = _PathG(WORKING_DIR) / "taaf_grafts_vendored"
    _pkg = _root / "taaf_grafts"
    _pkg.mkdir(parents=True, exist_ok=True)
    _files = __FILES__
    for _name, _b in _files.items():
        (_pkg / _name).write_bytes(_b64g.b64decode(_b))
    if str(_root) not in _sysg.path:
        _sysg.path.insert(0, str(_root))
    from taaf_grafts.composite import install as _graft_install
    _graft_install(bm, flags={"efficiency": True, "retry_guard": True,
                              "shortcircuit": False, "schema_helpers": True})
    print(f"GRAFTS_VENDORED {len(_files)} modulos en {_pkg}", flush=True)
except Exception as exc:
    print(f"[taaf_grafts] vendored install failed, running stock: "
          f"{type(exc).__name__}: {exc}", flush=True)
'''


def main() -> int:
    if not GRAFTS.is_dir():
        print(f"FALTA {GRAFTS} (descargar el fork de thtennant primero)")
        return 1
    files = {}
    for p in sorted(GRAFTS.glob("*.py")):
        files[p.name] = base64.b64encode(p.read_bytes()).decode("ascii")
    if "composite.py" not in files or "__init__.py" not in files:
        print("el paquete taaf_grafts no esta completo")
        return 1

    cell_src = CELL_TEMPLATE.replace("__FILES__", json.dumps(files, sort_keys=True))
    compile(cell_src, "<celda-injertos>", "exec")  # la celda tiene que compilar

    nb = json.loads(SRC.read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"])
           if c.get("cell_type") == "code" and HOOK_ANCHOR in "".join(c["source"])]
    if len(idx) != 1:
        print(f"esperaba 1 celda con el ancla del hook, hay {len(idx)}")
        return 1
    at = idx[0] + 1
    nb["cells"].insert(at, {"cell_type": "code", "execution_count": None, "metadata": {},
                            "outputs": [], "source": cell_src.splitlines(keepends=True)})
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # ---- compuerta: una celda anadida, ninguna otra tocada ---------------
    orig = json.loads(SRC.read_text(encoding="utf-8"))["cells"]
    new = json.loads(OUT.read_text(encoding="utf-8"))["cells"]
    ok = len(new) == len(orig) + 1
    for i, c in enumerate(orig):
        j = i if i < at else i + 1
        if "".join(c["source"]) != "".join(new[j]["source"]):
            print(f"celda original {i} cambio")
            ok = False
    malas = []
    for i, c in enumerate(new, 1):
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c["source"]), f"<c{i}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {i}: {e}")
    print(f"celda insertada en indice {at} | modulos embebidos {len(files)} | "
          f"tamano notebook {OUT.stat().st_size} bytes")
    print("una sola celda anadida y las demas intactas:", "OK" if ok else "FALLA")
    print("todas las celdas compilan:", "OK" if not malas else malas)
    return 0 if ok and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

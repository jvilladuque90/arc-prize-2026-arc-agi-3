"""Endurece la celda 2 (instalacion de arc-agi desde el wheelhouse de la competencia) de un
notebook NVFP4 de EXPERIMENTO: reintento con espera y stderr visible.

Por que: el 2026-09-14 el kernel arc-agi3-nvfp4-carry-yield-long v1 murio en esa celda
(QUEUED -> ERROR en un minuto, `pip install --no-index ... arc-agi` exit 1) con metadata
identico al kernel que corrio bien dos horas antes. La celda tiraba stdout a DEVNULL y no
capturaba stderr, asi que no quedo NINGUN diagnostico. Con un fallo de arranque de Kaggle
como sospechoso principal, la correccion es: reintentar, y si agota, decir por que.

Solo para kernels de experimento. notebooks/nvfp4.ipynb (el del envio) sigue verbatim.
COMPUERTA: exactamente UNA celda cambiada, y solo por esta sustitucion.

Uso:  python scripts/harden_nvfp4_cell2.py notebooks/nvfp4_carry_yield_long.ipynb
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

VIEJO = '''subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-index",
        "--no-warn-conflicts",
        "--disable-pip-version-check",
        "--find-links",
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
        "arc-agi",
    ],
    stdout=subprocess.DEVNULL,
)'''

NUEVO = '''# --- ENDURECIDO (kernel de experimento): localizar el wheelhouse recorriendo
# /kaggle/input (no fiarse de la ruta fija), reintento y stderr visible. El 2026-09-14
# esta celda murio dos veces con exit 1 porque la competencia no estaba montada en la
# ruta fija; la sonda arc-agi3-probe-mounts confirmo que en un kernel limpio si lo esta.
def _locate_wheelhouse():
    import os as _o
    _fijo = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
    _hits = []
    for _dp, _dn, _fn in _o.walk("/kaggle/input"):
        if _dp.count(_o.sep) - "/kaggle/input".count(_o.sep) > 4:
            _dn[:] = []
            continue
        if "arc_agi_3_wheels" in _dn:
            _hits.append(_o.path.join(_dp, "arc_agi_3_wheels"))
    _elegido = _hits[0] if _hits else _fijo
    print(f"WHEELHOUSE_LOCATE hits={_hits} usado={_elegido} existe={_o.path.isdir(_elegido)}", flush=True)
    return _elegido


def _pip_install_retry(cmd, tries=3, wait_s=20):
    import time as _t
    last = None
    for _i in range(1, tries + 1):
        _r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if _r.returncode == 0:
            print(f"WHEELHOUSE_INSTALL ok (intento {_i})", flush=True)
            return
        last = _r.stderr or ""
        print(f"WHEELHOUSE_INSTALL intento {_i} fallo (rc={_r.returncode}):\\n{last[-1500:]}", flush=True)
        if _i < tries:
            _t.sleep(wait_s)
    _wh = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
    try:
        import os as _o
        print("WHEELHOUSE_DIR", _wh, "existe:", _o.path.isdir(_wh),
              "entradas:", sorted(_o.listdir(_wh))[:12] if _o.path.isdir(_wh) else [], flush=True)
    except Exception as _e:
        print("WHEELHOUSE_DIR no listable:", _e, flush=True)
    raise RuntimeError(f"pip install arc-agi fallo {tries} veces; ultimo stderr:\\n{last[-1500:]}")


_pip_install_retry(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-index",
        "--no-warn-conflicts",
        "--disable-pip-version-check",
        "--find-links",
        _locate_wheelhouse(),
        "arc-agi",
    ]
)'''


def main() -> int:
    path = Path(sys.argv[1])
    nb = json.loads(path.read_text(encoding="utf-8"))
    orig = [ "".join(c["source"]) for c in nb["cells"] ]
    hits = [i for i, s in enumerate(orig) if VIEJO in s]
    if len(hits) != 1:
        print(f"esperaba exactamente 1 celda con el bloque original, hay {len(hits)}")
        return 1
    i = hits[0]
    nuevo_src = orig[i].replace(VIEJO, NUEVO, 1)
    compile(nuevo_src, "<celda2-endurecida>", "exec")
    nb["cells"][i]["source"] = nuevo_src.splitlines(keepends=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # compuerta: una sola celda cambiada, y solo por la sustitucion
    new = ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]
    distintas = [k for k in range(len(orig)) if orig[k] != new[k]]
    ok = len(new) == len(orig) and distintas == [i] and new[i].replace(NUEVO, VIEJO, 1) == orig[i]
    malas = []
    for k, c in enumerate(nb["cells"], 1):
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c["source"]), f"<c{k}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {k}: {e}")
    print(f"{path.name}: celda {i} endurecida | una sola celda cambiada y solo por la sustitucion: "
          f"{'OK' if ok else 'FALLA'} | compilan: {'OK' if not malas else malas}")
    return 0 if ok and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

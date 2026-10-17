"""Genera notebooks/animfast_long.ipynb: el kernel publico `thui-animfast` con UNA sola
edicion nuestra — el recorte de la ventana offline, igual que hicimos con el de keithtyser.

QUE ES Y DE DONDE SALE (DESIGN 8.79)
------------------------------------
Barrido de 110 kernels publicos de la competencia, ordenados por cuanto se apartan de la
base y cruzados con el puntaje. El hallazgo: varios kernels que puntuan alto montan una
combinacion que nosotros **nunca hemos corrido**:

  * el **stack de servicio NVFP4 de keithtyser** — el mismo que nos dio 3,55 — y
  * el **solver del bundle anim de jakobbrggen**, cuyo `HarnessSolver` lleva
    `animation_awareness=True` y `hard_noop_guard=True`.

Nosotros corrimos anim sobre el modelo VIEJO de 27B (v21/v23: 1,15 y 1,59) y NVFP4 sin anim
(v24: 3,55). La casilla que falta es justo esta.

Y encaja con lo unico que nos ha funcionado dos veces: **adoptar una base mejor, no anadirle
cosas**. Siete muestras propias mas la publicacion de Tufa Labs dicen que los injertos
artesanales restan (8.73); esto no es un injerto.

Ademas desbloquea lo que 8.44 dejo aparcado: sin senal consciente de animacion, el guard de
no-ops solo veia el fotograma final y su margen medido era del 2,0% (8.53). Con
`animation_awareness` el agente distingue un fotograma transitorio de uno asentado, que es
otra cosa.

ATRIBUCION: `sahasawatt/thui-animfast-v1`, que a su vez monta el bundle de servicio de
**keithtyser** y el solver del fork `feature/animation-awareness` de **jakobbrggen**, sobre
el duck harness de **Tufa Labs** (Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries
Smit, Michal Tesnar, Stefano Viel). Todo publico y de la propia competencia. Nuestro kernel
derivado se publica igualmente abierto.

LA UNICA EDICION
----------------
El kernel publico corre la ventana entera (32.400 s menos 600 de reserva). En Save & Run eso
se comeria la cuota de G4 de golpe. Se recorta a `TAAF_OFFLINE_SOFT_MIN` minutos, y **solo
fuera de un rerun real**: en el envio el presupuesto queda intacto. Es literalmente la misma
edicion que hicimos sobre el kernel de keithtyser para v24.

Uso:  python scripts/build_animfast_long.py [minutos]     (por defecto 60)
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "animfast.ipynb"
OUT = ROOT / "notebooks" / "animfast_long.ipynb"

ANCLA = '''soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(
    seconds=budget - 600.0
)'''

RECORTE = '''

# --- UNICA EDICION SOBRE EL KERNEL PUBLICO thui-animfast -------------------
# En Save & Run (no rerun) el presupuesto son 9 h y se comeria la cuota de G4
# entera. Se recorta la ventana OFFLINE a TAAF_OFFLINE_SOFT_MIN minutos. En el
# rerun real (TRUE_SUBMISSION) no se toca nada: el presupuesto queda intacto.
# Es la misma edicion que hicimos sobre el kernel de keithtyser para v24.
if not TRUE_SUBMISSION:
    _soft_min = float(os.environ.get("TAAF_OFFLINE_SOFT_MIN", "__MIN__"))
    soft_end = min(
        soft_end,
        datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(minutes=_soft_min),
    )
    print(f"OFFLINE_CUT soft_min={_soft_min} soft_end={soft_end}", flush=True)
# --------------------------------------------------------------------------'''


def main() -> int:
    minutos = sys.argv[1] if len(sys.argv) > 1 else "60"
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"])
           if c.get("cell_type") == "code" and ANCLA in "".join(c["source"])]
    if len(idx) != 1:
        print(f"esperaba 1 celda con el ancla de soft_end, hay {len(idx)}")
        return 1
    i = idx[0]
    s = "".join(nb["cells"][i]["source"])
    if "OFFLINE_CUT" in s:
        print("el recorte ya estaba puesto")
        return 1
    nuevo = s.replace(ANCLA, ANCLA + RECORTE.replace("__MIN__", minutos), 1)
    # la celda del harness tiene `await` de nivel superior, como en el kernel de keithtyser
    compile(nuevo, "<celda-recorte>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    nb["cells"][i]["source"] = nuevo.splitlines(keepends=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # --- compuertas ---
    orig = json.loads(SRC.read_text(encoding="utf-8"))["cells"]
    new = json.loads(OUT.read_text(encoding="utf-8"))["cells"]
    dif = [k for k in range(len(orig)) if "".join(orig[k]["source"]) != "".join(new[k]["source"])]
    una = len(new) == len(orig) and dif == [i]
    # y esa celda solo puede diferir por la insercion
    solo = "".join(new[i]["source"]).replace(RECORTE.replace("__MIN__", minutos), "") == s
    malas = []
    for k, c in enumerate(new, 1):
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c["source"]), f"<c{k}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {k}: {e}")
    print(f"{OUT.name}: ventana offline recortada a {minutos} min | {OUT.stat().st_size} bytes")
    print("  una sola celda cambiada :", "OK" if una else f"FALLA {dif}")
    print("  y solo por la insercion :", "OK" if solo else "FALLA")
    print("  todas las celdas compilan:", "OK" if not malas else malas)
    return 0 if una and solo and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

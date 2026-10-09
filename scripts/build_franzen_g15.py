"""Genera notebooks/franzen_g15.ipynb: Franzen FIEL, pero la demo juega los OTROS 15 juegos publicos.

DIAGNOSTICO, no brazo: la demo de Franzen excluye 15 juegos (bp35 cd82 cn04 dc22 g50t ka59 lf52 ls20 m0r0
s5i5 sk48 sp80 su15 tn36 wa30) y juega 10. Nunca hemos visto su comportamiento en esos 15, que son donde se
decide la diferencia entre la demo (46,3) y el oculto (25,7). Esta corrida da la cobertura (cuantos juegos
puntuan > 0) y los transcripts de los que fallan, para elegir la palanca grande con datos (plan.md, punto 3).

Unicos cambios, ambos en la celda de personalizacion de Franzen:
  * demo_excluded_games = los 10 que Franzen SI juega (se invierte la lista) -> juegan los otros 15.
  * tope por juego 25 min -> 37,5 min: son 15 juegos para 10 plazas; el tope cuenta desde que se crea la sesion
    (tambien envejece a los juegos en cola), asi que 37,5 min dejan ~25 min de plaza efectiva por juego.
No se envia al oculto (el notebook fuera del rerun escribe un parquet ficticio).

Uso:  python scripts/build_franzen_g15.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "franzen_m2.ipynb"
OUT = ROOT / "notebooks" / "franzen_g15.ipynb"

OLD_EXC = ("demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', 'ka59', "
           "'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']")
NEW_EXC = ("demo_excluded_games = [] if TRUE_SUBMISSION else ['ar25', 'ft09', 'lp85', 'r11l', 're86', 'sb26', "
           "'sc25', 'tr87', 'tu93', 'vc33']  # G15: se invierte la lista de Franzen")
OLD_CAP = "bm.solver.max_runtime_s_per_game = 25*60 #532*60 * bm.solver.concurrency // 110"
NEW_CAP = "bm.solver.max_runtime_s_per_game = int(37.5*60)  # G15: 15 juegos para 10 plazas"


def main() -> int:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    hits = [i for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"
            and OLD_EXC in "".join(c["source"])]
    if len(hits) != 1:
        print(f"esperaba 1 celda con la lista de exclusion, hay {len(hits)}")
        return 1
    i = hits[0]
    s = "".join(nb["cells"][i]["source"])
    if s.count(OLD_EXC) != 1 or s.count(OLD_CAP) != 1:
        print("anclas no unicas", s.count(OLD_EXC), s.count(OLD_CAP))
        return 1
    s2 = s.replace(OLD_EXC, NEW_EXC).replace(OLD_CAP, NEW_CAP)
    compile(s2, "<celda-g15>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    nb["cells"][i]["source"] = s2.splitlines(keepends=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    orig = json.loads(SRC.read_text(encoding="utf-8"))["cells"]
    new = json.loads(OUT.read_text(encoding="utf-8"))["cells"]
    dif = [k for k in range(len(orig)) if "".join(orig[k]["source"]) != "".join(new[k]["source"])]
    ok = len(orig) == len(new) and dif == [i]
    print(f"{OUT.name}: celda {i} cambiada | {OUT.stat().st_size} bytes")
    print("una sola celda distinta:", "OK" if ok else f"FALLA {dif}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

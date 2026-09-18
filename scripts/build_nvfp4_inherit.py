"""Genera notebooks/nvfp4_inherit.ipynb: kernel NVFP4 verbatim + UNA celda nuestra con la
herencia entre niveles (src/arc3/level_inherit.py) en la costura C.

BASE: `nvfp4.ipynb`, la del ENVIO. Sin ningun otro injerto.

EL FALLO QUE ARREGLA (medido sobre tres corridas de 60 min): el harness borra seis de sus
siete campos de conocimiento en cada transicion de nivel y conserva solo
`cross_level_notes`. Y el modelo escribe al reves — `World model:` 153-255 veces por
corrida (se borra) y `Cross-level notes:` 0, 0 y 20 (persiste). Cada vez que gana un nivel
se tira entero lo que acababa de aprender, que ademas es justo el contenido del manual del
agente que hizo el 100% del set publico: mecanicas, hipotesis refutadas y autocorrecciones.

Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda anadida en el hueco documentado
por el autor, ninguna otra tocada.

Uso:  python scripts/build_nvfp4_inherit.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_inherit.ipynb"
MOD = ROOT / "src" / "arc3" / "level_inherit.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# HERENCIA ENTRE NIVELES SOBRE EL STACK NVFP4 — kernel de EXPERIMENTO.
# El harness borra seis de sus siete campos de conocimiento en cada transicion de nivel y
# conserva SOLO cross_level_notes (_update_summarized_knowledge_from_step_summary). Y el
# modelo escribe al reves: medido en tres corridas, `World model:` 153-255 veces (se borra)
# frente a `Cross-level notes:` 0, 0 y 20 (persiste). Asi que en el instante en que gana un
# nivel se tira lo que acababa de aprender -- mecanicas, hipotesis refutadas y
# autocorrecciones, que es el contenido del manual del agente que hizo el 100% del publico.
# Este injerto dobla esos campos dentro del que sobrevive, JUSTO ANTES del borrado.
# Cero tokens de salida (el texto ya esta escrito), en el vocabulario del modelo (son sus
# palabras), y una sola vez por nivel completado.
try:
    import base64 as _b64i
    import inference.agent.tool_agent as _tai
    _nsi = {}
    exec(compile(_b64i.b64decode("__B64__").decode("utf-8"), "level_inherit.py", "exec"), _nsi)
    _heredar = _nsi["heredar"]
    _es_transicion = _nsi["es_transicion"]
    _orig_upd = _tai.ToolAgent._update_summarized_knowledge_from_step_summary

    def _upd_with_inherit(self):
        heredado = None
        try:
            sk = getattr(self, "_summarized_knowledge", None)
            if isinstance(sk, dict) and _es_transicion(getattr(self, "_last_step_summary", None)):
                n = int(getattr(self, "_arc3_niveles", 0)) + 1
                setattr(self, "_arc3_niveles", n)
                heredado = _heredar(sk, n)
        except Exception:
            heredado = None
        _orig_upd(self)          # el harness borra sus seis campos
        if heredado:
            try:
                self._summarized_knowledge["cross_level_notes"] = heredado
                print("LEVEL_INHERIT: nivel heredado, %d chars" % len(heredado), flush=True)
            except Exception:
                pass

    _tai.ToolAgent._update_summarized_knowledge_from_step_summary = _upd_with_inherit
    print("LEVEL_INHERIT injected on seam C:", len(_nsi), "symbols", flush=True)
except Exception as exc:
    print("[level_inherit] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-inherit>", "exec")

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
    fiel = base64.b64decode(b64) == MOD.read_bytes()
    print(f"celda insertada en indice {at} | tamano notebook {OUT.stat().st_size} bytes")
    print("una sola celda anadida y las demas intactas:", "OK" if ok else "FALLA")
    print("el modulo embebido se reconstruye exacto:", "OK" if fiel else "FALLA")
    print("todas las celdas compilan:", "OK" if not malas else malas)
    return 0 if ok and fiel and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

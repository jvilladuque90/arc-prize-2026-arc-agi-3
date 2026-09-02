"""Prueba el parche de RANURAS OBLIGATORIAS extrayendolo del notebook generado.

Misma disciplina que test_seam_c.py: se prueba el texto FINAL que correra en el
kernel, no una reimplementacion. Y ademas se prueba contra la FRASE REAL del
tool_agent del dataset desplegado, porque el replace es literal: si la frase
difiere en un caracter, el parche anexa en vez de reemplazar (degradacion
prevista, pero hay que saber en cual de los dos modos va a operar).

Verifica:
  1. el parche compila y se aplica sobre la clase
  2. con un prompt que CONTIENE la frase opcional real -> la reemplaza (y el
     resultado exige las 7 etiquetas, sin la palabra 'optional')
  3. con un prompt sin la frase -> anexa el requisito al final
  4. la frase objetivo existe literalmente en el tool_agent.py del dataset
     (si esta prueba falla tras actualizar el bundle, el parche opera en modo
     anexo: funciona pero conviene actualizar la frase)

Uso: python scripts/test_slots_patch.py [--taaf-src DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_TAAF = Path(r"C:\Users\Usuario\AppData\Local\Temp\claude"
                    r"\c--Users-Usuario-Documents-arc-prize-2026-arc-agi-3"
                    r"\71e400bd-234e-45c5-b12f-ee0542e6665f\scratchpad\taaf")

FRASE_REAL = ("If you include assistant text before a tool call, keep it short and "
              "use it to update the world model. Helpful optional prefixes are "
              "`World model:`, `Goal model:`, `Action model:`, `Recent findings:`, "
              "`Open questions:`, `Plan:`, and `Cross-level notes:`.")
ETIQUETAS = ["World model:", "Goal model:", "Action model:", "Recent findings:",
             "Open questions:", "Plan:", "Cross-level notes:"]


class _StubAgent:
    def _build_user_prompt(self, action_num, **kw):
        return kw.get("_base", "PROMPT DEL PADRE")


def extract_patch(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "SLOTS_MANDATORY injected" in src:
            start = src.index("_SLOTS_OLD = (")
            end = src.index("[slots] injection failed")
            end = src.index("\n", src.index("\n", end) + 1) + 1
            return src[start:end]
    raise SystemExit("no encontre el parche de slots en el notebook")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="notebooks/duck_slots.ipynb")
    ap.add_argument("--taaf-src", default=str(SCRATCH_TAAF))
    args = ap.parse_args()

    patch = extract_patch(ROOT / args.notebook)
    print(f"parche extraido: {len(patch)} chars")

    pkg = types.ModuleType("taaf_grafts")
    mod = types.ModuleType("taaf_grafts.schema_helpers")
    mod.SchemaHelpersToolAgent = _StubAgent
    pkg.schema_helpers = mod
    sys.modules["taaf_grafts"] = pkg
    sys.modules["taaf_grafts.schema_helpers"] = mod

    ns: dict = {}
    exec(compile(patch, "<slots_patch>", "exec"), ns)
    if _StubAgent._build_user_prompt.__name__ != "_bup_with_slots":
        print("FALLO 1: el parche no reemplazo _build_user_prompt")
        return 1
    print("1. parche aplicado sobre la clase")

    agent = _StubAgent()

    # 2. prompt con la frase real -> reemplazo
    base = f"CABECERA\n{FRASE_REAL}\nCOLA"
    out = agent._build_user_prompt(0, _base=base)
    faltan = [e for e in ETIQUETAS if f"`{e}`" not in out]
    if FRASE_REAL in out:
        print("FALLO 2: la frase opcional sigue presente")
        return 1
    if "optional" in out.lower():
        print("FALLO 2: quedo la palabra 'optional' en el prompt")
        return 1
    if faltan or "REQUIRED FORMAT" not in out or not out.endswith("COLA"):
        print(f"FALLO 2: requisito mal formado; faltan {faltan}")
        return 1
    print("2. frase opcional reemplazada por el formato requerido (7/7 etiquetas)")

    # 3. prompt sin la frase -> anexo al final
    out = agent._build_user_prompt(0, _base="PROMPT SIN LA FRASE")
    if not out.startswith("PROMPT SIN LA FRASE") or "REQUIRED FORMAT" not in out:
        print("FALLO 3: sin frase objetivo deberia anexar el requisito")
        return 1
    print("3. sin frase objetivo: anexa el requisito (modo degradado)")

    # 4. la frase existe en el tool_agent del dataset desplegado
    ta = Path(args.taaf_src) / "src/ARC3-Inference/inference/agent/tool_agent.py"
    if ta.exists():
        modo = "REEMPLAZO" if FRASE_REAL in ta.read_text(encoding="utf-8") else "ANEXO"
        print(f"4. contra el dataset real: el parche operara en modo {modo}")
        if modo == "ANEXO":
            print("   (funciona igual, pero conviene actualizar _SLOTS_OLD)")
    else:
        print("4. (dataset no disponible en local; se omite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Genera notebooks/nvfp4_vision.ipynb: kernel NVFP4 verbatim + UNA celda nuestra con la
imagen etiquetada (src/arc3/vision_labels.py) en la costura C.

BASE: `nvfp4.ipynb`, la del ENVIO.

EL HUECO (DESIGN 8.75): el modelo recibe imagen —3.687 consultas a la cache multimodal lo
confirman— pero es un mapa de colores plano, sin etiquetas, sin rejilla, sin coordenadas y
solo el fotograma actual. Y se le pide `MOUSE(row=X, col=Y)`: contar pixeles en 1024x1024
sin referencia. Tufa Labs, primero con 18,81, usa *labeled images* y varios fotogramas.
Llevamos nueve brazos escribiendo texto al final del prompt y CERO tocando la imagen.

NO contradice 8.73 ("dejar de anadir"): no inyecta consejo nuestro, aumenta la fidelidad de
lo que el modelo observa.

Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda anadida en el hueco documentado
por el autor, ninguna otra tocada.

Uso:  python scripts/build_nvfp4_vision.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_vision.ipynb"
MOD = ROOT / "src" / "arc3" / "vision_labels.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# IMAGEN ETIQUETADA SOBRE EL STACK NVFP4 — kernel de EXPERIMENTO.
# El modelo SI recibe imagen del tablero (3.687 consultas a la cache multimodal en nuestras
# corridas), pero el harness la renderiza como un mapa de colores plano: un pixel por celda
# escalado x16 a 1024x1024, SIN etiquetas, SIN rejilla, SIN coordenadas y solo el fotograma
# actual. Y se le pide emitir MOUSE(row=X, col=Y): tiene que contar pixeles sin referencia.
# Tufa Labs, primero de la tabla con 18,81, hace lo contrario: "renders recent frames as
# LABELED IMAGES". Llevamos nueve brazos escribiendo texto al final del prompt y cero
# tocando la imagen. Esto NO es de la familia que 8.73 desaconseja: no inyecta consejo
# nuestro, aumenta la fidelidad de la observacion. Y cuesta MENOS pixeles: 0,89 Mpx contra
# los 1,05 que manda el harness.
try:
    import base64 as _b64v
    import inference.agent.tool_agent as _tav
    _nsv = {}
    exec(compile(_b64v.b64decode("__B64__").decode("utf-8"), "vision_labels.py", "exec"), _nsv)
    _parte_imagen = _nsv["parte_imagen"]
    _orig_bum = _tav.ToolAgent._build_user_message
    _LEYENDA = (
        "\\n\\nBoard image: LEFT panel is the PREVIOUS frame, RIGHT panel is the CURRENT frame "
        "(a single panel means there is no previous frame yet). The cyan lines and the numbers "
        "along the top and left edges are row and column indices every 8 cells: read MOUSE "
        "row/col directly off them."
    )

    def _bum_labeled(self, user_prompt, current_frame):
        try:
            actual = getattr(current_frame, "grid", None)
            if not actual:
                return _orig_bum(self, user_prompt, current_frame)
            sesion = getattr(self, "_session_runtime_dir", None)
            previo = None
            if getattr(self, "_arc3_vis_sesion", None) == sesion:
                previo = getattr(self, "_arc3_vis_prev", None)
            parte = _parte_imagen(previo, actual)
            self._arc3_vis_sesion = sesion
            self._arc3_vis_prev = [list(f) for f in actual]
            if parte is None:
                return _orig_bum(self, user_prompt, current_frame)
            return {"role": "user", "content": [
                {"type": "text", "text": user_prompt + _LEYENDA},
                parte,
            ]}
        except Exception:
            return _orig_bum(self, user_prompt, current_frame)

    _tav.ToolAgent._build_user_message = _bum_labeled
    print("VISION_LABELS injected on seam C:", len(_nsv), "symbols", flush=True)
except Exception as exc:
    print("[vision_labels] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-vision>", "exec")

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

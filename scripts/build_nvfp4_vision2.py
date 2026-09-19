"""Genera notebooks/nvfp4_vision2.ipynb: kernel NVFP4 verbatim + UNA celda nuestra con la
imagen etiquetada, esta vez a UNA SOLA VARIABLE (src/arc3/vision_labels.py).

CORRIGE EL CONFUSO DE 8.76. Aquel brazo cambio dos cosas: etiquetas Y dos paneles, y para
que los dos paneles cupieran en el presupuesto de pixeles baje la escala de x16 a x10 --
degradando justo la capacidad que queria medir (resolver una celda para leer su
coordenada). No se podia separar "etiquetar no sirve" de "le baje la resolucion".

Aqui: UN panel, a **escala x16 EXACTA, la del harness**. La unica diferencia con la base
son la rejilla cian cada 8 celdas y los rotulos de fila/columna en el margen. El fotograma
anterior es un experimento distinto y no entra: el modelo ya tiene `previous_frame` en
Python y el harness le dice explicitamente que los compare.

Tambien corrige dos defectos de implementacion encontrados en la auditoria: la fuente por
defecto de PIL era diminuta en un lienzo de 1.050 px (ahora 22 px) y las lineas del borde
derecho e inferior caian fuera del lienzo y se perdian.

Kernel de EXPERIMENTO. COMPUERTA: exactamente una celda anadida en el hueco documentado
por el autor, ninguna otra tocada.

Uso:  python scripts/build_nvfp4_vision2.py
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_vision2.ipynb"
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
    import base64 as _b64v2
    import inference.agent.tool_agent as _tav2
    _nsv2 = {}
    exec(compile(_b64v2.b64decode("__B64__").decode("utf-8"), "vision_labels.py", "exec"), _nsv2)
    _parte_imagen = _nsv2["parte_imagen"]
    _orig_bum2 = _tav2.ToolAgent._build_user_message
    _LEYENDA = (
        "\\n\\nBoard image: LEFT panel is the PREVIOUS frame, RIGHT panel is the CURRENT frame "
        "(a single panel means there is no previous frame yet). The cyan lines and the numbers "
        "along the top and left edges are row and column indices every 8 cells: read MOUSE "
        "row/col directly off them."
    )

    def _bum_labeled2(self, user_prompt, current_frame):
        try:
            actual = getattr(current_frame, "grid", None)
            if not actual:
                return _orig_bum2(self, user_prompt, current_frame)
            # UN SOLO PANEL, a escala x16 EXACTA como el harness: la unica variable
            # frente a la base son las etiquetas. En 8.76 mandamos dos paneles y para
            # ello bajamos la escala a x10, lo que metia un confuso.
            parte = _parte_imagen(None, actual)
            if parte is None:
                return _orig_bum2(self, user_prompt, current_frame)
            return {"role": "user", "content": [
                {"type": "text", "text": user_prompt + _LEYENDA},
                parte,
            ]}
        except Exception:
            return _orig_bum2(self, user_prompt, current_frame)

    _tav2.ToolAgent._build_user_message = _bum_labeled2
    print("VISION_LABELS2 injected on seam C:", len(_nsv2), "symbols", flush=True)
except Exception as exc:
    print("[vision_labels] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-vision2>", "exec")

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

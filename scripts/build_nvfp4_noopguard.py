"""Genera notebooks/nvfp4_carry_noopguard_long.ipynb: consolidacion (60 min) + UNA celda con
el guard de no-ops del anfitrion (src/arc3/host_noop_guard.py). Cognicion sin texto:
cambia que acciones se ejecutan, no que lee el modelo.

Montaje: ToolAgent.analyze() recibe el step_env del solver y lo guarda en
self._step_env_callback; se parchea analyze() a nivel de clase para envolver ese
step_env con el guard (uno por sesion/juego, creado en _ensure_session). Las
estadisticas de bloqueo se escriben en WORKING_DIR/noop_guard_blocks.jsonl (el log del
kernel vuelve vacio en estos kernels; ese archivo si viaja en la salida).
"""

from __future__ import annotations

import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "nvfp4_carry_long.ipynb"
OUT = ROOT / "notebooks" / "nvfp4_carry_noopguard_long.ipynb"
MOD = ROOT / "src" / "arc3" / "host_noop_guard.py"
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL_TEMPLATE = '''# GUARD DE NO-OPS DEL ANFITRION (nuestro) SOBRE EL STACK NVFP4 — kernel de EXPERIMENTO.
# Bloquea, antes de llegar al entorno, repetir una accion ya probada inerte sobre el
# MISMO tablero del mismo nivel; el modelo recibe el mismo resultado que anim devuelve
# (stop_reason known_noop, "no action budget spent"). Cero tokens de prompt. Logica de
# NoopGuard tomada del fork anim de jakobbrggen (atribuido en el modulo); el montaje
# (envolver el step_env que analyze() recibe) es nuestro. Degrada a stock ante error.
try:
    import base64 as _b64q
    import inference.agent.tool_agent as _taq
    from inference.agent.runtime_state import load_runtime_state as _lrsq
    _nsq = {}
    exec(compile(_b64q.b64decode("__B64__").decode("utf-8"), "host_noop_guard.py", "exec"), _nsq)
    _NoopGuardQ = _nsq["NoopGuard"]
    _mk_guarded = _nsq["make_guarded_step_env"]
    _StatsQ = _nsq["GuardStats"]
    # nombres de motor -> nombres de modelo, si el harness los expone
    try:
        _tma = getattr(_taq, "to_model_action", None)
        if _tma is not None:
            _nsq["set_display_map"]({f"ACTION{i}": _tma(f"ACTION{i}") for i in range(1, 8)})
    except Exception:
        pass
    _statsq = _StatsQ(str(WORKING_DIR / "noop_guard_blocks.jsonl"))
    _orig_ensure_q = _taq.ToolAgent._ensure_session
    _orig_analyze_q = _taq.ToolAgent.analyze

    def _ensure_with_guard(self, state_path):
        _prev = getattr(self, "_session_runtime_dir", None)
        _orig_ensure_q(self, state_path)
        if getattr(self, "_host_noop_guard", None) is None or _prev != getattr(self, "_session_runtime_dir", None):
            self._host_noop_guard = _NoopGuardQ()

    def _analyze_with_guard(self, state_path, action_num, *args, step_env=None, **kw):
        try:
            if step_env is not None:
                self._ensure_session(state_path)
                _g = getattr(self, "_host_noop_guard", None)
                if _g is None:
                    _g = _NoopGuardQ()
                    self._host_noop_guard = _g
                step_env = _mk_guarded(step_env, state_path, _g, _lrsq, _statsq,
                                       lambda: list(getattr(self, "_current_valid_actions", []) or []),
                                       lambda: getattr(self, "_last_action_result", None),
                                       str(getattr(state_path, "stem", ""))[:13])
        except Exception:
            pass
        return _orig_analyze_q(self, state_path, action_num, *args, step_env=step_env, **kw)

    _taq.ToolAgent._ensure_session = _ensure_with_guard
    _taq.ToolAgent.analyze = _analyze_with_guard
    print("HOST_NOOP_GUARD injected:", len(_nsq), "symbols; stats ->", _statsq.path, flush=True)
except Exception as exc:
    print("[host_noop_guard] injection failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    b64 = base64.b64encode(MOD.read_bytes()).decode("ascii")
    cell_src = CELL_TEMPLATE.replace("__B64__", b64)
    compile(cell_src, "<celda-noopguard>", "exec")

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
    print(f"{OUT.name}: celda noopguard en indice {at} | una sola celda anadida y las demas intactas: "
          f"{'OK' if ok else 'FALLA'} | compilan: {'OK' if not malas else malas}")
    return 0 if ok and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

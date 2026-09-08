"""Compuerta CPU de la migracion al bundle anim (docs/AUDIT_2026-09-08.md).

Dos familias de comprobaciones, todas locales y gratuitas:

A. INVARIANTES DEL NOTEBOOK (notebooks/duck_anim.ipynb)
   Que lo generado sea de verdad "injertos nuestros sobre el harness de ellos":
   bundle anim adjunto y elegido de forma determinista, injertos montados desde
   el fork SIN sus copias del harness, shortcircuit apagado, guard de log
   presente, y ningun parche nuestro que razone sobre el frame final.

B. COMPATIBILIDAD DE COSTURAS (arbol del fork  vs  arbol de anim)
   Los injertos se escribieron contra el harness del fork: subclasean ToolAgent,
   copian verbatim el cuerpo de HarnessSolver._play_one y duck-tipan payloads.
   Cada costura de la que dependen se compara entre los dos arboles. Una costura
   que cambio no es "quizas rompe": es rompe.

Uso:
  python scripts/verify_anim_compat.py
  python scripts/verify_anim_compat.py --fork _tmp_fork_bundle --anim _tmp_anim
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ANIM_DATASET = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
GRAFTS_DATASET = "thtennant/taaf-kaggle-source-share-fork"

_FAILURES: list[str] = []
_CHECKS = 0


def check(ok: bool, name: str, detail: str = "") -> bool:
    global _CHECKS
    _CHECKS += 1
    print(("  ok   " if ok else "  FALLA") + f"  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)
    return ok


# --------------------------------------------------------------- utilidades

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def func_source(path: Path, name: str, *, cls: str | None = None) -> str | None:
    """Fuente exacta de una funcion/metodo, por AST (robusto a reindentado)."""
    tree = ast.parse(read(path))
    lines = read(path).splitlines()

    def find(nodes: list[ast.AST]) -> ast.AST | None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        return None

    if cls is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                found = find(node.body)
                if found is not None:
                    return "\n".join(lines[found.lineno - 1:found.end_lineno])
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    return None


def signature(path: Path, name: str, *, cls: str | None = None) -> str | None:
    """Firma normalizada (nombres de parametros en orden) de una funcion."""
    src = func_source(path, name, cls=cls)
    if src is None:
        return None
    node = ast.parse(ast.unparse(ast.parse(src.strip().replace("\n    ", "\n"))))
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            a = item.args
            names = [p.arg for p in a.posonlyargs + a.args]
            if a.vararg:
                names.append("*" + a.vararg.arg)
            names += [p.arg for p in a.kwonlyargs]
            if a.kwarg:
                names.append("**" + a.kwarg.arg)
            return ",".join(names)
    return None


def module_assign(path: Path, name: str) -> bool:
    tree = ast.parse(read(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return True
    return False


# --------------------------------------------- A. invariantes del notebook

def verify_notebook(nb_path: Path) -> None:
    print(f"\nA. INVARIANTES DEL NOTEBOOK  ({nb_path.name})")
    if not nb_path.exists():
        check(False, "el notebook existe", str(nb_path))
        return
    nb = json.loads(read(nb_path))
    cells = ["".join(c["source"]) for c in nb["cells"]]
    src = "".join(cells)

    # A1 — todas las celdas compilan (con await de nivel superior, como Jupyter)
    bad = []
    for i, cell in enumerate(cells, start=1):
        try:
            compile(cell, f"<celda {i}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as exc:
            bad.append(f"celda {i}: {exc}")
    check(not bad, "todas las celdas compilan", "; ".join(bad))

    # A2 — los dos datasets, con el bundle anim en la posicion 0
    check(src.count(f'DATASET_SOURCES = ["{ANIM_DATASET}"') == 1,
          "el bundle anim es DATASET_SOURCES[0]")
    check(src.count(f'"{GRAFTS_DATASET}"') == 1,
          "el fork de los injertos esta adjunto (una sola vez)")

    # A3 — con dos bundles adjuntos, la eleccion no puede depender del orden
    check('if "anim" in _label' in src,
          "el bundle se elige por benchmark_label, no por orden del rglob")

    # A4 — de ese fork entra SOLO la raiz de los injertos
    check("taaf-grafts/taaf_grafts/composite.py" in src,
          "los injertos se montan desde taaf-grafts")
    check("raiz de injertos demasiado ancha" in src,
          "hay assert contra montar de mas del fork")
    for shadow in ("ARC3-Inference", "tufa-arc-agi-framework"):
        check(f'sys.path.insert(0, "{shadow}' not in src,
              f"el fork NO aporta {shadow} al sys.path")

    # A5 — flags de injerto
    check('"shortcircuit": False' in src and '"shortcircuit": True' not in src,
          "shortcircuit APAGADO",
          "borraria frame_count/animation del payload (ver B5)")
    for flag in ("efficiency", "retry_guard", "schema_helpers"):
        check(f'"{flag}": True' in src, f"{flag} encendido")

    # A6 — el guard que distingue "adjunto" de "encendido"
    check("ANIM_BASE" in src, "banner ANIM_BASE para la compuerta del save&run")
    for field in ("hard_noop_guard", "animation_awareness"):
        check(f'setattr(bm.solver, _flag, True)' in src and field in src,
              f"{field} afirmado explicitamente en el solver")
    check("import inference.utils.animation" in src,
          "el guard prueba que animation.py es importable de verdad")

    # A7 — nada nuestro que razone sobre el frame final (auditoria 2.4)
    for name, needle in (("nav", "_nav_shift"), ("effects", "action_effect_table"),
                         ("objects", "OBJECT MAP")):
        check(needle not in src, f"sin parche {name} (razona sobre el frame final)")


# ------------------------------------------ B. compatibilidad de costuras

def verify_seams(fork: Path, anim: Path) -> None:
    print("\nB. COMPATIBILIDAD DE COSTURAS  (fork -> anim)")
    fi = fork / "src" / "ARC3-Inference" / "inference"
    ai = anim / "src" / "ARC3-Inference" / "inference"
    if not fi.exists() or not ai.exists():
        check(False, "los dos arboles de fuente estan disponibles",
              f"{fi.exists()=} {ai.exists()=}")
        return

    f_solver, a_solver = fi / "framework" / "solver.py", ai / "framework" / "solver.py"
    f_agent, a_agent = fi / "agent" / "tool_agent.py", ai / "agent" / "tool_agent.py"
    f_box, a_box = fi / "agent" / "python_tool_sandbox.py", ai / "agent" / "python_tool_sandbox.py"

    # B1 — SessionSeamMixin lleva una copia VERBATIM de _play_one. Si anim lo
    #      cambio, montar cualquier injerto de sesion revierte ese cambio en
    #      silencio.
    fp, ap = func_source(f_solver, "_play_one", cls="HarnessSolver"), \
             func_source(a_solver, "_play_one", cls="HarnessSolver")
    check(fp is not None and fp == ap,
          "HarnessSolver._play_one identico",
          "la copia verbatim de solver_base.SessionSeamMixin sigue siendo valida")

    # B2 — las subclases de ToolAgent (efficiency, schema_helpers) llaman a
    #      super() con estas firmas exactas.
    for name, cls in (("_build_user_prompt", "ToolAgent"),
                      ("_run_python_tool", "ToolAgent"),
                      ("analyze", "ToolAgent")):
        fs, as_ = signature(f_agent, name, cls=cls), signature(a_agent, name, cls=cls)
        check(fs is not None and fs == as_, f"firma de ToolAgent.{name} identica",
              "" if fs == as_ else f"{fs}  !=  {as_}")

    # B3 — la fabrica de analizadores de los injertos construye ToolAgent con
    #      un subconjunto de kwargs; anim solo puede haber ANADIDO opcionales.
    f_init = set((signature(f_agent, "__init__", cls="ToolAgent") or "").split(","))
    a_init = set((signature(a_agent, "__init__", cls="ToolAgent") or "").split(","))
    check(f_init and f_init <= a_init,
          "ToolAgent.__init__ solo gano parametros",
          "nuevos: " + ", ".join(sorted(a_init - f_init)))

    # B4 — el injerto context_window reasigna este global de modulo.
    check(module_assign(a_agent, "_LOCAL_ANALYZER_CONTEXT_WINDOW"),
          "_LOCAL_ANALYZER_CONTEXT_WINDOW sigue siendo global de modulo")

    # B5 — LA INCOMPATIBILIDAD. anim ensambla frame_count y animation en el
    #      payload final de step_env; la copia verbatim del injerto shortcircuit
    #      es anterior a esas lineas, asi que en un lote homogeneo >=2 las
    #      borraria y _action_animated() leeria "no animada" -> el guard duro
    #      bloquearia una accion que SI funciono.
    a_step = func_source(a_solver, "step_env", cls="_HarnessGameSession") or ""
    f_step = func_source(f_solver, "step_env", cls="_HarnessGameSession") or ""
    anim_only = ('final_payload["frame_count"]' in a_step
                 and "pick_animation" in a_step
                 and 'final_payload["frame_count"]' not in f_step)
    check(anim_only, "anim ensambla frame_count/animation en step_env",
          "por eso shortcircuit debe ir apagado")
    sc = fork / "src" / "taaf-grafts" / "taaf_grafts" / "shortcircuit_solver.py"
    if sc.exists():
        sc_step = func_source(sc, "step_env", cls="ShortCircuitSessionMixin") or ""
        check("frame_count" not in sc_step,
              "el injerto shortcircuit NO copia frame_count",
              "confirma que apagarlo es obligatorio, no una precaucion")

    # B6 — la consulta de animacion llega por step_env con query='animation'.
    #      Un injerto de sesion debe dejarla pasar a super().
    check("query" in a_step and '"animation"' in a_step,
          "la consulta animation viaja por step_env (query='animation')")

    # B7 — anim pasa las dos palancas a cada ToolAgent; nuestras fabricas no las
    #      pasan, asi que el default de modulo tiene que ser True.
    a_txt = read(a_agent)
    for const, envvar in (("_HARD_NOOP_GUARD_ENABLED", "ARC3_HARD_NOOP_GUARD"),
                          ("_ANIMATION_AWARENESS_ENABLED", "ARC3_ANIMATION_AWARENESS")):
        line = next((l for l in a_txt.splitlines()
                     if l.startswith(const + " =")), "")
        check(envvar in line and line.rstrip().endswith("True)"),
              f"{const} por defecto True",
              "nuestras fabricas de analizador omiten el kwarg")

    # B8 — schema_helpers inyecta su prelude en el mismo sandbox restringido.
    #      SAFE_BUILTINS vive DENTRO del texto del programa hijo (una cadena),
    #      asi que no se puede leer por AST: se compara el bloque literal.
    def safe_builtins_block(path: Path) -> str:
        out, taking = [], False
        for line in read(path).splitlines():
            if line.strip().startswith("SAFE_BUILTINS = {"):
                taking = True
            if taking:
                out.append(line)
                if line.strip() == "}":
                    break
        return "\n".join(out)

    f_safe, a_safe = safe_builtins_block(f_box), safe_builtins_block(a_box)
    check(bool(f_safe) and f_safe == a_safe,
          "SAFE_BUILTINS del sandbox sin cambios",
          "el prelude de schema_helpers compila igual")

    # B9 — la sesion lee contadores nuevos del analizador; RetryGuard los
    #      proxya por __getattr__, pero solo si sigue leyendose con getattr().
    a_sess = read(a_solver)
    check('getattr(self.analyzer, "animation_counters"' in a_sess,
          "animation_counters se lee con getattr (RetryGuard lo proxya)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=str(ROOT / "notebooks" / "duck_anim.ipynb"))
    ap.add_argument("--fork", default=str(ROOT / "_tmp_fork_bundle"))
    ap.add_argument("--anim", default=str(ROOT / "_tmp_anim"))
    args = ap.parse_args()

    print("COMPUERTA DE MIGRACION AL BUNDLE ANIM")
    verify_notebook(Path(args.notebook))
    verify_seams(Path(args.fork), Path(args.anim))

    print(f"\n{_CHECKS - len(_FAILURES)}/{_CHECKS} comprobaciones pasan")
    if _FAILURES:
        print("FALLAN: " + "; ".join(_FAILURES))
        return 1
    print("VERDE: la base anim y los injertos son compatibles como estan montados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

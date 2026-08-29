"""Genera notebooks/duck.ipynb: baseline LLM (Fase 3) sobre la G4.

Reproduce el harness público del ganador del milestone (Tufa Labs) para establecer el
baseline VLM-local a batir. Adjunta los 3 datasets públicos (bundle del solver + wheels
vLLM + pesos Qwen3-27B-FP8) y corre en la RTX Pro 6000.

IMPORTANTE (cuota G4): en Save & Run (no rerun) el harness juega los environment_files
offline con un soft-deadline. Aquí lo recortamos con TAAF_OFFLINE_SOFT_MIN para una
validación corta (confirmar que vLLM arranca y juega), NO un run completo de 9h.

Uso:  python scripts/build_duck_notebook.py
Basado en: notebooks pull de jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "duck.ipynb"

# Celda de setup: idéntica en espíritu al launcher TAAF, con corte de validación corto.
CELLS = [
    '''\
# Fase 3 baseline: duck harness (Tufa Labs) en la G4 — validación offline CORTA.
import json, os, pickle, subprocess, sys, sysconfig, time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1","true"}
NOTEBOOK_START = time.time()
os.environ["MPLBACKEND"] = "Agg"
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
os.environ["ONLY_RESET_LEVELS"] = "true"
# CUDA linker path para vLLM/torch en imagen Kaggle
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    e for e in ["/usr/local/nvidia/lib64", os.environ.get("LIBRARY_PATH","")] if e)
# Corte de validación offline (minutos) para NO gastar 9h de G4 cuando no es rerun.
OFFLINE_SOFT_MIN = float(os.environ.get("TAAF_OFFLINE_SOFT_MIN", "25"))
WORKING = Path("/kaggle/working"); WORKING.mkdir(parents=True, exist_ok=True)
print("TRUE_SUBMISSION =", TRUE_SUBMISSION)
''',
    '''\
# Instalar arc-agi del wheelhouse de la competencia (offline)
COMP_ROOT = None
for dp, dn, _ in os.walk("/kaggle/input"):
    if "arc_agi_3_wheels" in dn:
        COMP_ROOT = Path(dp); break
assert COMP_ROOT, "wheelhouse no encontrado"
subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--no-index",
    "--no-warn-conflicts","--disable-pip-version-check",
    f"--find-links={COMP_ROOT/'arc_agi_3_wheels'}","arc-agi"], stdout=subprocess.DEVNULL)
import arc_agi; print("arc_agi OK")

# Localizar el bundle del solver por su marker
BUNDLE = None
for m in Path("/kaggle/input").rglob("taaf-kaggle-bundle.json"):
    BUNDLE = m.parent; break
assert BUNDLE, "bundle TAAF no encontrado (adjunta thtennant/taaf-kaggle-source-share-fork)"
print("BUNDLE =", BUNDLE)

# Mapear datasets adjuntos a sus mounts
DATASET_SOURCES = ["thtennant/taaf-kaggle-source-share-fork",
                   "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                   "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]
def mount(ref):
    o,s = ref.split("/",1)
    for c in (Path("/kaggle/input")/s, Path("/kaggle/input/datasets")/o/s):
        if c.exists(): return str(c)
    return str(Path("/kaggle/input")/s)
paths = {r: (str(BUNDLE) if i==0 else mount(r)) for i,r in enumerate(DATASET_SOURCES)}
env_extra = {"TAAF_KAGGLE_INPUT_PATHS": json.dumps(paths, sort_keys=True),
             "TAAF_KAGGLE_DATASET_SOURCES": json.dumps(DATASET_SOURCES),
             "TAAF_KAGGLE_KERNEL_SOURCES": json.dumps([])}
os.environ.update(env_extra)
SETUP_ENV = WORKING/"taaf_setup_env.json"; SETUP_ENV.write_text(json.dumps(env_extra))
print(paths)
''',
    '''\
# Importar repos del bundle y correr setup_commands (instala vLLM, arranca el server)
def source_entries(b):
    out=[]
    for repo in sorted((b/"src").iterdir(), reverse=True):
        for c in (repo/"src", repo):
            if c.is_dir(): out.append(c)
    return out
entries = source_entries(BUNDLE)
for e in entries: sys.path.insert(0, str(e))
pth = Path(sysconfig.get_paths()["purelib"])/"taaf_sources.pth"
pth.write_text("".join(f"{e}\\n" for e in entries))

def cmd_env():
    env = os.environ.copy(); env["PYTHON"]=sys.executable
    env["TAAF_KAGGLE_BUNDLE_DIR"]=str(BUNDLE); env["TAAF_KAGGLE_WORKING_DIR"]=str(WORKING)
    env["TAAF_KAGGLE_SETUP_ENV"]=str(SETUP_ENV)
    env.update({str(k):str(v) for k,v in json.loads(SETUP_ENV.read_text()).items()})
    return env
env = cmd_env()
for c in json.loads((BUNDLE/"setup_commands.json").read_text()):
    print("setup:", c[:80], flush=True)
    subprocess.run(c, shell=True, check=True, cwd=WORKING, env=env)
    env = cmd_env(); os.environ.update(env)
for e in reversed([x for x in os.environ.get("PYTHONPATH","").split(os.pathsep) if x]):
    if e not in sys.path: sys.path.insert(0, e)
print("setup completo")
''',
    '''\
# Cargar benchmark + target, jugar (offline recortado / gateway en rerun)
with open(BUNDLE/"deploy_target.pkl","rb") as f: target = pickle.load(f)
target.actual_run_as_submission = TRUE_SUBMISSION
target.is_competition_rerun = TRUE_SUBMISSION
with open(BUNDLE/"benchmark_initial.pkl","rb") as f: bm = pickle.load(f)
bm.job_dir = WORKING; bm.n_passes = 1; bm.game_weights = None
os.environ.setdefault("RECORDINGS_DIR", str(WORKING/"server_recording"))

# Graft install (base = config v12 de thtennant, que marco 1.17) + schema_helpers:
# precarga helpers de analisis testeados (grid_diff, connected_components,
# action_effect_summary, recent_history) en el sandbox python del agente — el 27B
# reescribe esa plomeria con bugs en cada juego. NUESTRA tesis de feature injection,
# implementada por el autor del fork como graft sin habilitar (WP3).
# goalkeep NO va: marco 0.81 en el set oculto (-0.36 vs v12; ver working notes
# 2026-08-12). Blindado: cualquier fallo -> stock.
# Verificado en CPU local con scripts/smoke_graft_install.py (banner + prelude 8KB).
try:
    from taaf_grafts.composite import install as _graft_install
    _graft_install(bm, flags={"efficiency": True, "retry_guard": True,
                              "shortcircuit": True, "schema_helpers": True})
except Exception as exc:
    print(f"[taaf_grafts] graft failed, running stock: {type(exc).__name__}: {exc}")

import arc_agi, taaf.game_api
def games_offline(d):
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=d)
    ar = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=d)
    return [taaf.game_api.GameAPI(env_name=e.game_id, arcade_spec=spec) for e in ar.available_environments]
def games_comp():
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.COMPETITION,
                                    arc_base_url=os.environ["ARC_BASE_URL"], environments_dir="")
    ar = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION,
                        arc_base_url=spec.arc_base_url, environments_dir="")
    return [taaf.game_api.GameAPI(env_name=e.game_id, arcade_spec=spec) for e in ar.available_environments]

soft_end = None
if TRUE_SUBMISSION:
    os.environ.setdefault("ARC_API_KEY","test-key-123")
    os.environ.setdefault("ARC_BASE_URL","http://gateway:8001/")
    dl = time.monotonic()+600
    while time.monotonic()<dl:
        try:
            with urlopen(os.environ["ARC_BASE_URL"]+"api/games", timeout=10) as r:
                if r.status<500: break
        except Exception: pass
        time.sleep(5)
    bm.games = games_comp()
else:
    bm.games = games_offline(str(COMP_ROOT/"environment_files"))
    soft_end = datetime.fromtimestamp(NOTEBOOK_START)+timedelta(minutes=OFFLINE_SOFT_MIN)

import pandas as pd
pd.DataFrame([["1_0","1",True,1]], columns=["row_id","game_id","end_of_game","score"]).to_parquet(WORKING/"submission.parquet", index=False)

try:
    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)
finally:
    for c in json.loads((BUNDLE/"teardown_commands.json").read_text()):
        subprocess.run(c, shell=True, check=False, cwd=WORKING, env=cmd_env())
print("run terminado")
''',
]




def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    # Experimento de prellenado (2026-08-16): la caché de prefijos acierta solo
    # 44% porque 28 conversaciones de hasta 32768 tokens no caben en los 177.968
    # de memoria de atención -> desalojo y recálculo. Bajar la ventana debería
    # subir el acierto y liberar la tarjeta para GENERAR en vez de releer.
    ap.add_argument("--context-window", type=int, default=None,
                    help="pasa context_window al composite (default: sin tocar = 32768)")
    ap.add_argument("--out", default=None, help="ruta del notebook generado")
    # Concurrencia: 28 juegos simultaneos saturan la memoria de atencion (177.968
    # tokens => 6.356 por juego, con contextos de 32.768) -> desalojo y recalculo.
    # Bajarla da mas memoria por juego SIN recortarle el historial al agente.
    ap.add_argument("--concurrency", type=int, default=None)
    # Ventana offline: el default de 25 min solo produce ~9.500 tokens por juego,
    # mientras el rerun oculto produce ~52.000 -> los experimentos cortos NO ven
    # el regimen de historial largo donde vive el problema real.
    ap.add_argument("--soft-min", type=int, default=None)
    # AMPLIFICACION: inyecta los helpers de navegacion de src/arc3 en el mismo
    # prelude del sandbox que usa schema_helpers, para que UN turno de pensamiento
    # ejecute una ruta completa en vez de una accion. Ver src/arc3/sandbox_nav.py.
    ap.add_argument("--nav", action="store_true")
    # CARGA v2 (seam C): inyecta la tabla de efectos MEDIDA como texto en el
    # prompt, calculada del historial que el harness ya tiene. Cero turnos.
    # Justificada por el banco micro: 44.0% -> 66.1% en planificacion a 4B,
    # con 24 de 24 discordantes a favor. Ver docs/DESIGN.md 8.11.
    ap.add_argument("--effects", action="store_true")
    # HIBRIDO (docs/DESIGN.md 8.22): el explorador CPU juega antes que el LLM.
    # 2.000 acciones sobre el gateway a ~0.15 s = ~5 min de la ventana de 132.
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--hybrid-actions", type=int, default=2000)
    ap.add_argument("--hybrid-seconds", type=float, default=420.0)
    ap.add_argument("--banking", action="store_true")
    args = ap.parse_args()

    cells = list(CELLS)
    if args.soft_min:
        cells = [c.replace('TAAF_OFFLINE_SOFT_MIN", "25"',
                           f'TAAF_OFFLINE_SOFT_MIN", "{args.soft_min}"') for c in cells]
    if args.concurrency:
        for i, c in enumerate(cells):
            if 'bm.job_dir = WORKING' in c:
                cells[i] = c.replace(
                    'bm.job_dir = WORKING',
                    f'bm.solver.concurrency = {args.concurrency}\nbm.job_dir = WORKING')
                break
        else:
            print("ERROR: no encontre donde inyectar concurrency")
            return 1
    if args.banking:
        # CANDIDATO FUERTE #1 (STRATEGY §10): win-then-replay. Toca la mecanica del
        # score (MAX sobre plays: cada victoria compra un segundo intento mas
        # hondo). Analogo del cheap-first de AG2 (+1.67, su mayor palanca unica).
        # Injerto del autor del fork, disenado para degradar a stock ante cualquier
        # divergencia. Pre-filtro: A/B offline 2h en regimen (esta build) antes de
        # gastar noches de lectura.
        for i, c in enumerate(cells):
            if '"shortcircuit": True, "schema_helpers": True' in c:
                cells[i] = c.replace(
                    '"shortcircuit": True, "schema_helpers": True',
                    '"shortcircuit": True, "schema_helpers": True,\n'
                    '                              "banking": True')
                break
        else:
            print("ERROR: no encontré el dict de flags para inyectar banking")
            return 1

    if args.context_window:
        for i, c in enumerate(cells):
            if '"shortcircuit": True, "schema_helpers": True' in c:
                cells[i] = c.replace(
                    '"shortcircuit": True, "schema_helpers": True',
                    f'"shortcircuit": True, "schema_helpers": True,\n'
                    f'                              "context_window": {args.context_window}')
                break
        else:
            print("ERROR: no encontré el dict de flags para inyectar context_window")
            return 1

    if args.nav:
        import base64
        sys.path.insert(0, str(ROOT / "src"))
        from arc3.sandbox_nav import NAV_PROMPT_NOTE, build_nav_prelude
        # base64 evita todo problema de escapado: el prelude lleva comillas triples.
        b64 = base64.b64encode(build_nav_prelude().encode("utf-8")).decode("ascii")
        note_b64 = base64.b64encode(NAV_PROMPT_NOTE.encode("utf-8")).decode("ascii")
        patch = f'''
# AMPLIFICACION (nuestro diferencial): anadimos los helpers de navegacion al mismo
# prelude que schema_helpers inyecta en el sandbox del agente. schema_helpers lee
# SANDBOX_HELPERS_PRELUDE y HELPERS_PROMPT_NOTE en CADA llamada (no los captura al
# importar), asi que extenderlos aqui basta. Verificado en CPU con
# scripts/test_sandbox_nav.py (compila bajo SAFE_BUILTINS, aprende el modelo de
# movimiento, descarta acciones erraticas, degrada a vacio sin transiciones).
try:
    import base64 as _b64
    import taaf_grafts.schema_helpers as _sh
    _nav_src = _b64.b64decode("{b64}").decode("utf-8")
    _nav_note = _b64.b64decode("{note_b64}").decode("utf-8")
    compile(_sh.SANDBOX_HELPERS_PRELUDE + "\\n" + _nav_src, "<check>", "exec")
    _sh.SANDBOX_HELPERS_PRELUDE = _sh.SANDBOX_HELPERS_PRELUDE + "\\n" + _nav_src
    _sh.HELPERS_PROMPT_NOTE = _sh.HELPERS_PROMPT_NOTE + "\\n" + _nav_note
    print("NAV_HELPERS injected:", len(_nav_src), "chars")
except Exception as exc:
    print(f"[nav_helpers] injection failed, running stock: {{type(exc).__name__}}: {{exc}}")
'''
        for i, c in enumerate(cells):
            if "taaf_grafts.composite import install" in c:
                cells[i] = c.replace("\nimport arc_agi, taaf.game_api",
                                     patch + "\nimport arc_agi, taaf.game_api")
                break
        else:
            print("ERROR: no encontre la celda del install de grafts")
            return 1

    if args.hybrid:
        import base64
        sys.path.insert(0, str(ROOT / "src"))
        mods = {}
        for name in ("features", "agent", "hybrid_prelude"):
            src = (ROOT / "src" / "arc3" / f"{name}.py").read_text(encoding="utf-8")
            mods[name] = base64.b64encode(src.encode("utf-8")).decode("ascii")
        pares = ", ".join(f'("{n}", "{b}")' for n, b in mods.items())
        patch = f'''
# HIBRIDO SECUENCIAL: el explorador CPU juega ANTES que el LLM en cada juego.
#
# Medido (docs/DESIGN.md 8.22) sobre los 25 juegos locales con regimen de 2 h:
#   LLM 9 niveles | explorador 18 | UNION 21 — y en parte disjuntos: bp35 y sb26
#   los gana solo el LLM (el explorador da cero alli incluso con 40.000 acciones),
#   tu93 y vc33 solo el explorador.
#
# SECUENCIAL, no concurrente: corre antes de `play()`, asi que `seed_initial_history`
# abre el historial del LLM desde el estado resultante. No hay accion obsoleta, ni
# historial con jugadas ajenas, ni carrera sobre el fichero de estado (los tres
# problemas de 8.21).
#
# Probado contra el `taaf.game.Game` REAL en local (scripts/test_hybrid_prelude.py):
# 6 niveles en 12 juegos con 2.000 acciones. Ese test cazo que `available_actions`
# son ints y no nombres — un stub no lo habria visto.
try:
    import base64 as _b64, sys as _sys, types as _types
    _pkg = _types.ModuleType("arc3"); _pkg.__path__ = []
    _sys.modules["arc3"] = _pkg
    for _n, _s in ({pares}):
        _m = _types.ModuleType("arc3." + _n)
        _m.__package__ = "arc3"
        exec(compile(_b64.b64decode(_s).decode("utf-8"), "arc3/" + _n + ".py", "exec"),
             _m.__dict__)
        _sys.modules["arc3." + _n] = _m
        setattr(_pkg, _n, _m)
    from arc3.hybrid_prelude import run_prelude as _run_prelude
    import inference.framework.solver as _slv
    _HY_ACC = int(os.environ.get("HYBRID_ACTIONS", "{args.hybrid_actions}"))
    _HY_SEC = float(os.environ.get("HYBRID_SECONDS", "{args.hybrid_seconds}"))
    _orig_play = _slv._HarnessGameSession.play

    def _play_con_preludio(self):
        try:
            _r = _run_prelude(self.game, max_actions=_HY_ACC, max_seconds=_HY_SEC,
                              should_stop=self.should_stop)
            print(f"[hybrid] {{getattr(self.game.game_run,'game_id','?')}}: "
                  f"{{_r['acciones']}} acciones, nivel {{_r['niveles']}}, {{_r['motivo']}}",
                  flush=True)
        except Exception as _exc:   # el preludio NUNCA debe impedir jugar
            print(f"[hybrid] preludio fallo, sigo stock: {{type(_exc).__name__}}: {{_exc}}",
                  flush=True)
        return _orig_play(self)

    _slv._HarnessGameSession.play = _play_con_preludio
    print(f"HYBRID_PRELUDE installed: {{_HY_ACC}} acciones / {{_HY_SEC}}s por juego")
except Exception as exc:
    print(f"[hybrid_prelude] injection failed, running stock: {{type(exc).__name__}}: {{exc}}")
'''
        for i, c in enumerate(cells):
            if "taaf_grafts.composite import install" in c:
                cells[i] = c.replace("\nimport arc_agi, taaf.game_api",
                                     patch + "\nimport arc_agi, taaf.game_api")
                break
        else:
            print("ERROR: no encontre la celda del install de grafts")
            return 1

    if args.effects:
        import base64
        sys.path.insert(0, str(ROOT / "src"))
        src = (ROOT / "src" / "arc3" / "effects_model.py").read_text(encoding="utf-8")
        b64 = base64.b64encode(src.encode("utf-8")).decode("ascii")
        patch = f'''
# CARGA DEL SEAM C (v2 de la amplificacion). La v1 inyectaba una FUNCION en el
# sandbox: se adoptaba (726 llamadas en 25/25 juegos) pero costaba un turno
# llamarla y devolvia vacio en los juegos sin movimiento. Esta v2 entrega el DATO
# YA CALCULADO como texto en el prompt: cero turnos, cero llamadas al sandbox, y
# nota no vacia en 25/25 juegos locales.
#
# Evidencia (banco micro, T4, 2026-08-19): con la tabla de efectos medida la
# planificacion sube de 44.0% a 66.1% en Qwen3-4B y de 24 items discordantes los
# 24 van a favor de la tabla, ninguno en contra (p aprox 0). OJO: a 1.7B el mismo
# dato PERJUDICA (15 vs 5, p=0.041) — hay un umbral de capacidad, y el modelo de
# produccion (27B) esta por encima de ambos.
#
# El detector esta validado por prediccion fuera de muestra sobre los 25 juegos
# locales: 96.6% (141/146) con confianza >= 0.6. Por debajo de ese umbral la nota
# NO afirma un desplazamiento, degrada a incertidumbre honesta.
try:
    import base64 as _b64
    import taaf_grafts.schema_helpers as _sh
    _ns = {{}}
    exec(compile(_b64.b64decode("{b64}").decode("utf-8"), "effects_model.py", "exec"), _ns)
    _effects_from_history = _ns["effects_from_history"]
    _render_effects_note = _ns["render_effects_note"]
    _orig_bup = _sh.SchemaHelpersToolAgent._build_user_prompt

    def _bup_with_effects(self, action_num, **kw):
        base = _orig_bup(self, action_num, **kw)
        try:
            note = _render_effects_note(_effects_from_history(kw.get("history_entries") or []))
        except Exception:
            return base          # cualquier fallo => prompt del padre, intacto
        return f"{{base}}\\n{{note}}" if note else base

    _sh.SchemaHelpersToolAgent._build_user_prompt = _bup_with_effects
    print("EFFECTS_NOTE injected on seam C:", len(_ns), "symbols")
except Exception as exc:
    print(f"[effects_note] injection failed, running stock: {{type(exc).__name__}}: {{exc}}")
'''
        for i, c in enumerate(cells):
            if "taaf_grafts.composite import install" in c:
                cells[i] = c.replace("\nimport arc_agi, taaf.game_api",
                                     patch + "\nimport arc_agi, taaf.game_api")
                break
        else:
            print("ERROR: no encontre la celda del install de grafts")
            return 1

    out = Path(args.out) if args.out else OUT
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}},
          "cells": [code_cell(c) for c in cells]}
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"generado {out}" + (f" (context_window={args.context_window})"
                               if args.context_window else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

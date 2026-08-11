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

# Graft install (replica del duck v12 publico, thtennant/taaf-kaggle-source-share-fork):
# analizadores de eficiencia sobre el solver TAAF. Blindado: cualquier fallo -> stock.
try:
    from taaf_grafts.composite import install as _graft_install
    _graft_install(bm, flags={"efficiency": True, "retry_guard": True, "shortcircuit": True})
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
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}},
          "cells": [code_cell(c) for c in CELLS]}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"generado {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

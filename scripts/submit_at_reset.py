"""Espera al reset UTC de la cuota de envios y manda un kernel al set oculto.

La competencia admite UN envio por dia UTC. Cuando el cupo del dia ya esta
gastado, la API responde 400 con FAILED_PRECONDITION y el CLI solo muestra
"400 Client Error" — por eso aqui se imprime el cuerpo del error, que si dice
el motivo real.

Verifica el resultado contra la lista de envios (el CLI no imprime nada al
tener exito, asi que "sin salida" no es prueba de nada).

Uso:
  python scripts/submit_at_reset.py juliancamilovilla/arc-agi3-duck-anim "mensaje"
  python scripts/submit_at_reset.py <kernel> "<mensaje>" --now   # sin esperar
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "arc-prize-2026-arc-agi-3"
LOG = ROOT / "daily_submit.log"


def log(msg: str) -> None:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{stamp}  {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--now"]
    wait = "--now" not in sys.argv
    kernel = args[0]
    message = args[1]
    version = json.loads((ROOT / "kernel_versions.json").read_text(encoding="utf-8"))[kernel]

    load_env()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    # NO se calcula el instante del reset: se INTENTA y se reintenta hasta que el
    # cupo abra. Asi el envio es inmune a la zona horaria de la maquina, al
    # desfase del reloj del servidor y a que la tarea arranque antes o despues.
    # El 400 del cupo agotado es FAILED_PRECONDITION y trae el motivo en el
    # cuerpo (el CLI lo oculta); cualquier otro error se registra igual y se
    # reintenta, porque un fallo de red no puede costar el envio del dia.
    max_minutes = int(os.environ.get("SUBMIT_MAX_MINUTES", "240")) if wait else 1
    deadline = time.time() + max_minutes * 60
    log(f"armado: {kernel} v{version} — reintentando hasta {max_minutes} min "
        f"o hasta que el cupo abra")
    attempt = 0
    while True:
        attempt += 1
        try:
            api.competition_submit_code("submission.parquet", message, COMP,
                                        kernel, int(version))
            log(f"ENVIADO {kernel} v{version} (intento {attempt})")
            break
        except Exception as exc:
            resp = getattr(exc, "response", None)
            body = resp.text[:300] if resp is not None else str(exc)[:300]
            if attempt == 1 or attempt % 15 == 0:
                log(f"intento {attempt}: {body}")
            if time.time() >= deadline:
                log(f"AGOTADO tras {attempt} intentos: {body}")
                return 1
            time.sleep(60)

    # El CLI no imprime nada al tener exito -> verificar contra la lista.
    time.sleep(15)
    subs = api.competition_submissions(COMP)
    if not subs:
        log("VERIFICACION: la lista de envios vino vacia")
        return 1
    newest = subs[0]
    log(f"envio mas reciente: ref={getattr(newest, 'ref', '?')} "
        f"fecha={getattr(newest, 'date', '?')} estado={getattr(newest, 'status', '?')}")
    desc = str(getattr(newest, "description", ""))
    ok = message[:40] in desc
    log("OK (verificado en la lista)" if ok else f"AVISO: descripcion inesperada: {desc[:120]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

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


def seconds_to_reset() -> float:
    now = datetime.datetime.now(datetime.timezone.utc)
    nxt = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=20, microsecond=0)
    return (nxt - now).total_seconds()


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

    if wait:
        delay = seconds_to_reset()
        log(f"esperando {delay/60:.1f} min al reset UTC para enviar {kernel} v{version}")
        while delay > 0:
            time.sleep(min(delay, 60))
            delay = seconds_to_reset()
            if delay > 23 * 3600:  # ya pasamos el reset
                break

    # Hasta 6 intentos: el cupo se abre en el instante exacto y el reloj del
    # servidor puede ir unos segundos por detras del nuestro.
    for attempt in range(1, 7):
        try:
            api.competition_submit_code("submission.parquet", message, COMP,
                                        kernel, int(version))
            log(f"ENVIADO {kernel} v{version}")
            break
        except Exception as exc:
            resp = getattr(exc, "response", None)
            body = resp.text[:400] if resp is not None else str(exc)[:400]
            log(f"intento {attempt} fallo: {body}")
            if attempt == 6:
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

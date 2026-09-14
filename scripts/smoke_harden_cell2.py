"""Smoke LOCAL de la celda 2 endurecida: extrae la funcion _pip_install_retry de la celda
EXACTA del notebook y prueba su semantica con comandos reales de este interprete:
(1) un comando que falla siempre -> N intentos, stderr impreso, RuntimeError al final;
(2) un comando que tiene exito -> retorna sin excepcion al primer intento.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "nvfp4_carry_yield_long.ipynb"

fallos: list[str] = []


def check(ok, nombre, detalle=""):
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print(f"SMOKE DE LA CELDA 2 ENDURECIDA ({NB.name})")
    nb = json.loads(NB.read_text(encoding="utf-8"))
    celda = next(("".join(c["source"]) for c in nb["cells"]
                  if c.get("cell_type") == "code" and "_pip_install_retry" in "".join(c["source"])), None)
    check(celda is not None, "la celda endurecida esta en el notebook")
    if celda is None:
        return 1
    # Solo la definicion de la funcion (sin ejecutar la instalacion real).
    defn = celda[celda.index("def _pip_install_retry"):celda.index("\n\n_pip_install_retry(")]
    ns = {"subprocess": subprocess, "sys": sys}
    exec(compile(defn, "<def>", "exec"), ns)
    f = ns["_pip_install_retry"]

    falla = [sys.executable, "-c", "import sys; sys.stderr.write('motivo simulado\\n'); sys.exit(1)"]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            f(falla, tries=3, wait_s=0)
        check(False, "un comando que falla siempre termina en RuntimeError")
    except RuntimeError as e:
        out = buf.getvalue()
        check(out.count("fallo (rc=1)") == 3, "hace exactamente 3 intentos", f"{out.count('fallo (rc=1)')}")
        check("motivo simulado" in out, "imprime el stderr del pip en cada intento")
        check("WHEELHOUSE_DIR" in out, "al agotar, intenta listar el wheelhouse (diagnostico)")
        check("motivo simulado" in str(e), "la excepcion final lleva el ultimo stderr")

    ok_cmd = [sys.executable, "-c", "pass"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        f(ok_cmd, tries=3, wait_s=0)
    check("WHEELHOUSE_INSTALL ok (intento 1)" in buf.getvalue(), "un comando que va bien retorna al primer intento")

    print(f"\nSMOKE: {'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

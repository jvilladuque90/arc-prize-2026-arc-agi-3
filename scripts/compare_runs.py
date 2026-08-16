"""Compara dos validaciones Save&Run juego a juego (acciones y niveles).

La metrica que importa no es el total de acciones sino cuantos juegos cruzan el
umbral empirico de ~18 acciones: en las validaciones medidas, NINGUN juego con
menos de 18 acciones completo jamas un nivel.

Uso: python scripts/compare_runs.py _tmp_ducklog_v4 _tmp_ctxlog16
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

THRESHOLD = 18


def parse(job_dir: str) -> dict[str, tuple[float, int]]:
    text = (Path(job_dir) / "summary.txt").read_text(errors="replace")
    out = {}
    for m in re.finditer(
        r"(\w+)-\w+: score=([\d.]+), levels=([\d.]+)/(\d+), actions=(\d+), tokens=(\d+)",
        text,
    ):
        out[m.group(1)] = (float(m.group(3)), int(m.group(5)))
    return out


def main() -> int:
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    a, b = parse(a_dir), parse(b_dir)
    print(f"{'juego':8} {'A_acc':>6} {'B_acc':>6} {'delta':>7}   niveles A->B")
    ta = tb = 0
    ca = cb = 0
    for g in sorted(a):
        la, aa = a[g]
        lb, ab = b.get(g, (0.0, 0))
        ta += aa
        tb += ab
        ca += aa >= THRESHOLD
        cb += ab >= THRESHOLD
        print(f"{g:8} {aa:6} {ab:6} {ab - aa:+7}   {la:.0f} -> {lb:.0f}")
    print(f"\nA = {a_dir}   B = {b_dir}")
    print(f"acciones totales: {ta} -> {tb} ({(tb / ta - 1) * 100:+.0f}%)")
    print(f"juegos con >={THRESHOLD} acciones: {ca} -> {cb}")
    print(f"niveles: {sum(v[0] for v in a.values()):.0f} -> "
          f"{sum(v[0] for v in b.values()):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

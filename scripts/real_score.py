"""Puntua corridas offline en UNIDADES DEL CONCURSO. La vara del ciclo.

Hasta 2026-09-01 todos nuestros numeros offline eran "niveles completados",
que NO es la metrica del concurso. La real (verificada contra dos fuentes:
arc_agi/scorecard.py y docs.arcprize.org/methodology):

    score_nivel = min((baseline_humano / acciones)^2 * 100, 115)   si completado
    score_juego = suma(score_nivel_i * i) / suma(i)                peso = nivel
    score_final = media sobre los juegos

El framework TAAF ya la calcula (taaf/game.py::_compute_final_score, "Mirrors
arc_agi.scorecard") y la escribe como `final_score` en benchmark.json de cada
validacion — o sea que TODAS las corridas pasadas son re-puntuables gratis.
Este script la recalcula desde base_actions_per_level/actions_per_level (no se
fia del campo) y verifica que ambas coincidan: si divergen, el instrumento esta
roto y lo dice.

Calibracion (2026-09-01, verificacion cruzada 0 discrepancias en 9 corridas):
  validaciones cortas 2h: 0.09-0.45 | largas: _tmp_long_c28=1.06, _tmp_nav=1.32
  (cuadra con LB 0.95-1.17 de esas fechas). El objetivo del ciclo es 7.0 aqui.

Uso: python scripts/real_score.py [dir ...]     (sin args: todos los _tmp_*)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def score_juego(base: list[int], acts: list[int], done: int) -> float:
    """Formula oficial exacta (scorecard.py:168-206 / taaf game.py:381-411)."""
    ts, tw, mw = 0.0, 0, 0
    for i, b in enumerate(base):
        w = i + 1
        tw += w
        a = acts[i] if i < len(acts) else 0
        s = min(115.0, (b / a) ** 2 * 100) if i < done and a > 0 else 0.0
        if s > 0:
            mw += w
        ts += s * w
    return min(ts / tw, mw / tw * 100) if tw else 0.0


def puntuar_corrida(d: Path) -> dict | None:
    bj = d / "benchmark.json"
    if not bj.exists():
        return None
    runs = json.loads(bj.read_text(encoding="utf-8"))["game_runs"]
    juegos, discrepancias = [], 0
    for r in runs:
        mio = score_juego(r["base_actions_per_level"], r["actions_per_level"],
                          r["levels_completed"])
        marco = r.get("final_score") or 0.0
        if abs(mio - marco) > 0.01:
            discrepancias += 1
        juegos.append({"id": r["game_id"], "score": round(mio, 3),
                       "niveles": r["levels_completed"],
                       "acciones": sum(r["actions_per_level"]),
                       "baseline": r["base_actions_per_level"],
                       "acciones_nivel": r["actions_per_level"]})
    n = len(juegos)
    return {"dir": str(d), "n_juegos": n,
            "score": round(sum(j["score"] for j in juegos) / n, 3) if n else 0.0,
            "niveles": sum(j["niveles"] for j in juegos),
            "discrepancias_vs_framework": discrepancias,
            "juegos": sorted(juegos, key=lambda j: -j["score"])}


def main() -> int:
    dirs = ([Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1
            else sorted(ROOT.glob("_tmp_*")))
    filas = [r for d in dirs if (r := puntuar_corrida(d))]
    if not filas:
        print("ninguna corrida con benchmark.json")
        return 1
    print(f'{"corrida":24} {"SCORE":>7} {"niveles":>8} {"top juegos (score real)"}')
    for r in filas:
        if r["discrepancias_vs_framework"]:
            print(f'!! {r["dir"]}: {r["discrepancias_vs_framework"]} juegos no '
                  f'cuadran con el framework — instrumento roto, no uses esta fila')
        top = ", ".join(f'{j["id"].split("-")[0]}={j["score"]:.1f}'
                        for j in r["juegos"][:4] if j["score"] > 0) or "-"
        print(f'{Path(r["dir"]).name:24} {r["score"]:7.3f} {r["niveles"]:8} {top}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

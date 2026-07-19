# ARC Prize 2026 — ARC-AGI-3

Solución para la competencia de Kaggle
[ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3).

> **Objetivo:** un agente que juegue environments interactivos ocultos (grids 64×64,
> colores 0–15) explorando, modelando el mundo y completando niveles. A diferencia de
> ARC-AGI-2 (estático), aquí el agente **actúa**: RESET, ACTION1–5 y 7 (simples),
> ACTION6 (click x,y). El score sale de los niveles completados en el set oculto.

## Mecánica de la competencia

| Aspecto | Detalle |
|---|---|
| Tipo | Code competition: notebook Kaggle que ejecuta el agente **offline** |
| Evaluación | Fase A: "Save & Run All" valida el notebook; Fase B: submit → rerun sobre juegos ocultos |
| Internet | **NO** durante evaluación (los wheels de la competencia instalan arcengine/arc_agi offline) |
| GPU | CPU / T4×2 / P100 / **RTX Pro 6000 ("G4", exclusiva de ARC-AGI-3)** — `machine_shape: NvidiaRtxPro6000` |
| Open source | Obligatorio publicar la solución para optar a premios (milestones 30-jun y 30-sep 2026) |

## Datos (NO versionados — ver `.gitignore`)

`python scripts/download_data.py` descarga y descomprime en el root:

- `environment_files/<juego>/<hash>/` — 25 juegos públicos (código python del environment + metadata).
- `ARC-AGI-3-Agents/` — framework oficial de agentes (repo de arcprize).
- `arc_agi_3_wheels/` — wheels para instalar `arcengine` + `arc_agi` sin internet.

Por privacidad/reglas de Kaggle, nada de eso (ni `.env` con credenciales) entra al repo.

## Arquitectura (truco save-and-run de AG2)

Sin sesiones interactivas de Kaggle: los notebooks se publican **headless** con
`kaggle kernels push` (batch = "Save & Run All"). Los kernels CPU no gastan cuota de
GPU; la G4 (RTX Pro 6000) se reserva para entrenamiento. La config de la G4 está
copiada del notebook guía `juliancamilovilla/rtx-g4` (imagen Docker pineada).

```bash
# credenciales Kaggle en .env (KAGGLE_API_TOKEN, kaggle_username)
python scripts/download_data.py                # datos de la competencia
python scripts/build_features_notebook.py      # regenera notebooks/features.ipynb desde src/
python scripts/push_kernels.py features        # publica y corre en Kaggle (CPU, sin cuota GPU)
python scripts/push_kernels.py features --gpu  # solo si hace falta la RTX Pro 6000
```

## Feature engineering (`src/arc3`)

- [`features.py`](src/arc3/features.py) — por frame: histograma de 16 colores, entropía,
  densidad de bordes, simetrías H/V, objetos por componentes conexas (tamaño, bbox,
  centroide, top-8). Por transición (s, a, s'): píxeles cambiados, bbox del cambio,
  colores ganados/perdidos y **detección de traslación** (vector dy,dx del objeto movido).
- [`probe.py`](src/arc3/probe.py) — política de sondeo: round-robin de acciones simples +
  malla de clicks para ACTION6; maneja GAME_OVER→RESET; presupuesto por juego.
- [`env.py`](src/arc3/env.py) — ejecución local de environments vía `arc_agi`/`arcengine`
  (idéntico al modo offline del rerun de Kaggle).

Salidas (`scripts/extract_features.py`, local, o el kernel `arc-agi3-features` en Kaggle):

- `transitions.parquet` — una fila por (juego, paso): ~90 features.
- `action_summary.csv` — perfil por (juego, acción): p_change, píxeles cambiados, p_level_up.
- `games_summary.csv` — una fila por juego (tags, niveles, respuesta a acciones).

Estas features alimentan al agente: clasificación de juego (keyboard vs click),
regiones interactivas y firma estructural de objetos.

## Agente: GraphExplorer (`src/arc3/agent.py`)

Exploración de grafo de estados (síntesis de lo mejor del LB público, ver
[docs/STRATEGY.md](docs/STRATEGY.md)): hashing de frames con máscara de borde 3px +
máscara de contador aprendida; candidatos de click por componentes conexas ordenadas por
*button-likeness* + rejilla de cobertura; supresión *deadsig* de clases de click inertes;
BFS sobre el grafo aprendido hacia nodos con acciones pendientes y replay determinista.

Evaluación local (`python scripts/eval_agent.py`): **20 niveles / 25 juegos**
(0.80/juego) con 5000 acciones y ≤90 s por juego, CPU pura.

La submission (`scripts/build_submit_notebook.py` → `notebooks/submit.ipynb`,
kernel `arc-agi3-submit`) es dual-mode: en el rerun real espera el gateway
(`http://gateway:8001`) y juega los juegos ocultos vía `arc_agi` en modo competition;
en Save & Run juega los 25 públicos offline como validación. El `submission.parquet`
es un dummy — el score lo calculan las partidas contra el gateway.

## Setup local

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install arc_agi_3_wheels\arcengine-0.9.3-py3-none-any.whl arc_agi_3_wheels\arc_agi-0.9.8-py3-none-any.whl pandas pyarrow
python scripts/extract_features.py --games ls20 --budget 120
```

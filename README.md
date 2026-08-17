# ARC Prize 2026 — ARC-AGI-3

Solución para la competencia de Kaggle
[ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3).

> **Objetivo:** un agente que juegue environments interactivos ocultos (grids 64×64,
> colores 0–15) explorando, modelando el mundo y completando niveles. A diferencia de
> ARC-AGI-2 (estático), aquí el agente **actúa**: RESET, ACTION1–5 y 7 (simples),
> ACTION6 (click x,y). El score sale de los niveles completados en el set oculto.

## Estado actual (2026-08-17)

| | |
|---|---|
| **Configuración en producción** | duck v4: harness TAAF + injertos `efficiency`/`retry_guard`/`shortcircuit`/`schema_helpers` |
| Puntaje oculto de v4 | **1.10** (n=1) |
| Línea base del mismo harness | media **0.98** sobre 4 muestras {1.17, 1.03, 0.76, 0.96} — mismo código, **rango 0.41** |
| Mejor puntaje histórico | **1.17** · leaderboard: puntero 2.52, segundo 1.86 |
| Nuestro stack propio de exploración | 0.25 (techo medido en 7 envíos) |
| Envío diario | automático a las 20:00 (`scripts/daily_submit.ps1`, envía solo versiones validadas) |

**Dónde está el cuello** ([DESIGN.md §8.9](docs/DESIGN.md)): cada juego dispone de ~52.000 tokens
y ejecuta ~94 acciones en las 8 horas del rerun. Medimos que **se releen ~26 tokens por cada uno
que se escribe** y que la caché de prefijos solo acierta el 44% — una ineficiencia real. Pero los
dos experimentos que la atacaron de frente **fallaron** (recortar contexto: 0.60 en el oculto;
bajar concurrencia: menos niveles), lo que indica que **el presupuesto no es la restricción
activa**. Por encima de un piso de ~18 acciones por juego, acciones y niveles están desacoplados:
la frontera es **semántica** — el agente no infiere la regla ni la meta.

**Lo que sí está validado como capacidad**: los *seams* de inyección
([ARCHITECTURE.md §2](docs/ARCHITECTURE.md)) — una sola línea de nota bastó para que el modelo
adoptara código nuestro **726 veces en 25 de 25 juegos**. El canal funciona; el siguiente paso es
la carga correcta: inyectar **información ya calculada** en el prompt (coste cero en turnos) en
vez de funciones que el modelo deba llamar.

Documentación viva: [docs/STRATEGY.md](docs/STRATEGY.md) (estado y palancas, con las cerradas
marcadas) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (el stack, los seams, los instrumentos de
medición) · [docs/DESIGN.md](docs/DESIGN.md) (problema, features, física del presupuesto y su
corrección) · [paper/working_note_es.md](paper/working_note_es.md) /
[working_note_en.md](paper/working_note_en.md) (bitácora experimental bilingüe, con cada resultado
y las lecturas honestas de los errores).

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

## Agente actual: harness duck + injertos (`scripts/build_duck_notebook.py`)

Desde el 2026-08-11 competimos con una réplica del harness público de Tufa Labs (modelo
Qwen3-27B-FP8 servido con vLLM en la RTX Pro 6000) más los injertos del fork de thtennant, que
se instalan con una sola llamada blindada: si algo falla, cae a la configuración estándar.

| Injerto | Qué hace | Estado |
|---|---|---|
| `efficiency`, `retry_guard`, `shortcircuit` | nota de presupuesto por turno, reintentos, recorte de sobre-exploración | activos (línea base 0.98) |
| `schema_helpers` | precarga funciones de análisis de grillas en el entorno aislado del agente para que no las reescriba (con errores) en cada juego | **activo desde v4** — −20% de tokens; validado en CPU: 4,6 ms contra un presupuesto de 30 s |
| `goalkeep` | retiene el modelo del mundo entre game-overs | apagado (0.81 en su única muestra, dentro del rango de la línea base → sin evidencia a favor) |
| `banking`, `transfer`, `recovery`, `schema_notes`, `context_window` | disponibles en el paquete, sin activar por la referencia | `context_window` en experimento (ver §8 de DESIGN) |

Verificaciones locales sin gastar cuota: `scripts/smoke_graft_install.py` (el injerto se instala
y engancha el solver) y `scripts/test_schema_helpers.py` (las funciones inyectadas son correctas
y suficientemente rápidas para el entorno aislado).

## Agente propio: GraphExplorer (`src/arc3/agent.py`)

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

## Open source y atribución (elegibilidad ARC Prize)

Las reglas del ARC Prize exigen que **todo el código y métodos sean open source** para optar a
premios (milestone #2: 2026-09-30). Cumplimiento:

- **Repo público**: https://github.com/jvilladuque90/arc-prize-2026-arc-agi-3 (licencia [MIT](LICENSE)).
- **Kernels públicos** en Kaggle: `arc-agi3-duck`, `arc-agi3-explorer054`, `arc-agi3-llm`,
  `arc-agi3-submit`, `arc-agi3-features` (usuario `juliancamilovilla`).
- **Atribución de código de terceros** (todo público, de la propia competencia):
  - `vendor/my_agent_v47.py`: vendorizado del notebook público de poby7722 (LB 0.54 en junio),
    que a su vez porta técnicas de Occam (MIT) y la solución 3rd-place "just-explore" (MIT).
  - `notebooks/duck.ipynb`: réplica del harness TAAF de Tufa Labs (público, ganador milestone jun)
    en su variante fork pública de thtennant (`taaf-kaggle-source-share-fork`, con taaf-grafts),
    con wheels vLLM y snapshot Qwen3-27B-FP8 públicos de driessmit1.
- Los datos de la competencia **no** se redistribuyen en este repo (ver `.gitignore`).

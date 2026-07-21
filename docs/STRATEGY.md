# Estrategia ARC-AGI-3 (2026-07-18)

Objetivo: ganar (milestone 30-sep-2026 y Grand Prize). Basado en el análisis de los 10
notebooks top de la competencia (descargados y estudiados; resúmenes de esta fecha).

## Contrato de submission (confirmado leyendo al ganador del milestone)

- El notebook detecta `KAGGLE_IS_COMPETITION_RERUN`. En rerun real: esperar al **gateway**
  `http://gateway:8001/api/games` (hasta 600 s), jugar los ~110 juegos ocultos vía la
  librería `arc_agi` en modo online/competition (`ARC_API_KEY=test-key-123`). **El score lo
  calcula el gateway con las partidas** (niveles completados); el `submission.parquet`
  (`row_id, game_id, end_of_game, score`) es un dummy que se escribe solo fuera del rerun.
- Fuera del rerun (Save & Run All): jugar los 25 `environment_files` offline con el mismo
  código (`OPERATION_MODE=offline`) — validación gratuita del pipeline completo.
- `ONLY_RESET_LEVELS=true`: RESET reinicia el **nivel**, no el juego.
- Presupuesto: ~8 h de wall-clock; todos los top usan corte a 8h−5min y presupuesto por juego.
- Sin internet: wheels de `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels`;
  código/pesos propios via datasets adjuntos.

## Lecciones del leaderboard

| Enfoque | LB | Lección |
|---|---|---|
| Tufa Labs duck harness (LLM Qwen3-27B-FP8 + vLLM en RTX Pro 6000) | **1.21** (ganador milestone jun) | El techo actual es agentic-LLM local con harness robusto |
| Gemma-4-31B VLM + memoria de reflexión (mbmmurad) | 0.86 | VLM + higiene de ingeniería; la versión simple (1 candidato) ganó a la compleja |
| Base Gemma + **saliencia de objetos + deadsig** (2º milestone) | ~0.9 | El delta vino de features objetuales puras (numpy): botones = componente pequeño + color raro; suprimir clases estructuralmente inertes |
| **Exploración de grafo de estados, sin ML** (poby7722 v47) | 0.54 | Mejor cost/benefit. Hashing con máscara de borde 3px + máscara de contador aprendida; BFS sobre el grafo aprendido; replay tras reset (juegos deterministas) |
| BFS/planning sobre el simulador local + RL CNN online (Forge) | 0.35–0.46 | Cargar el `.py` del juego y planificar offline es potente; el RL online por nivel es peso muerto |

Claves transversales: (1) el problema central es el **hashing de estados** (bordes = HUD,
contadores/animaciones); (2) clicks nunca por fuerza bruta 64×64 → componentes conexas
ordenadas por *button-likeness* + rejilla gruesa; (3) nunca bloquear el loop de acciones;
(4) nunca crashear (try/except + fallback); (5) presupuesto por juego adaptativo.

## Plan por fases

**Fase 1 (ya):** `GraphExplorer` propio — exploración de grafo de estados estilo v47 +
mejoras objetuales del 2º puesto (saliencia + deadsig por clase estructural), sobre nuestro
`src/arc3` de features. Sin GPU → submission CPU (no gasta cuota G4). Meta: ≥0.5 LB y
pipeline de submission validado end-to-end.

**Fase 2:** planner sobre simulador local cuando el fuente del juego esté accesible en el
rerun (cascada estilo FORGE: transferencia entre niveles → BFS con hash trigger-aware →
beam), en thread background, con el GraphExplorer como fallback online. Trucos: pickle-clone,
`hash(tobytes())`, win-field parseado del fuente, campos trigger vs clock.

**Fase 3 (la G4 entra aquí):** harness agentic-LLM local (estilo Tufa/Gemma): VLM servido
con vLLM en la RTX Pro 6000, memoria de reflexión, inyectando **nuestras features
objetuales en el prompt** (hueco que ninguno de los top explota). El código del ganador es
público (`jeroencottaar/taaf-kaggle-source-share`, descargado en scratchpad) — estudiar
`ARC3-Inference/inference/agent/tool_agent.py`.

## Registro de submissions

| ver | método | offline (25) | LB oculto | nota |
|---|---|---|---|---|
| v1 | GraphExplorer serie, CPU | 24 niveles | **0.25** | baseline |
| v2 | + runner paralelo (14 workers) | 18 (contención CPU) | **0.25** | el paralelismo no movió el score → el cuello NO es throughput |
| v3 | + reinicio diversificado (salt: densifica clicks, rota orden) | 17 | **0.25** | los reinicios tampoco mueven el LB oculto → exploración pura tope-capada |
| llm | LLMAgent (Qwen3-27B-FP8 en G4, features en prompt) | 1 (30min) | pend. | vLLM arranca (Marlin FP8); el modelo LEE nuestras features; llm_fails 6.5% |
| hybrid | explorador (piso ~0.25) + LLM al atascarse (techo) | — | pend. | domina a ambos: mantiene el piso y añade el techo LLM gastando pocas inferencias |

**Conclusión clave:** v2==v1 prueba que la exploración pura tiene un techo semántico
(~0.25) en los juegos ocultos, no un límite de compute. Los juegos que quedan en 0
(g50t, re86, wa30, tr87) necesitan entender el objetivo, no explorar más. → la subida
real exige la Fase 3 (LLM). v3 es la última mejora barata de la rama de exploración.

## Fase 3 — harness LLM en la G4 (plan concreto)

El ganador (Tufa Labs) y el 0.86 son VLM locales. Piezas públicas ya localizadas:
- `jeroencottaar/taaf-kaggle-source-share` — bundle con el solver (descargado en scratchpad).
- `driessmit1/arc3-vllm-h100-wheelhouse-v3` — wheels vLLM 0.19 (verificado accesible).
- `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` — pesos 27B FP8 (~27 GB, verificado).
- `deploy_target.pkl`: `accelerator=NvidiaRtxPro6000`, `max_runtime_s≈32400` (9h).

Ruta económica (no quemar cuota G4 a ciegas):
1. Fork del notebook legible del ganador adjuntando esos 3 datasets + GPU RTX Pro 6000;
   correr **una** validación offline con `soft_end` recortado (~20 min) para confirmar que
   vLLM arranca y juega los 25 públicos. Es el baseline LLM a batir.
2. Nuestro aporte diferencial (lo que ninguno del top explota): inyectar en el prompt del
   VLM nuestras **features objetuales** (`src/arc3/features`: componentes, saliencia de
   botones, vector de movimiento, diffs) y usar el GraphExplorer como **fallback** cuando
   el LLM falla/timeout — ensemble LLM+búsqueda. Estudiar `ARC3-Inference/inference/agent/
   tool_agent.py` (scratchpad/taaf_source) para el punto de inyección del prompt.

## Validación

Local (25 juegos públicos) con `arcade.get_scorecard()`: score, niveles, acciones por juego.
Los `baseline_actions` de cada metadata.json dan la referencia de acciones "razonables".
Save & Run offline en Kaggle = misma vara antes de cada submit (1 submit/día).

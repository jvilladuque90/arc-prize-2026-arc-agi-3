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

## Validación

Local (25 juegos públicos) con `arcade.get_scorecard()`: score, niveles, acciones por juego.
Los `baseline_actions` de cada metadata.json dan la referencia de acciones "razonables".
Save & Run offline en Kaggle = misma vara antes de cada submit (1 submit/día).

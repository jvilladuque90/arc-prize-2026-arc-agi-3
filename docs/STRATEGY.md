# Estrategia ARC-AGI-3

> **Documento en dos capas.** La cabecera (§0) es el **estado vigente**; lo que sigue es el
> análisis fundacional del 2026-07-18 (contrato de submission, lecciones del leaderboard,
> plan por fases) que se conserva porque su diagnóstico técnico sigue siendo válido, aunque
> varias de sus conclusiones fueron **corregidas por medición** — cada corrección está marcada.

## §0. Estado vigente (2026-08-16)

**Dónde estamos:** puntaje oculto con el harness duck (Tufa Labs) + injerto `schema_helpers`:
**1.10** en su primera muestra; línea base del mismo harness sin ese injerto = **0.98 de media**
sobre 4 muestras {1.17, 1.03, 0.76, 0.96}. Nuestro stack propio de exploración topaba en 0.25.
Leaderboard: puntero 2.52, segundo 1.86, Tufa 1.62, la masa de forks del duck entre 1.1 y 1.3.

**Las tres correcciones que la medición impuso al plan original:**

1. *"La exploración pura topa en 0.25 por límite semántico"* → **parcialmente falso**. Era el
   techo de NUESTRA implementación; la referencia pública hacía 0.54 en junio. Pero el set
   oculto ROTÓ tras el milestone: la réplica fiel de esa referencia marcó **0.22** en el set
   actual. Lección permanente: **toda referencia tiene fecha; calibrar contra la vigente**.
2. *"Un puntaje es una medición"* → **falso**. El harness tiene varianza enorme: el MISMO código
   dio 0.76 y 1.17 (rango de ~41 niveles). Ninguna decisión con n=1 es válida — dos veces me
   equivoqué así (celebré el 1.17, condené el `goalkeep` con su 0.81, que en realidad cae
   dentro del rango de la línea base).
3. *"Nuestro diferencial será inyectar features objetuales en el prompt"* → **la idea era
   correcta pero llegó por otra vía**: el autor del fork ya la había implementado como injerto
   sin activar (`schema_helpers`, que precarga funciones de análisis de grillas en el entorno
   aislado del agente). La adoptamos en vez de reconstruirla.

**La aritmética que gobierna el puntaje ahora (medida, 2026-08-16):**

```
puntaje ≈ niveles completados ∝ acciones útiles por juego
acciones por juego = (tokens disponibles por juego) / (tokens por acción)
```

Con 195 tokens/segundo de generación repartidos entre 28 juegos durante 8 horas, cada juego
recibe ~52.000 tokens y ejecuta ~94 acciones — y el primer nivel de un juego típico cuesta
entre 7 y 55 acciones jugando perfecto. **De ahí que el techo natural sea ~1 nivel por juego,
que es exactamente donde estamos.** Para llegar a 2.5 (el puntero) hacen falta ~3× más acciones
útiles. Como el caudal de la tarjeta está fijo, la única vía es bajar el costo por acción.

**Dónde se está yendo el presupuesto (auditoría del log del servidor de inferencia):**

| Medida | Valor | Lectura |
|---|---|---|
| Tokens de entrada procesados | 1.950–5.775/s | **26 tokens releídos por cada 1 escrito** |
| Tokens de salida generados | 110–236/s | lo único que produce decisiones |
| Acierto de caché de prefijos | **44%** | debería ser 85–95% en un diálogo que solo crece |
| Memoria de atención | 177.968 tokens para 28 conversaciones de hasta 32.768 | **sobresuscrita ~5×** → desalojo y recálculo |
| Decodificación especulativa | apagada | palanca sin usar |

**Las tres palancas vivas, por valor esperado:**

1. **Reducir la relectura** (perilla `context_window` del composite): si la caché sube de 44% a
   ~85%, se libera la mitad del trabajo de la tarjeta para generar. Experimento CTX-8192 en
   curso en un kernel aparte, con umbrales pre-registrados. Validable **gratis** (el log
   imprime acierto y tokens/s) — sin gastar envío.
2. **Amplificación por programas**: que un turno de pensamiento ejecute muchas acciones. El
   entorno ya permite `action([...])` con listas y bucles; el modelo casi nunca lo usa. Nuestra
   navegación de la Fase 3 (modelo de movimiento aprendido + búsqueda en anchura) inyectada por
   el mismo mecanismo de `schema_helpers` convierte un turno en una secuencia completa.
3. **Decodificación especulativa** (hoy apagada): con n-gramas no requiere modelo extra ni
   dataset nuevo. Acelera la escritura, que es la parte menor del costo → tercera prioridad.

**Palancas cerradas con evidencia** (no reabrir sin dato nuevo): más exploración bruta (0.25
contra ~1.0 del agente con lenguaje: la calidad por acción vale ~200× el volumen); bajar la
temperatura como fin en sí mismo (el leaderboard retiene el MÁXIMO de nuestros envíos, así que
comprimir la cola alta puede perjudicar); proxies de comportamiento del agente en tarjetas T4
gratuitas (3 intentos: el harness necesita un modelo grande y rápido para siquiera arrancar).

**Método adoptado del proyecto hermano (arc-agi-2):** umbrales de decisión **pre-registrados
antes de ver el resultado**, y matar hipótesis en el banco más barato disponible antes de
gastar cuota o envíos. Ambas disciplinas nacieron de errores propios documentados.

---

## Análisis fundacional (2026-07-18)

Basado en el análisis de los 10 notebooks top de la competencia (descargados y estudiados;
resúmenes de esta fecha).

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
| hybrid | explorador (piso ~0.25) + LLM al atascarse (techo) | — | **0.25** | el LLM propio no superó el piso de exploración de forma medible |
| v10–v12 | loop agéntico propio: navegación guiada, memoria de reflexión, click_all | 9 | **0.26 / 0.25 / 0.25** | el 0.26 resultó ser ruido (ablación del mismo config dio 0.25) |
| explorer054 | réplica fiel de la referencia pública de junio (0.54) | — | **0.22** | el set oculto ROTÓ: la referencia estaba vencida |
| duck v2 | harness TAAF + injertos {efficiency, retry_guard, shortcircuit} | 3 niveles/16min | **1.17 · 1.03 · 0.76 · 0.96** | 4 muestras del MISMO código: media 0.98, rango 0.41 |
| duck v3 | + `goalkeep` (retiene modelo del mundo entre game-overs) | 1 nivel | **0.81** | cae dentro del rango de la línea base → veredicto suspendido, apagado por falta de evidencia a favor |
| duck v4 | + `schema_helpers` (funciones de análisis precargadas en el entorno del agente) | 3 niveles/16min | **1.10** (n=1) | −20% de tokens medido; helpers validados en CPU (4,6 ms contra un presupuesto de 30 s) |

**Conclusión original (v2==v1 → techo semántico de la exploración): corregida.** El 0.25 era el
techo de nuestra implementación, no de la exploración; y la exploración en el set actual rinde
0.22–0.25 incluso en la mejor referencia pública. Lo que sí quedó confirmado por otra vía: **un
harness con lenguaje que razona sobre objetivos vale ~4× frente a la exploración pura** (0.98
contra 0.25) — la hipótesis semántica era correcta, lo que faltaba era madurez del harness.

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

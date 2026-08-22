# Estrategia ARC-AGI-3

> **Documento en dos capas.** La cabecera (§0) es el **estado vigente**; lo que sigue es el
> análisis fundacional del 2026-07-18 (contrato de submission, lecciones del leaderboard,
> plan por fases) que se conserva porque su diagnóstico técnico sigue siendo válido, aunque
> varias de sus conclusiones fueron **corregidas por medición** — cada corrección está marcada.

## §0. Estado vigente (2026-08-16)

**Dónde estamos (actualizado 2026-08-22):** v4 (duck + `schema_helpers`) cerró con **5 muestras
{1.10, 0.95, 0.87, 0.93, 1.04} = media 0.978**, indistinguible de la línea base v2 {1.17, 1.03,
0.76, 0.96} = 0.98 — las nueve muestras juntas dan **0.98 de media, rango 0.76–1.17**, y esa es
la referencia contra la que se lee v6 (desplegado 2026-08-21, primera muestra en camino).
Nuestro stack propio de exploración topaba en 0.25.
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

> **Actualización 2026-08-19 — un instrumento nuevo, y un error de medición propio.** Se construyó
> un **banco micro**: preguntas sobre la mecánica con respuesta derivada del propio environment,
> cientos por minuto en CPU (el envío diario daba *un dato por noche* con varianza 0.41). Lo primero
> que midió fue **un fallo nuestro**: el detector de movimiento (`sandbox_nav._nav_shift`) alineaba
> *todo* el conjunto de celdas no-fondo y, sobre tableros densos (630–855 celdas de 4096), ajustaba
> ruido — llegaba a dar el **mismo desplazamiento para cuatro acciones distintas**, con 100% de
> *consistencia*. **La consistencia no valida nada**; sólo la **predicción fuera de muestra** lo hace.
> Corregido (huellas por color, sólo celdas cambiadas) sube a **96.6%** (141/146) con filtro de
> confianza ≥0.6. El banco de 201 items, cuya verdad salía del detector roto, quedó invalidado y se
> reconstruyó a 176. **Lección permanente añadida a las tres de abajo: un instrumento se valida
> prediciendo lo que no vio, nunca repitiéndose a sí mismo.**
>
> Estado de la palanca semántica: la carga del seam C ya produce nota **no vacía en 25/25 juegos**
> (la v1 era vacía justo en los juegos sin movimiento) y detecta **5 juegos donde ninguna acción
> simple hace nada** pero los clics sí responden — ahí el agente puede quemar la partida entera
> pulsando botones muertos.

> **Actualización 2026-08-17 — el reanálisis que reordena todo.** Cuatro experimentos después, la
> tesis "el cuello es el número de acciones" queda **refutada**: recortar la ventana de contexto dio
> +48% de acciones y **0.60** en el set oculto (peor); bajar la concurrencia dio −27% de acciones y
> menos niveles (peor); inyectar helpers de navegación dio −18% de acciones y **los mismos niveles**
> con +25% de score. Por encima de un piso duro (~18 acciones por juego, por debajo del cual ningún
> juego completó nunca un nivel) **las acciones y los niveles están desacoplados**. En el rerun de
> 8 h ya estamos muy por encima de ese piso (~94 acciones por juego), así que **la restricción
> activa no es el presupuesto sino la comprensión**: el agente no infiere la regla ni la meta, y
> más acciones no compran entendimiento. La rama de infraestructura de memoria queda **cerrada**
> (ver §8.9 de DESIGN). **Lo que sí ganamos hoy es un canal:** los *seams* de inyección están
> validados en producción — una sola línea de nota bastó para que el modelo adoptara código nuestro
> en 25 de 25 juegos (726 llamadas). El canal funciona; lo que falló fue la carga.

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

**Palancas, actualizadas al 2026-08-17:**

1. ~~Reducir la relectura (`context_window`)~~ → **CERRADA**: +48% de acciones offline pero
   **0.60** en el oculto. Recortar historial compra acciones vendiendo calidad.
2. ~~Bajar la concurrencia~~ → **CERRADA**: −27% de acciones, menos niveles. El aprovechamiento
   del lote en vLLM pesa más que el desalojo de memoria.
3. ~~Amplificación por programas~~ (helpers que el modelo debe **llamar**) → **NEUTRA**: adopción
   masiva (726 llamadas, 25/25 juegos) pero −18% de acciones, porque en los juegos sin "jugador"
   que se traslade la función devuelve vacío y cada llamada infructuosa cuesta un turno.
4. **VIVA — inyección de INFORMACIÓN por el prompt** (seam C, `_build_user_prompt`): entregar el
   modelo de movimiento y el perfil de efectividad por acción **ya calculados**, en vez de como
   función que cuesta un turno llamar. Coste cero en turnos, sirve en todos los juegos. Es la
   tesis de Fase 3 de este proyecto, ahora con el canal demostrado.
5. **VIVA — atacar lo semántico**: el agente no infiere la meta. Ideas ordenadas en DESIGN §8.5
   (diseño experimental por ganancia de información: elegir la acción que más discrimina entre
   hipótesis rivales, en vez de la que parece prometedora).
6. Decodificación especulativa (hoy apagada, sin modelo extra con n-gramas): acelera la escritura,
   que es la parte menor del costo → baja prioridad, y menos aún ahora que sabemos que el
   presupuesto no es la restricción activa.

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

## §9. Transferencias del proyecto hermano AG2 (revisión sistemática, 2026-08-22)

AG2 terminó con un mapa de palancas medido a golpe de submission (28.47→30.14% y cuatro
negativos limpios). Revisión de qué transfiere a AG3, ahora que tenemos banco e inyección:

**1. LA PALANCA QUE NO HEMOS PROBADO: asignación de presupuesto.** La mayor ganancia única de
AG2 fue *cheap-first* (+1.67): reordenar la cola para que el presupuesto rinda cobertura. Su
lección estructural: `score = cobertura × calidad` — idéntica a nuestro `niveles ≈ acciones ×
calidad` (§8.9). En AG3 los injertos de asignación **existen y están sin usar** en el fork:
- `banking` (win-then-replay): tras ganar, RESET + replay podado del trace ganador en una play
  nueva de la misma card (score = MAX sobre plays) → llegar rápido al estado ganado y seguir
  **más hondo**. No depende de clones: candidato real.
- `transfer` (replay entre clones): depende de que el set oculto sean clones de los 25 públicos.
  Tras la rotación del set (nuestra réplica 0.54→0.22), probablemente degrada a no-op declarado.
Advertencia de AG2 en la misma línea: la asignación mal hecha REGRESA (−2.08 con presupuesto
adaptativo; ambas direcciones desde su óptimo empeoraron). Probar `banking` exige el régimen
real (2h offline G4) — **candidato al gasto de Kaggle de la próxima semana, tras leer v6**.

**2. RELECTURA DEL "GAP" CON thtennant (1.28 vs nuestra media 0.97).** AG2 corrió un pipeline
byte-idéntico al notebook público 33.89 y midió 28.47/30.14/29.72/29.31: el spread de 3-4 pts
era varianza de cobertura run-to-run, y perseguirlo fue declarado pozo sin fondo. Nuestro caso
es análogo: corremos la config v12 de thtennant, nuestro máximo muestral fue 1.17, y su 1.28
puede ser un sorteo afortunado de la MISMA distribución (n=1 suyo vs nuestra media de 4).
**Consecuencia: no gastar envíos persiguiendo ese 0.3; el climbing real es nuestra línea
semántica (seam C), no la caza del fork.**

**3. PUERTA CERRADA: ensembles / segundo modelo.** Dos negativos medidos de AG2 (2B: break-even;
TRM: −2.92) con diagnóstico airtight: *un worker fuerte y limitado por cobertura quiere TODO el
cómputo; cualquier cómputo cedido a un partner más débil es neto-negativo*. En AG3: *nunca*
partir el throughput del 27B con un modelo auxiliar "para acciones fáciles" — las 28
conversaciones ya saturan la tarjeta. Idea vetada sin gastar un envío.

**4. PUERTA CERRADA: SFT sobre tareas sintéticas mecánicas.** Dos negativos medidos (v1 estilo,
v2 interferencia: exec rate a la mitad, 0 aciertos): *la distribución sintética ≠ real; componer
mecánicamente enseña la forma y desplaza el prior*. Para AG3: la idea pendiente de juegos
sintéticos vale como **held-out de evaluación**, no como data de entrenamiento; y nuestro banco
está en el lado correcto de esa línea (preguntas derivadas de juegos REALES con verdad del
environment, no tareas inventadas).

**Convergencias ya aprendidas por cuenta propia** (llegamos a lo mismo por caminos distintos,
lo que refuerza ambas): su smoke contaminado = nuestro v5 (offline miente fuera de régimen); su
truncación de MAX_NEW midiendo al evaluador = nuestro brazo inglés; su disciplina de poder
estadístico = nuestro contraste pareado con p de signo.

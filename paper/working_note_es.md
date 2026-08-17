# La exploración está capada, entender es difícil: un agente híbrido explorador–LLM honesto para ARC-AGI-3, y qué rompió y qué no el 0.25

*Working Note — ARC Prize 2026 · ARC-AGI-3 (Kaggle).*
**Autor:** Julian Camilo Villa Duque.

> **Documento vivo.** Se actualiza a medida que llegan resultados. Ver el [registro de decisiones](#apéndice-a--registro-de-decisiones-y-mediciones-clave) y el doc de diseño [`docs/DESIGN.md`](../docs/DESIGN.md). Última actualización: **2026-08-17**. Estado: harness duck (TAAF) + injertos en **1.10** oculto (media base 0.98); el cuello es **semántico, no de presupuesto** (cuatro experimentos en el Apéndice A).

---

## El problema en términos llanos (introducción de 60 segundos)

**ARC-AGI-3** mide *inteligencia fluida*: un agente debe jugar **juegos interactivos que nunca ha
visto**, descubrir las reglas actuando, construir un modelo del mundo, e inferir un objetivo que nunca
se enuncia. A diferencia de ARC-AGI-2 (puzzles estáticos entrada→salida), aquí el agente **actúa** y
aprende de las consecuencias.

- **Observación:** un grid **64×64** de colores 0–15 — una imagen (`FrameData`) que trae además el
  `state` del juego, `levels_completed`, `win_levels` y la lista por frame de `available_actions`.
- **Acciones:** `RESET`(0), seis "botones" simples `ACTION1–5,7`, y `ACTION6 = click(x,y)` en una
  celda. Los juegos se etiquetan `keyboard`, `click` o `keyboard_click`.
- **Qué se puntúa:** niveles completados en el set **oculto** (~110 juegos). No hay ejemplos resueltos
  — el agente debe *explorar y luego explotar*.
- **Reglas de cómputo:** la submission es un notebook. En el *rerun* real espera un gateway
  (`http://gateway:8001`) y juega los juegos ocultos ~8 h, **offline** (sin internet); el score sale de
  las partidas (el `submission.parquet` es un dummy). En *Save & Run* juega los 25 juegos públicos
  **offline** — nuestro banco de pruebas gratis que **no** consume submission de competencia.

![Un frame 64×64 (ejemplo sintético).](01_frame.png)

*Figura 0 — Cómo se ve un frame (sintético; no es un juego de la competencia). Un fondo, una región
grande, un "botón" pequeño de color raro, un avatar y una barra de estado pintada en el borde.*

**Por qué es difícil:** el sondeo aleatorio completa ~0 niveles en casi todos los juegos de train; y la
*exploración sistemática* — nuestra primera vía — se estanca en **0.25** en el set oculto. Ese 0.25 es
la fracción de juegos ocultos resolubles **sin entender el objetivo**. Todo lo que está por encima
exige un modelo del mundo e inferencia de objetivo. Ahí entra un LLM.

### El método en esencia — y su estructura

Nuestro agente es un **híbrido** con un piso barato y un techo semántico:

```
HybridAgent(frame):
  si el explorador no está atascado:  acción = GraphExplorer(frame)   # bankea niveles fáciles (piso 0.25)
  si no:                              acción = LLMAgent(frame)        # razona el objetivo (el techo)
```

**El GraphExplorer (piso).** Trata el juego como búsqueda sobre un **grafo de estados**. Cada frame
distinto es un nodo; una acción es una arista. El nudo de ingeniería es el **hash de estado**: los
juegos pintan contadores de paso y HUD en un borde de 3 px y animan contadores interiores, así que un
hash ingenuo hace cada frame único y el grafo explota. Hasheamos un grid **enmascarado**,

```
key(frame) = hash( grid ⊙ (1 − M) ) ,   M = borde(3px) ∪ máscara_contador_aprendida
```

donde `máscara_contador_aprendida` congela las celdas interiores que cambian en ≥80% de las
transiciones (animaciones), acotada a ≤20% del interior. Los clicks nunca se hacen por fuerza bruta
sobre 64×64; los candidatos son los objetos por componentes conexas ordenados por un score de
**button-likeness**,

```
button_score(obj) = 0.4·rareza(color) + 0.3·score_tamaño(área) + 0.3·relleno(obj)
```

(objetos pequeños, de color raro y compactos puntúan alto — botones y avatares), más una rejilla gruesa
de cobertura. Las clases de click estructuralmente inertes se suprimen (deadsig), y cuando se agotan las
acciones de un nodo, un BFS vuelve al nodo más cercano con acciones sin probar; el grafo aprendido se
reproduce tras RESET porque los juegos son deterministas.

**El LLMAgent (techo).** Cuando el explorador se atasca, decide un LLM congelado de pesos abiertos
(Qwen3-27B-FP8, servido por vLLM en la GPU RTX Pro 6000 de la competencia). Su ventaja — y nuestra
apuesta central — es que **no** le damos solo píxeles crudos; le inyectamos la **estructura objetual que
ya computamos** como texto (*"obj3 color=2 size=16 center=(46,46) button_score=1.00"*, más el efecto
numérico de la última acción), para que razone sobre **datos duros**, no sobre píxeles alucinados.

![Las features objetuales inyectadas en el prompt (frame sintético).](02_features.png)

*Figura 0b — Extracción de features sobre el frame sintético. Los objetos reciben bounding boxes y un
`button_score` (verde = alto; el botón rojo pequeño y el avatar puntúan 1.00; la región azul grande
0.28). El recuadro blanco es el borde de 3 px que el hash de estado ignora. Esta estructura es lo que el
LLM recibe como texto, junto a la imagen.*

Tres mecanismos convierten la predicción de-una-acción en un **loop agéntico**:

1. **Memoria de reflexión** — cada 15 transiciones, una segunda llamada al LLM resume el historial en
   una memoria markdown compacta (`## Rules / ## Goal / ## Avoid`) que se re-inyecta en cada prompt de
   acción siguiente. Es adaptación en test *en contexto*.
2. **Inyección de efectividad** — la `P(cambio)` observada por acción se re-alimenta como dato duro,
   para que el modelo deje de re-elegir acciones que su propia experiencia mostró inertes.
3. **Navegación guiada** — el LLM puede devolver un **sub-objetivo** espacial `goal:{x,y}`; un
   controlador con un **modelo de movimiento aprendido** (`acción → (dy,dx) medio`, leído del vector de
   movimiento de las features) lleva al avatar hacia el objetivo *sin gastar una llamada al LLM por
   paso*. El LLM razona *qué* objetivo; la búsqueda ejecuta el *cómo*.

Cualquier fallo del LLM (excepción, parseo vacío, ninguna acción legal) cae al GraphExplorer, así que el
agente nunca crashea ni se queda sin acción legal.

---

## Resumen (Abstract)

Construyo un agente para ARC-AGI-3, donde el score son niveles completados en juegos interactivos
ocultos. Mi pipeline final es un **híbrido**: un **explorador de grafo de estados** determinista que
bankea los niveles alcanzables sin entender (un techo medido de **0.25** en el set oculto), más una
**política LLM** de pesos abiertos congelada (Qwen3-27B-FP8 vía vLLM en la RTX Pro 6000) que toma el
control cuando el explorador se atasca, y cuyo input distintivo es la **estructura objetual que
computamos offline inyectada como texto en el prompt** — un lever que ninguna solución pública top usa.
Sobre el LLM añado un **loop agéntico**: **memoria de reflexión** periódica, realimentación de
**efectividad de acciones**, y **sub-objetivos espaciales propuestos por el LLM ejecutados por un
navegador con modelo de movimiento aprendido**. Esta nota está deliberadamente inclinada hacia **lo que
aprendí y lo que falló**, porque en esta tarea los resultados negativos son donde está el valor.
Mis conclusiones principales: (1) el score oculto se descompone en un **piso de exploración de
exactamente 0.25** — confirmado idéntico en tres variantes independientes de exploración (paralelismo,
reinicios diversificados) — y un **techo semántico** que exige inferencia de objetivo; (2) el LLM
*técnicamente funciona* — el diagnóstico de Save & Run muestra que lee nuestro `button_score` inyectado,
y su memoria de reflexión infiere reglas correctas al pie de la letra (*"los clicks son inefectivos, el
movimiento funciona, el objetivo es alcanzar un estado-objetivo"*) — pero **completa los mismos niveles
que el explorador ya alcanza**, así que la selección de-una-acción del LLM, por muy contextualizada que
esté, **no** rompe el piso offline; (3) la lección disciplinada — **el Save & Run offline de 30 minutos
está limitado por tiempo y es un mal proxy del rerun oculto de 8 horas; solo el score oculto es
confiable, y es 1/día** — es lo que me mantuvo honesto; y (4) una secuencia de diez builds (runner
paralelo, exploración de clicks salientes, reflexión, inyección de efectividad, navegación guiada) llevó
la tasa de fallo del LLM de **98.6% → 3.7%** e hizo al agente estrictamente más capaz y más barato (la
navegación reemplaza ~la mitad de las llamadas al LLM), mientras el **conteo de niveles offline se
mantuvo plano en 9** — un plateau que leo como limitación de tiempo compuesta con juegos genuinamente
más difíciles (click-puzzles, match-config) que la navegación por sí sola no ataca. Tres fallos de
arranque de vLLM afinaron una lección de infraestructura reutilizable: los pesos FP8 ruteados por el
autotuner de flashinfer crashean en esta GPU, resuelto forzando el kernel Marlin FP8
(`VLLM_TEST_FORCE_FP8_MARLIN=1`); y Qwen3, un modelo de razonamiento, gasta silenciosamente su
presupuesto de tokens en `<think>` salvo `enable_thinking=False`. Reporto la contribución por componente
de todo, y soy franco en que mi resultado oculto está actualmente en el **piso de exploración (0.25)**;
la pregunta abierta — que decide el score oculto pendiente de v10 — es si la eficiencia del loop agéntico
*compone en las 8 horas* donde el banco offline no puede mostrarlo.

*Nota de alcance: el razonamiento, los hallazgos de datos y todo resultado negativo se comparten
completos — ahí está el valor para la comunidad. Los prompts y constantes exactos viven en el código
versionado; los datos de la competencia (archivos de juego, pesos) nunca se commitean, por privacidad.*

---

## 1. La tarea, y a qué se reduce

El objetivo son **niveles completados** en ~110 juegos ocultos en ≤8 h. Dos hechos estructurales lo
determinan todo:

- **Determinismo.** Los juegos son (casi-)deterministas: reproducir una secuencia de acciones desde
  RESET recrea el estado. Esto hace del *grafo de estados con replay* un planificador válido y barato, y
  es por lo que la exploración logra algo.
- **La división piso–techo.** Empíricamente el score oculto es `piso + techo`, donde `piso ≈ 0.25` es lo
  que alcanza la exploración pura (niveles-1 y juegos poco profundos) y `techo` exige inferir el
  objetivo. Confirmar que el piso es *plano* entre variantes de exploración (§4) es la medición más
  importante — me dijo que la exploración era un sub-problema resuelto y capado, y que la inversión debía
  moverse a entender.

## 2. Hallazgos sobre los juegos (los datos)

La extracción de features sobre los 25 juegos públicos (`src/arc3/features.py`, volcado en
`features_out/`) da la estructura sobre la que se para todo el agente:

- **Tags de modalidad.** Cada juego es `keyboard`, `click` o `keyboard_click`. `available_actions` da lo
  mismo por frame, pero el tag es un prior global sobre si importan flechas o clicks.
- **Esparsidad de respuesta `p_change`.** Fracción de acciones que cambian el frame, va de **0.02**
  (`lp85`) y **0.07** (`ft09`) — casi nada responde salvo la acción exacta correcta — a **1.0** (`ls20`,
  `tu93`) donde todo se mueve. Los juegos de `p_change` bajo son agujas en un pajar que castigan la
  acción ciega y premian el movimiento entendido y dirigido.
- **Objetos y button-likeness.** Componentes conexas excluyendo el fondo (color mayoritario); los
  pequeños, raros y compactos son los elementos interactivos. Esta única heurística dirige tanto los
  candidatos de click del explorador como el targeting del LLM.
- **El borde es HUD.** Contadores y estado viven en el borde de 3 px y en celdas interiores animadas;
  enmascararlos para el hash de estado es la diferencia entre un grafo que converge y uno que explota. Es
  el hallazgo de datos más determinante.
- **El azar no basta.** Bajo un sondeo aleatorio de 300 acciones el nivel máximo alcanzado es 0 en 24/25
  juegos (solo `r11l` llega a 1) — confirma que se requiere estructura, no suerte.

## 3. Método (cada etapa tiene una función)

1. **Features (`features.py`)** — objetos, button_score, diffs de transición, vectores de movimiento,
   perfiles de efecto por acción. Numpy puro, offline, mismo código local y en el kernel.
2. **GraphExplorer (`agent.py`)** — grafo de estados con hash enmascarado, candidatos de click
   salientes, supresión deadsig, BFS-a-pendientes, replay determinista. El piso 0.25, a coste cero de
   LLM.
3. **LLMAgent (`llm_agent.py`)** — Qwen3-27B congelado vía vLLM; imagen + texto objetual inyectado +
   memoria de reflexión + efectividad; parseo JSON robusto; cola de plan; memoria de acciones inefectivas
   por estado.
4. **Reflexión** — segunda llamada al LLM periódica → memoria markdown re-inyectada (TTT en contexto).
5. **Navegación guiada** — `parse_goal` extrae `goal:{x,y}`; un modelo de movimiento aprendido
   (`acción→vector`) y una estimación de posición del avatar llevan un controlador hacia el sub-objetivo,
   evitando llamadas al LLM por paso.
6. **HybridAgent** — enruta explorador→LLM al atascarse, preservando el piso y añadiendo el techo.
7. **Runner paralelo (`runner.py`)** — pool de threads con presupuesto de tiempo compartido; en el rerun
   HTTP-latencia esto multiplica el throughput (vLLM agrupa los requests concurrentes en una GPU).
8. **Diagnóstico de Save & Run** — el notebook vuelca, por juego, el prompt→respuesta cruda→acciones
   parseadas del LLM, las reflexiones, y un desglose categorizado de fallos, para juzgar cada cambio
   sobre el comportamiento real del modelo antes de gastar una submission.

## 4. Amplitud y profundidad de la exploración (negativos valorados por igual)

Este es el corazón de la nota. Cada fila es un build; los negativos son tan informativos como los
positivos.

- **El paralelismo de exploración no movió el score oculto.** v1 (serial) y v2 (14 workers paralelos)
  ambos dieron **0.25**. El rerun es HTTP-latencia, así que la concurrencia multiplica las acciones
  ~14×; que no cambiara nada probó que el cuello era **semántico, no de cómputo**.
- **Los reinicios diversificados tampoco lo movieron.** v3 (salt que densifica rejillas de click y rota
  el orden de acciones al agotar el grafo) también dio **0.25**. Tres variantes independientes de
  exploración en el número idéntico es evidencia fuerte de un **piso duro**.
- **El LLM arranca pero no rompe el piso offline.** Después de que el primer rerun oculto del híbrido
  también diera **0.25**, el diagnóstico de Save & Run mostró al LLM funcionando bien (lee
  `button_score`, JSON válido) pero completando los *mismos* niveles — el muro es inferencia de objetivo,
  no el prompt.
- **Los ajustes de prompt mejoran robustez, no niveles.** Reflexión (v7), inyección de efectividad (v9)
  y navegación guiada (v10) llevaron la tasa de fallo del LLM **98.6% → 7.7% → 5.0% → 3.7%** e hicieron
  al agente más barato (la navegación reemplazó ~la mitad de las llamadas al LLM; `nav_used=1692` en
  v10), pero el **conteo de niveles offline se mantuvo en 9** (v6–v10). Una ganancia real de
  capacidad/eficiencia que el banco offline limitado por tiempo no puede convertir en niveles — el nudo
  de la incertidumbre en §6.
- **Negativos de infraestructura (tres arranques G4 desperdiciados, cada uno diagnóstico).** (i) vLLM
  crasheó en el autotuner de cudagraph de flashinfer → no era la causa; (ii) `--enforce-eager` lo aisló
  al kernel **GEMM FP8** (`FlashInferFP8ScaledMMLinearKernel`); (iii) `VLLM_TEST_FORCE_FP8_MARLIN=1` lo
  arrancó. Luego un bug trivial del runner (`len(agent._nodes)` — atributo que solo tiene el explorador)
  descartaba el resultado de cada juego aunque el LLM había jugado; y los tokens de razonamiento de Qwen3
  ahogaban el JSON hasta `enable_thinking=False`. Cada fallo fue barato porque el soft-deadline offline
  lo acotaba y el fallback mantenía la corrida distinta de cero.

**No construido aún (con franqueza), los levers que apunta el diagnóstico:** un **planner sobre el
simulador** (búsqueda estilo FORGE sobre el `.py` del juego, si el source es accesible en el rerun);
**LoRA-SFT** sobre trayectorias generadas por el solver (con riesgo de overfit a los 25 juegos de train);
y tipos de sub-objetivo más ricos (`click_all`, `match_target`) que extiendan el loop guiado más allá de
la navegación.

### 4-bis. Sobre LoRA y test-time training (una decisión de diseño deliberada)

Una pregunta recurrente: ¿ayudarían **LoRA** y **TTT** aquí? Mi posición razonada (detallada en
`docs/DESIGN.md`):

- **Adaptación en contexto (lo que hacemos)** — memoria de reflexión + inyección de features + memoria de
  inefectividad por estado — es la forma barata y generalizante de adaptación en test, y es lo que usó el
  agente público single-LLM más fuerte (LB 0.86). Primer lever a exprimir.
- **LoRA-SFT (offline):** *condicionalmente útil*. Podemos generar trayectorias casi-óptimas cargando el
  `.py` del juego y resolviéndolo con búsqueda en el simulador, y luego fine-tunear un adaptador pequeño
  para que el modelo sea nativamente fluido en el formato de acción e idiomas generales ("clickear
  botones", "explorar y luego explotar"). El **riesgo es overfit** a los 25 juegos de train — los ocultos
  son distintos, así que el valor es enseñar *skills generales*, no memorizar soluciones; augmentación
  fuerte y un held-out de juegos son obligatorios.
- **TTT online sobre el LLM:** *no recomendado*. El servido con vLLM no compone con updates de pesos
  online; correr entrenamiento + servido de un 27B en una GPU es caro y frágil, y el ganador del milestone
  no lo hizo. La memoria de reflexión es el "test-time training" práctico — adapta por-juego *en contexto*
  sin tocar pesos. Si de verdad se quiere aprendizaje online, una CNN ligera estilo StochasticGoose es la
  opción pragmática, pero topa alrededor de 0.35–0.46.

**Reencuadre (2026-07-27) — la pregunta de entrenamiento es hoy secundaria.** La ablación mostró que
nuestro loop en-contexto es noise-bound en 0.25, pero la restricción vinculante resultó **no** ser una
táctica de entrenamiento: nuestro *explorador modelo-del-mundo* (sin gradientes, sin GPU) da 0.25 donde
el mejor explorador público da 0.54. El aprendizaje de mayor valor a mejorar ahora es el **modelo del
mundo sin gradientes** (mejor hashing de estado, cobertura de clicks, presupuesto por juego), no ningún
esquema de entrenar pesos. LoRA y TTT quedan en reserva hasta cerrar la brecha del explorador.
Tratamiento completo en `docs/DESIGN.md §4`.

## 5. Contribución de las ideas individuales

| Idea | Dónde se midió | Contribución |
|---|---|---|
| Grafo de estados + hash enmascarado | LB oculto | **todo el piso 0.25**; sin la máscara de borde/contador el grafo no converge |
| Runner paralelo | throughput del rerun | ~14× acciones (HTTP-latencia); **sin cambio en score oculto** — probó el piso |
| Clicks salientes (button_score) | explorador + targeting LLM | el mecanismo por el que cualquiera de los dos agentes encuentra celdas interactivas |
| Inyección de features en el prompt | diagnóstico de Save & Run | **cualitativamente decisiva** — el LLM razona sobre `button_score` al pie de la letra; nuestro diferenciador |
| Memoria de reflexión | tasa de fallo, volcado de memoria | infiere reglas correctas en contexto; fallo 7.7% y `Rules/Goal/Avoid` coherentes |
| Inyección de efectividad | fallo 7.7%→5.0% | robustez; contrarresta re-elegir acciones inertes |
| Navegación guiada | fallo 5.0%→3.7%, nav_used=1692 | capacidad + eficiencia (reemplaza ~½ llamadas LLM); niveles offline planos |
| Fixes de arranque vLLM (Marlin FP8, thinking off) | éxito de boot | **habilitantes** — sin ellos el LLM nunca corre (tres arranques fallidos) |

## 6. Estimación de incertidumbre

### 6-bis. Protocolo de reducción de varianza (2026-07-27, el siguiente paso elegido)

El par v10=0.26 / v10-rerun=0.25 es una medición directa, mismo-config, de la banda de ruido oculta:
**≈0.01 (un nivel) de varianza run-to-run para una submission idéntica.** Antes de confiar en cualquier
delta futuro, reducimos y cuantificamos la varianza:

- **Fuente 1 — muestreo del LLM.** Eliminada: temperatura de decodificación en **0** (greedy), así la
  política es determinista dado el prompt.
- **Fuente 2 — RNG del agente.** El GraphExplorer **no tiene aleatoriedad** (verificado: sin
  `random`/`shuffle`); el camino LLM ahora es greedy. El agente es determinista.
- **Fuente 3 — timing/concurrencia (irreducible).** El rerun de 8 h con pool de threads y latencia del
  gateway depende del reloj: cuántas acciones recibe cada juego cambia entre corridas. Offline, un
  config determinista fijo aún varía **~±2 niveles** entre repeticiones solo por esto — el piso de qué
  tan pequeña puede ser una mejora *confiable*.
- **Protocolo.** Congelar el mejor config; dejar que el auto-envío diario acumule **N muestras ocultas
  repetidas** de esa versión congelada para estimar su media real ± banda; solo un cambio cuyo efecto
  esperado supere esa banda (aprox. **≥0.03–0.05**, es decir 3–5 niveles ocultos) vale un slot de
  submission. Es la disciplina que compró el error de v10.

- **Offline ≠ oculto.** El banco de Save & Run son 30 min / 8 workers sobre 25 juegos — **limitado por
  tiempo**. El rerun oculto son 8 h sobre ~110 juegos, así que por-juego el agente tiene mucho más tiempo;
  las ganancias de eficiencia (navegación, menos llamadas al LLM) **componen allí** y no se pueden mostrar
  offline. Los "9 niveles" offline son una cota inferior de capacidad, no el score oculto.
- **Semilla/estocasticidad.** La temperatura 0.3 del LLM y el azar de la exploración hacen ruidosas las
  corridas offline individuales; solo se confían diferencias que sobreviven entre juegos y en el score
  oculto.
- **La señal confiable es el score oculto 1/día.** Una tarea programada de Windows auto-envía el mejor
  kernel más reciente a las 20:00 hora local (justo tras el reset de las 00:00 UTC), así cada ventana se
  usa sin intervención manual.

## 7. Reflexión: entendimiento genuino vs optimizar la métrica

La tensión honesta es que nuestro LLM *técnicamente funcional* aún no le gana al explorador
*mecánicamente simple* en el set oculto. Los volcados de Save & Run son el antídoto contra el
autoengaño: muestran al modelo **infiriendo genuinamente reglas del juego** (bp35: *"clickear y=63
desplaza el grid hacia arriba; el objetivo probablemente implica limpiar todas las celdas o igualar una
configuración"*), lo que es entendimiento real — pero entender la mecánica no es lo mismo que *planear
hacia el objetivo*. El plateau no es un bug de prompt que se pueda ajustar; es el núcleo genuino y
difícil del benchmark (inferencia de objetivo y planeación multi-paso), y he intentado reportarlo con
claridad en vez de perseguir un número offline que el rerun oculto no premia.

## 8. Conclusión

La exploración es un sub-problema resuelto y capado (0.25) y entender es la parte difícil — la misma
división que el benchmark está diseñado para exponer. Construí un híbrido sofisticado y bien
instrumentado que hace que el LLM lea nuestras features objetuales, aprenda reglas del juego en
contexto, y navegue hacia sub-objetivos elegidos por el LLM, llevando la tasa de fallo de 98.6% a 3.7%
mientras mantiene un piso garantizado. Si esa capacidad se convierte en niveles ocultos — si el loop
agéntico compone en las 8 horas — es la pregunta abierta que responderá el score oculto pendiente de
v10. Si rompe el 0.25, el camino es apilar sub-objetivos más ricos; si no, el diagnóstico apunta a un
planner-sobre-simulador o LoRA-SFT como la próxima inversión.

---

## Apéndice A — Registro de decisiones y mediciones clave

| Fecha | Build / decisión | Niveles offline | LB oculto | Lectura |
|---|---|---|---|---|
| 2026-07-18 | GraphExplorer (CPU) | 24/25 | 0.25 | piso de exploración establecido |
| 2026-07-19 | v2 runner paralelo (14 workers) | — | 0.25 | HTTP-latencia; el paralelismo no lo movió |
| 2026-07-20 | v3 reinicios diversificados | 17 | 0.25 | exploración tope-capada (v1=v2=v3) |
| 2026-07-22 | Híbrido explorador+LLM (features en prompt) | 9 | 0.25 | LLM arranca, lee features; mismos niveles |
| 2026-07-22 | + Reflexión (v7) | 8 | — | reglas inferidas en contexto; fallos 5% |
| 2026-07-22 | + Efectividad (v9) | 9 | — | fallos 7.7%→5.0%; niveles planos |
| 2026-07-22 | + Navegación guiada (v10) | 9 | **0.26** | **primer quiebre por encima del piso 0.25** — el loop agéntico compone en las 8 h donde offline (plano en 9) no puede mostrarlo |
| 2026-07-24 | + sub-objetivo click_all (v11) | 9 | **0.25** | **regresión al piso** — click_all clickeaba 16 objetos a ciegas y malgastaba el presupuesto de acciones donde clickear es inerte, desplazando la ganancia de navegación de v10 |
| 2026-07-24 | + click_all guardado (v12) | 7 | **0.25** | incluso el click_all *guardado* no recuperó 0.26 → fuerza la lectura honesta de abajo: 0.26 estaba dentro del ruido run-to-run |

> **Actualización 2026-07-27 — confirmado por ablación, y un reencuadre de estrategia.** Re-enviar el
> config *exacto* de v10 (nav-sola) dio **0.25** (la primera vez dio 0.26) — una medición directa,
> mismo-config, de la banda de ruido que **confirma definitivamente que 0.26 fue varianza de semilla**.
> El loop agéntico LLM no supera 0.25. **Pero la realización más importante es un reencuadre de qué
> significa 0.25:** es el techo de *nuestro explorador*, no de la exploración. El mejor agente de
> **exploración pura** público (poby7722 v47) da **0.54** en el set oculto — más del doble que el
> nuestro — sin ML, sin GPU, solo con mejor hashing de estado, cobertura de candidatos de click,
> detección de ciclos y disciplina de presupuesto por juego. Construimos el LLM sobre la premisa de que
> "la exploración está capada en 0.25", pero esa premisa era falsa: nuestra *implementación* topó en
> 0.25; la exploración tiene un headroom probado de 0.54 a **coste cero de GPU**. El pivote de mayor
> valor esperado no es entonces un lever LLM más fuerte, sino **cerrar la brecha de exploración hasta la
> referencia pública de 0.54** (solo-CPU, sin cuota, con la referencia en mano) — y luego poner el LLM
> solo sobre lo que la exploración genuinamente no alcanza. Es la decisión que fuerza la ablación.

> **Actualización 2026-07-26 — la corrección honesta: 0.26 estaba dentro del ruido.** En **siete**
> submissions LLM/híbrido el score oculto es **0.26 una vez (v10) y 0.25 seis veces** (v1–v3, híbrido,
> v11, v12). Con métrica discreta (~0.01 ≈ un nivel oculto de ~110), temperatura 0.3 del LLM y azar de
> exploración, un solo 0.26 entre siete corridas es **más consistente con varianza run-to-run que con un
> quiebre reproducible** — que el click_all guardado (v12) no lo recuperara es la evidencia decisiva.
> Sobre-leí v10=0.26 como "romper el piso"; la conclusión disciplinada es que el **loop agéntico LLM no
> ha superado 0.25 de forma robusta**, y los sub-objetivos incrementales producen diferencias que la
> banda de ruido de ±un-nivel se traga. Es justo la disciplina de varianza-por-semilla que el género
> exige ("la varianza run-to-run se traga la mayoría de las mejoras"): **con el signal-to-noise actual,
> una submission 1/día no distingue estas variantes.** Consecuencia estratégica: dejar de gastar slots
> diarios en tweaks de sub-objetivo a nivel de ruido; el próximo movimiento debe ser un lever
> *cualitativamente más fuerte* cuyo efecto esperado supere la banda de ruido de un-nivel (planner sobre
> simulador si el source del juego es accesible en el rerun, o LoRA-SFT), o un protocolo explícito de
> reducción de varianza (repetir la misma config para estimar la media real antes de confiar en un
> delta).

> **Actualización 2026-07-24 — un negativo limpio y su corrección.** v11 (añadiendo un sub-objetivo
> `click_all` ciego) dio **0.25**, *por debajo* del 0.26 de v10 — una regresión real al piso de
> exploración. La métrica es discreta (~0.01 ≈ un nivel oculto), así que 0.26→0.25 significa que
> click_all *perdió* el nivel que la navegación había ganado: su controlador cometía hasta 16 clicks por
> invocación sin verificar efecto, quemando el presupuesto de acciones en juegos donde clickear es
> inerte. La lección es la misma que enseñó el A/B de exploración: **una excursión sin guardar es
> net-negativa en una tarea donde el presupuesto de acciones es la restricción vinculante.** Corrección
> (v12): la excursión click_all aborta en cuanto un click no produce cambio de frame, así solo puede
> ayudar; la navegación — el lever probado de 0.26 — queda intacta.

> **Actualización 2026-07-23 — aparente quiebre del piso (luego revisado a la baja como ruido; ver
> 2026-07-26).** v10 dio **0.26** en el set oculto, la primera lectura por encima de 0.25 en seis
> submissions, que en el momento pareció el loop agéntico componiendo en las 8 h. Dos submissions
> posteriores (v11, v12) volvieron ambas a 0.25, y la re-lectura honesta — un solo 0.26 entre siete
> corridas en métrica discreta — es que **0.26 estaba dentro de la varianza run-to-run, no un quiebre
> reproducible** (ver la actualización 2026-07-26). Conservo esta entrada para dejar registro del error:
> sobre-leí un punto favorable antes de conocer la banda de ruido.

| 2026-08-10 | **Réplica fiel del pipeline público 0.54** (harness Swarm oficial + Explore2 vendorizado, con atribución) | — | pendiente | el pivote auditado: nuestro 0.25 era el techo de *nuestra implementación*; la réplica cierra la brecha de harness (todos los juegos concurrentes, 8 h c/u) y las de algoritmo (reset del contador al descubrir, sin deadsig, likeness plano fill/(1+size)) |

> **Actualización 2026-08-10 — auditoría de estrategia y la submission réplica.** Dos negativos
> operativos y un pivote: (1) la tarea de auto-envío diario **falló en silencio ~2 semanas**
> (`$ErrorAction=Stop` abortaba antes de loguear; `kaggle` fuera del PATH de la tarea) — slots
> perdidos; ya robustecida con try/catch, logging completo y resolución del ejecutable. (2) La
> **auditoría de la estrategia sin-gradientes** pedida por el usuario confirmó la dirección (evidencia
> pública: exploración sin gradientes 0.54 > RL-online CNN 0.35–0.46) pero marcó el fallo de proceso:
> declaramos "exploración capada en 0.25" **sin calibrar contra la mejor referencia pública**, y
> pivotamos al LLM sobre esa premisa falsa. Regla nueva: replicar la referencia antes de declarar un
> techo. (3) La **idea del usuario de juegos sintéticos** se adopta como lever de validación: los
> environment files son subclases `ARCBaseGame` en python puro, así que podemos generar variantes como
> held-out de generalización para iterar sin gastar slots. La réplica fiel del pipeline 0.54 (kernel
> `arc-agi3-explorer054`, CPU) queda enviada; su score oculto real es la nueva base sobre la que el
> stack LLM se re-monta solo donde la exploración se agote de verdad.

| 2026-08-10 | Réplica 0.54 enviada → **0.22** | — | **0.22** | ¡el set oculto CAMBIÓ (~1-jul)! El 0.54 era del set de junio; en el actual la exploración rinde 0.22–0.25 y nuestro explorador ya era mejor que la referencia. Lección: calibrar contra referencias VIGENTES |
| 2026-08-11 | **Réplica duck v12-fork enviada** (TAAF + grafts, cluster 1.5 del LB actual) | 4 niveles/16min | **1.17** | LB recalibrado: top 1.86, cluster 1.5–1.7 = forks del duck evolucionado. Validación G4: vLLM 200 tok/s, solver jugó 25 juegos. La submission más importante hasta ahora |

> **Actualización 2026-08-11 — dos hallazgos que reordenan la estrategia.** (1) La réplica fiel del
> "0.54" dio **0.22**: el set oculto rotó tras el milestone de junio — la exploración pura rinde menos
> en el set actual, nuestro 0.25 ya era mejor que la referencia, y la "brecha a 0.54" era un número
> stale. Segunda instancia de la misma lección meta: **toda referencia tiene fecha; calibrar contra la
> vigente**. (2) El LB actual (top 1.86, cluster denso 1.5–1.7 con fechas de agosto) está dominado por
> forks del duck harness TAAF evolucionado. Adaptamos nuestra infraestructura duck al fork v12
> (bundle thtennant con taaf-grafts), validamos en la G4 (solver completo jugando, 4 niveles en la
> ventana corta de 16 min) y la enviamos. Trigger diario apuntado al duck. Sobre esa base (~1.5
> esperado) se re-montan nuestros diferenciadores.

> **Actualización 2026-08-11 (más tarde) — el duck marcó 1.17: el salto de estrategia es real.** La
> réplica duck v12-fork dio **1.17 — 4.7× sobre todo lo que logró nuestro stack propio (0.25–0.26)** y
> esencialmente el número del ganador del milestone de junio (TAAF stock = 1.21), pero sobre el set
> oculto *actual, más difícil*. Quedó **por debajo del cluster 1.5–1.7**. El log de la validación
> confirma que los grafts SÍ se instalaron (`TAAF_GRAFTS FEATURES={efficiency,retry_guard,
> shortcircuit} API_VERSION=1`, sin línea de fallo). Conclusión firme: **un harness LLM que razona
> sobre objetivos vale ~5× frente a la exploración pura**, y nuestra plataforma (G4 + vLLM + receta
> de arranque Qwen3-27B-FP8) reproduce de forma demostrable la clase de resultado del ganador del
> milestone.

| 2026-08-11 | Brecha re-diagnosticada leyendo el notebook de referencia: el v12 de thtennant corre flags **idénticos a los nuestros** | — | — | la brecha 1.17→1.5 NO es config. El duck tiene varianza alta entre corridas (Tufa mismo anota que la versión legible "no tuvo la misma suerte" que su 1.21) y el cluster 1.5–1.7 es el máximo de N envíos diarios — nuestro 1.17 es una muestra |
| 2026-08-11 | **duck v3 = + `goalkeep`** (siguiendo el v18 de thtennant, publicado el mismo día) | 1 nivel/15min; banner `[goalkeep] armed` ✓ | pendiente | goalkeep corrige un defecto medido: el harness stock borra el modelo del mundo del agente en cada game-over/cambio de nivel (no-vacío solo 33/481 turnos) e inyecta un digest por turno de resultados MEDIDOS (tasa de cambio por acción, niveles, cadencia de game-over) — convergente con nuestra tesis de inyección de effectiveness de Fase 3. Install blindado: peor caso = config v12 (1.17). Validación: 25 juegos jugados, 198 tok/s; el slot de las 8pm de hoy la lleva |

> **Actualización 2026-08-11 (resolución de la brecha + goalkeep).** Bajar los notebooks reales de
> thtennant resolvió la pregunta: su v12 (la referencia del cluster) habilita exactamente nuestros
> tres flags — la réplica era fiel, y la distancia residual 1.17-vs-1.5 es **varianza de muestreo más
> selección best-of-N**, no configuración. Consecuencia estratégica: con un harness de varianza alta,
> cada slot diario es un boleto de lotería a la media actual; el camino hacia arriba es (a) sacar una
> muestra cada día (el trigger de las 8pm ya lo hace) y (b) adoptar cambios que muevan la MEDIA. El
> primero de esos cambios es gratis: thtennant publicó hoy el v18, que añade el graft `goalkeep` —
> retiene el modelo del mundo del agente a través de game-overs e inyecta estadísticas medidas de
> resultado por acción en cada turno. Eso es *precisamente* la tesis diferenciadora que construimos en
> Fase 3 (inyección de action-effectiveness), ahora implementada dentro del harness fuerte. Nuestro
> duck v3 la habilita; el slot de esta noche la lleva.

| 2026-08-12 | Barrido del LB (el usuario notó movimiento; verificado que NO hubo rescore de nuestro 1.17 — idéntico en el API de submissions y en el CSV completo del LB, rank 266) | — | — | nuevo líder destacado: **cstl 2.52** (privado, 23 envíos), despegándose del 1.86 de Kojima (65 envíos). Recalibración clave: thtennant mismo (autor del goalkeep, 25 envíos) está en **1.28**, el equipo de poby en 1.21, Tufa en 1.62; la banda 1.1–1.3 tiene ~330 equipos = la masa duck-base, donde nuestro 1.17 de una muestra cae normal. Lectura best-of-N confirmada: una corrida del duck ≈1.1–1.2, el mejor-de-25 de thtennant = 1.28. La banda 1.5–1.7 (~35 equipos) = variantes evolucionadas/afinadas, no forks planos del v12 como asumimos al inicio |
| 2026-08-16 | **HALLAZGO MAYOR — el cuello no es pensar, es RELEER** (auditoría del log del servidor de inferencia, gratis) | entrada 1.950–5.775 tok/s vs salida 110–236 tok/s (**26:1**); acierto de caché de prefijos **43,6–45,1%**; memoria de atención 177.968 tokens para 28 conversaciones de hasta 32.768 (**sobresuscrito ~5×**); decodificación especulativa APAGADA | — | por cada token que el agente escribe, el servidor relee ~26. En un diálogo que solo crece la caché debería acertar 85–95%; la nuestra acierta 44% → **más de la mitad del trabajo de lectura es recálculo desperdiciado** por desalojo de memoria. Y empeora con el tiempo: los historiales crecen, el desalojo aumenta, el agente se frena justo cuando está cerca de completar un nivel. Palanca: reducir la ventana de contexto (perilla `context_window` del composite, ya soportada). **Validable GRATIS**: el log imprime acierto y tokens/s, sin gastar envío |
| 2026-08-16 | **Experimento CTX-8192 lanzado** (kernel aparte `arc-agi3-duck-ctx`, no toca el trigger) — **umbrales PRE-REGISTRADOS antes de ver el resultado** | línea base = validación v4 del 15-ago: 328 acciones, 3 niveles, 195,25 tok/s generados, caché 44% | pendiente | **VERDE** si caché ≥70% **y** generación ≥+30% (≥254 tok/s) **y** acciones ≥262 (no caer >20%) → adoptar en duck v5. **AMARILLO** si caché sube pero acciones caen >20% → probar 16.384 (término medio). **ROJO** si la caché no mejora → la causa no es el desalojo sino la inestabilidad del prefijo (contenido dinámico al principio del prompt) → arreglo distinto: reordenar el prompt para que lo variable quede al final |
| 2026-08-17 | **v5 (ctx=16384) = 0.60 en el set oculto — REVERTIDO a v4. Ley de Goodhart en carne propia** | 0.60 queda **por debajo de TODO el rango del baseline** {0.76–1.17} | **0.60** | **El defecto de diseño, declarado:** la validación offline dura 16 min y genera ~9.500 tokens por juego; el rerun oculto genera **~52.000**. Mi experimento nunca ejercitó el régimen donde vive el problema — midió el arranque, no la carrera. Con 16k de ventana el agente gana acciones (+48% medido, dos corridas) pero **pierde memoria de trabajo justo cuando los historiales se alargan**, que es donde se completan los niveles. **Optimicé el proxy (acciones) y perdí el objetivo (niveles): Goodhart puro.** Corrección al análisis del 16-ago: la ecuación no es "puntaje ∝ acciones" sino **puntaje ∝ acciones × calidad por acción**, y recortar contexto compra lo primero vendiendo lo segundo. **Regla de reversión refinada** (distingue este caso del de `goalkeep`): revertir con n=1 solo si la muestra cae **fuera** del rango observado de la alternativa (0.60 fuera de {0.76–1.17} → revertir; el 0.81 de goalkeep caía dentro → esperar). Trigger devuelto a v4 y verificado en seco |
| 2026-08-17 | **Validación 1 h del ducknav: la INYECCIÓN FUNCIONA y el modelo la ADOPTA masivamente; la amplificación no paga (aún)** | `NAV_HELPERS injected: 7601 chars` ✓ · **726 llamadas a `plan_moves` en 25 de 25 juegos** (+6 a `player_pos`, 0 a `motion_model`) · score medio **1.06 → 1.32 (+25%)** · niveles **9 → 9** · acciones **1420 → 1159 (−18%)** | — | **Puerta 1 (mecanismo): PASA de forma aplastante.** Nuestro código propio corre dentro del harness público más fuerte, y el modelo lo usa en todos los juegos guiado por una sola línea de nota. Esa capacidad —inyectar lo que queramos en el razonamiento del agente— es el activo real de hoy. **Puerta 2 (amplificación): FALLA.** 726 llamadas produjeron solo 1159 acciones (~29 llamadas por juego para ~46 acciones): `plan_moves` devuelve rutas cortas o vacías la mayor parte del tiempo, porque **muchos juegos no tienen un "jugador" que se traslade** (son de click, de cambio de color, de configuración) y ahí `motion_model()` no encuentra nada consistente. Cada llamada infructuosa cuesta un turno de herramienta. **Puerta 3 (sin daño): PASA** — mismos 9 niveles y score medio +25%, aunque con n=1 offline eso no decide. **Diagnóstico del rediseño:** el error de concepto fue entregar la información como *función que hay que llamar* en vez de como *dato que ya está en el prompt*. La v2 debe **inyectar el modelo de movimiento medido como TEXTO en el prompt** (coste cero en llamadas) parcheando `_build_user_prompt` — que es exactamente nuestra tesis original de Fase 3 (inyección de efectividad de acciones), ahora con el seam ya demostrado |
| 2026-08-17 | **AMPLIFICACIÓN construida: helpers de navegación propios inyectados en el sandbox del agente** (`src/arc3/sandbox_nav.py`, 7.601 chars de prelude) | tests locales en CPU: compila y corre bajo **SAFE_BUILTINS restringidos**, aprende `{UP:[-1,0], RIGHT:[0,1]}` de transiciones simuladas, **descarta acciones erráticas y las que no mueven nada**, `plan_moves(-3,2)` → ruta correcta de 5 pasos, respeta el límite, degrada a vacío sin transiciones | pendiente | **la única palanca viva tras cerrar la rama de memoria.** Tres funciones puras: `motion_model()` (qué tecla mueve cuánto, MEDIDO de las propias transiciones del agente), `plan_moves(drow,dcol)` (ruta lista para `action(...)`) y `player_pos()`. Convierte en una llamada lo que el modelo redescubre a mano en cada juego, y sobre todo **permite ejecutar una ruta completa en UN turno** en vez de una acción por turno — ataca directamente los 556 tokens por acción. Seam: `schema_helpers` lee `SANDBOX_HELPERS_PRELUDE` y `HELPERS_PROMPT_NOTE` en cada llamada, así que basta extenderlos por monkeypatch (inyección en base64, protegida con try/except → peor caso = config v4). **El test local cazó un bug real antes de gastar GPU**: el detector de desplazamiento seguía al FONDO en vez del objeto y devolvía el signo invertido (`RIGHT` como `[0,-1]`); corregido excluyendo el color mayoritario. Umbrales pre-registrados para la validación de 1 h: **(1) mecanismo** — ≥20 llamadas a los helpers en las transcripciones (si el modelo los ignora, ROJO y hay que rediseñar la nota); **(2) amplificación** — acciones ≥1420 (control); **(3) sin daño** — niveles ≥7 (control 9 menos ruido) |
| 2026-08-17 | **Concurrencia 14 = ROJO decisivo; y nace un instrumento de medición de verdad** | control (28 juegos, 1h): **1420 acciones, 9 niveles, 25/25 juegos sobre el umbral, 203,9 tok/s, score medio 1.06** · prueba (14 juegos, 1h): **1042 acciones (−27%), 7 niveles, 14/25 sobre el umbral, 182,5 tok/s (−10%)** | — | bajar la concurrencia empeora TODO: menos secuencias en paralelo = peor aprovechamiento del lote en la generación, y esa pérdida supera con creces lo que se gana en desalojo. **Con esto se cierra la rama de infraestructura de memoria**: recortar contexto daña la calidad (0.60 oculto), recortar concurrencia daña el caudal, y subir `gpu-memory-utilization` de 0,90 a 0,95 solo daría ~+11% de memoria de atención frente a una sobresuscripción de 5× (verificado: el comando de arranque del bundle no fija esa opción, así que usa el default). **El 44% de acierto de caché es el precio estructural de correr 28 conversaciones largas en una tarjeta.** Hallazgo colateral que vale más que el experimento: **la validación de 1 hora con la configuración de producción es el instrumento que faltaba** — 9 niveles y 25/25 juegos activos dan señal real donde la de 16 min daba ruido. Todo futuro cambio que toque historial se mide así |
| 2026-08-17 | **Nuevo experimento: CONCURRENCIA en vez de ventana** — atacar el desalojo sin quitarle memoria al agente | dos brazos en paralelo (Kaggle admite 2 sesiones GPU), ambos con contexto 32.768 y **ventana offline de 70 min** para al fin ejercitar historiales largos: control con 28 juegos simultáneos vs prueba con 14 | pendiente | con 28 conversaciones la memoria de atención da 6.356 tokens por juego para contextos de 32.768 → sobresuscrito 5×. Con 14, cada juego dispone del doble **sin recortarle una sola línea de historial**. Aritmética esperada: menos desalojo → más generación agregada → más tokens por juego aunque cada juego corra menos horas de pared. Umbrales pre-registrados: **VERDE** si generación ≥+25% **y** niveles ≥ control; **ROJO** si los niveles caen (misma trampa que acabamos de aprender: las acciones solas no bastan) |
| 2026-08-16 | **duck v5 validado — SEGUNDA muestra del mismo config, aún mejor** | acciones **328 → 505 (+54%)**, juegos sobre el umbral de 18 **8 → 12**, niveles **3 → 4**, generación 250,6 tok/s; banner `{"context_window":16384, efficiency, retry_guard, schema_helpers, shortcircuit}` ✓ | — | dos corridas independientes de ctx=16384 dan 468 y 505 acciones (media 486, **+48%** sobre 328) y niveles 2 y 4 contra 3 de la base → **confirma que las acciones son señal estable y los niveles offline son ruido**. El disparador de las 20:00 ya apunta a v5; la protección funcionó (kernel_versions volvió a v4 durante la validación y saltó a v5 al marcar COMPLETE) |
| 2026-08-16 | **CTX-16384: la palanca funciona — +43% de acciones** (adoptada como duck v5, con anulación EXPLÍCITA de un criterio pre-registrado que resultó estar confundido) | acciones **328 → 468 (+43%)**, generación **195 → 256 tok/s (+31%)**, juegos que cruzan el umbral de 18 acciones **8 → 10**, presión de memoria 75% → 43% · caché de prefijos **44% → 38%** (¡baja!), niveles 3 → 2 | — | **Por qué anulo mi propia regla:** el criterio primario era "caché ≥70%" y falla — pero el criterio estaba CONFUNDIDO en ambas direcciones: (a) el 91% del CTX-8192 era artefacto de la tormenta de reintentos (la misma petición fallida repetida acierta siempre), (b) al recortar más el historial, el prefijo se desplaza y **invalida la caché**, así que menos ventana puede bajar el acierto y aun así reducir el trabajo total. Las métricas causalmente cercanas al resultado —acciones y generación— sí mejoran, y el **total de acciones es una métrica ESTABLE** entre corridas (336, 328, 284 en tres configuraciones previas; 468 queda muy fuera de ese rango), a diferencia de los niveles, que en 25 juegos y 16 minutos son puro ruido (el 3→2 es un reshuffle: cd82/sb26/tn36 pierden, lp85/su15 ganan). **Lectura pre-registrada para el set oculto:** v5 necesita 3 muestras; media ≥1,05 → se queda; ≤0,85 → revertir a v4; en medio → seguir acumulando. Costo asumido y declarado: v4 se queda con n=1 (1.10) |
| 2026-08-16 | **CTX-8192: el mecanismo se CONFIRMA y el agente se ROMPE** — resultado AMARILLO por la regla pre-registrada | servidor: caché **44% → 91,0%**, generación **~170 → ~500 tok/s** (×2,9), lectura 4.500 → ~800 tok/s, memoria de atención 75% → 14% (sin desalojo) · harness: **328 → 1 acción**, 189.616 → 1.123 tokens, 0 niveles | — | **la hipótesis del prellenado era correcta**: quitar el desalojo triplica la generación. Pero 8.192 está por DEBAJO del piso viable del harness: el prompt de sistema + esquema de herramientas + imagen + reserva de respuesta ya no dejan sitio para el historial → tormenta de reintentos (el servidor generaba 500 tok/s que el harness descartaba: 25 peticiones activas produciendo basura). **Lección metodológica de primer orden: las métricas del servidor miden el vehículo, no el viaje.** Sin el piso de acciones pre-registrado habría cantado victoria con un 91% de caché mientras el agente no jugaba. Siguiente paso pre-registrado: **16.384** (mismos umbrales; si las acciones vuelven a colapsar, el piso del harness está por encima y la ruta de la ventana se cierra → pivotar a reordenar el prompt) |
| 2026-08-16 | Proxy #3 (temp, con los 5 fixes) corrió LIMPIO pero **sin poder discriminar**: ambos brazos 4 acciones, 0 niveles, 1.7 tok/s | A(0.6): 4 acc, 22 tracebacks, 2 fallos de analyzer · B(0.2): 4 acc, 52 tracebacks, 3 fallos | — | los fixes hicieron su trabajo (server vivo, temperatura sí aplicada, métricas limpias) y aun así el experimento no tiene poder: 2 de 3 juegos generaron **0 tokens**. Diagnóstico estructural, no bug: **el harness TAAF está construido para un 27B rápido** (contexto 32k, transcripts largos de tool-calling, timeout de 30 s por herramienta); un 4B en T4 no lo puede conducir. **Cierro la línea de proxear COMPORTAMIENTO del agente en T4** tras 3 intentos — el instrumento real de A/B es el stream de submissions diarias (27B real, set oculto real, 1 muestra/día) |
| 2026-08-16 | **Validación local de los helpers de v4** (`scripts/test_schema_helpers.py`, CPU, gratis): ¿el graft ESTORBA dentro del sandbox de 30 s? | `connected_components` 64×64 = **4.6 ms**, `grid_diff` = 0.8 ms, 4-conectividad y bbox correctos | — | la apuesta de v4 es mecánicamente sana: los helpers cuestan ~0.02% del presupuesto del sandbox. También verifiqué el seam: el prompt dice que el grid numérico "no se expone", pero el `FrameView` del sandbox sí guarda `_grid` (python_tool_sandbox.py:133) y los helpers leen ese atributo privado — por eso funcionan. Este tipo de chequeo (CPU, segundos) es lo que reemplaza a los proxies de T4 |
| 2026-08-16 | **duck v4 (v12 + schema_helpers) muestra #1 = 1.10** vs baseline media 0.98 {1.17,1.03,0.76,0.96} | banner `[schema_helpers] armed` en la G4 ✓ | **1.10** | primer punto de v4 por encima de la media del baseline, pero DENTRO de su rango — n=1 no decide nada (lección del 1.17 y del 0.81 aprendida). Se necesitan 3–4 muestras; el trigger las produce solas. Nota de diseño de proxies: **thinking ON/OFF NO es proxeable en T4** — allí el thinking satura el throughput (timeouts) y "OFF gana" sería un artefacto de la T4 lenta, no una verdad de la G4 a 198 tok/s; la temperatura sí es proxeable porque cuesta lo mismo en tokens |
| 2026-08-15 | v2 muestra #4 (trigger) = **0.96** → baseline v12 con 4 muestras: **{1.17, 1.03, 0.76, 0.96}**, media 0.98, mediana ~1.0, rango 0.41 | — | **0.96** | el baseline queda caracterizado: distribución centrada en ~1.0 con cola alta ocasional. Desde esta noche el trigger envía v4 (schema_helpers) — su distribución se compara contra ESTA, no contra el 1.17 afortunado |
| 2026-08-14 | Proxy #2 (temp 0.6 vs 0.2) = **INVÁLIDO**, con dos bugs descubiertos que corrigen al proxy #1 | brazo A degradado (2.4 tok/s; cd82: 10 acciones), brazo B: 0 tokens, 33 tracebacks | — | (1) el server vLLM murió entre brazos → B corrió contra un puerto muerto (fix: health-check + reinicio por brazo); (2) `_LOCAL_ANALYZER_TEMPERATURE` se congela como global al primer import → la env var entre brazos NO cambiaba nada (fix: parchear el global, mismo seam que context_window); (3) **corrección al proxy #1**: la métrica `helper_calls` contaba las menciones de la nota HELPERS del prompt (4 por turno en los transcripts) — el "104 vs 0" NO prueba adopción del modelo (fix: filtrar las firmas literales de la nota). Lo que sí sobrevive del #1: tokens −20% en el brazo B a igual wallclock (métrica del summary, no contaminada) y el mecanismo del graft probado (banner, prelude, sin crashes). Señal débil nueva: thinking OFF multiplica acciones cuando el server responde (cd82: 10 vs 1–4). Rerun del #2 con los 3 fixes cuando las cuentas T4 recuperen ventana |
| 2026-08-14 | v2 muestra #3 (trigger automático) = **0.76** → config v12 con 3 muestras: {1.17, 1.03, 0.76} | — | **0.76** | media 0.99, rango 0.41 (≈41 niveles) con código IDÉNTICO. Dos correcciones: (1) **el 1.17 fue un draw afortunado**, no el valor típico — la "base 1.17" real es ~1.0 de mediana con cola alta; (2) **goalkeep (0.81) cae DENTRO del rango del v12** — la conclusión "goalkeep daña" fue prematura con n=1; queda apagado por falta de evidencia a favor, no por evidencia en contra. Meta-lección: con esta varianza, comparar configs exige varias muestras por config — el LB retiene el máximo (la cola alta importa para el score; la media/mediana para decidir). schema_helpers sigue siendo la mejor apuesta: su −20% de tokens = más turnos por juego en 8 h = mover la media por mecánica, no por suerte |
| 2026-08-13 | **Proxy Colab T4 COMPLETADO** (4º intento; fixes: torchaudio CUDA-mismatch, flag vllm removido, asyncio en hilo): A/B floor-F vs +schema_helpers, Qwen3-4B, 6 juegos × 7 min/brazo | A: 0 helper_calls, 15263 tok, 0 tracebacks · B: **104 helper_calls**, 12153 tok (−20%), 3 tracebacks | — | **señal principal: el modelo ADOPTA los helpers masivamente** (104 llamadas vs 0) sin reescribir plomería propia en ningún brazo; tokens −20% con el mismo wallclock. Caveat serio: la T4 rinde 14–18 tok/s (vs 198 en G4) con thinking ON → timeouts del analyzer y juegos con 0–4 acciones ("gave_up"); niveles 0 en ambos brazos — el proxy mide ADOPCIÓN y eficiencia, no capacidad. Verde para el v4 del viernes; el efecto en niveles lo dirá el set oculto. Lección para proxy #2: en T4 usar thinking OFF o menos juegos con más tiempo |
| 2026-08-13 | **Primera submission AUTOMÁTICA de la competencia**: el trigger corregido envió duck v2 a las 00:00:07 UTC (id 55469251, muestra #2 del config v12) | — | **1.03** | el fix del `-v` funcionó a la primera. Detalle: el CLI no imprime nada en éxito → la detección del log marcaba FALLO falso; robustecido verificando contra la lista de submissions. Colab T4: capacidad del tier gratis agotada ~1 h de reintentos; relanzado con cadencia horaria (12 h) |

> **Actualización 2026-08-13 — la varianza del duck, medida de verdad.** Config v12, dos muestras:
> **1.17 y 1.03** — dispersión de 0.14 (≈14 niveles) con el MISMO código. Confirma y amplía la
> lectura best-of-N: las corridas del duck caen ~1.0–1.2, el cluster 1.5–1.7 es el máximo de decenas
> de envíos (thtennant: mejor-de-25 = 1.28), y goalkeep (0.81) sigue siendo el peor dato pero la
> distancia a 1.03 ya no es tan concluyente como parecía contra 1.17. Dos consecuencias: (1) cada
> muestra diaria pesa poco individualmente — lo que importa es mover la MEDIA (schema_helpers, temp)
> y dejar que el trigger acumule; (2) **bajar la temperatura (0.6 → 0.2–0.3) sube de prioridad**:
> si la exploración no se degrada, comprimir la dispersión vale tanto como subir la media, porque
> el LB retiene el máximo pero nuestro percentil-50 define cuántos días tardamos en alcanzarlo. |
| 2026-08-12 | **Experimento proxy lanzado en Colab T4 gratis** (`scripts/colab_taaf_proxy.py`, headless vía `colab run`): harness TAAF REAL + Qwen3-4B (misma familia/thinking/temp que el 27B), A/B pareado floor-F vs +schema_helpers, 6 juegos (canario + su15/sb26), 7 min/juego/brazo | en curso | — | primer uso en AG3 de la regla AG2 "Colab = pre-filtro, Kaggle = confirmatorio". Señales que busca (relativas, no niveles absolutos): ¿el 4B LLAMA los helpers precargados?, ¿deja de reescribir su plomería (`def connected_components`)?, tracebacks del sandbox, tokens/acciones por juego. Hallazgo colateral del contrato de env del duck: corre con **temperatura 0.6 y thinking ON** — la fuente estructural de la varianza entre corridas que venimos midiendo |
| 2026-08-12 | **v4 pre-validado en CPU gratis** (`scripts/smoke_graft_install.py`): desempaqueta el benchmark real del bundle, corre `composite.install` con el flag set del v4 — banner + `[schema_helpers] armed` + solver injertado + prelude de 8 KB con los 4 helpers | PASS | — | la validación G4 del viernes queda como formalidad: la lógica del install está probada localmente (necesitó los wheels de la competencia + parche PosixPath→WindowsPath para el pickle de Linux). El usuario además ofreció Colab T4 gratis vía CLI headless (flujo documentado en el repo AG2, `docs/COLAB.md`): inútil para el 27B-FP8 en sí (16 GB, sin FP8) pero viable para tests end-to-end del TAAF con un modelo pequeño — en reserva. Regla adoptada de AG2: Colab = pre-filtro barato de hipótesis, Kaggle = instrumento confirmatorio |
| 2026-08-12 | **duck v3 (goalkeep) marcó 0.81** — caída de −0.36 vs el config v12 (1.17) | 1 nivel/15min (chequeo de arranque) | **0.81** | la primera evidencia del set oculto sobre goalkeep es NEGATIVA: 0.81 queda por debajo de TODA la banda duck-base (1.1–1.3, ~330 equipos), así que es muy improbable que sea varianza. Hipótesis: retener el modelo del mundo a través de game-overs atrinchera modelos equivocados, y el digest por turno gasta contexto. Ojo: thtennant publicó el v18 ese mismo día — tampoco tenía evidencia del set oculto. Acción (cero GPU): `kernel_versions.json` revertido a duck **v2** (config v12, ya validado) para que el trigger diario muestree el config bueno esta semana; goalkeep puede ganarse un re-test si los números del autor mejoran. El test de `schema_helpers` del viernes va sobre el config v12, NO sobre goalkeep |
| 2026-08-12 | **Cuota GPU agotada hasta el viernes** (~22 min restantes); plan de la semana: el trigger diario acumula muestras de v3 (goalkeep) — los submits no gastan nuestra cuota; viernes: duck v4 + `schema_helpers` | — | — | minando el bundle apareció nuestra tesis de feature injection YA implementada como graft sin habilitar: `schema_helpers` precarga helpers testeados (`grid_diff`, `connected_components`, `action_effect_summary`, `recent_history`) en el sandbox python del agente, porque el 27B reescribe esa plomería con bugs en cada juego. El prompt TAAF ya expone `segmentation` (objetos con color/hash/pixels/boundary/children + adyacencia). TPU (20 h disponibles) descartada: el stack es CUDA-only (vLLM Marlin FP8; el deploy target exige RTX Pro 6000) |
| 2026-08-11 (noche) | **Causa raíz del trigger diario hallada y corregida**: el submit de code competitions exige `-v <versión del kernel>` además de `-f`; el script nunca lo pasó | — | — | la tarea de las 8pm disparó puntual (exit 0) pero Kaggle siempre rechaza sin `-v` — o sea, el trigger NUNCA había enviado con éxito; todas las submissions exitosas fueron manuales. La lectura previa de ese error como "cupo diario agotado" era incorrecta (el cupo real responde "Submission limit exceeded"). Fix: `push_kernels.py` registra cada versión publicada en `kernel_versions.json`; `daily_submit.ps1` la lee y pasa `-v` (más modo `-DryRun`, verificado). v3 (goalkeep) se envió manual esta noche: **id 55445915**, pendiente |

> **Síntesis 2026-08-17 — qué produjo la línea de "optimizar la tokenización".** La pregunta del
> usuario sobre *nibbles* no se implementó tal cual (no se puede cambiar el vocabulario de un
> modelo entrenado, y el agente no lee la grilla cruda), pero **su reencuadre —comprimir lo que de
> verdad cuesta tokens— fue el disparador de todo lo demás**: auditar el servidor, descubrir que se
> releen 26 tokens por cada uno escrito y que la caché acierta 44%, y montar cuatro experimentos.
> Balance honesto de la línea: **1 diagnóstico mayor** (la física del presupuesto, §8 de DESIGN),
> **2 ramas cerradas con dato** (ventana de contexto, concurrencia), **1 capacidad nueva validada
> en producción** (los seams de inyección: 726 llamadas a código nuestro en 25/25 juegos) y
> **0 puntos de mejora en el set oculto hasta ahora**. La corrección más valiosa es negativa:
> el presupuesto de tokens **no** es lo que capa el puntaje, así que optimizarlo no paga; lo que
> capa es la comprensión del juego. Sin la línea de tokenización no habríamos podido descartar esa
> hipótesis con evidencia, y seguiríamos gastando envíos en ella.

**Trayectoria de la tasa de fallo (llamadas al LLM que cayeron al fallback):** 98.6% → 7.7% → 5.0% →
3.7% → **3.2%**.
**vLLM en RTX Pro 6000:** modelo 33.7 GiB, KV cache 45 GiB; arranca con Marlin FP8 + FLASH_ATTN +
`enable_thinking=False`.

*Este apéndice se actualiza en cada resultado nuevo; las secciones narrativas de arriba se revisan
cuando una decisión cambia la estrategia.*

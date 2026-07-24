# La exploración está capada, entender es difícil: un agente híbrido explorador–LLM honesto para ARC-AGI-3, y qué rompió y qué no el 0.25

*Working Note — ARC Prize 2026 · ARC-AGI-3 (Kaggle).*
**Autor:** Julian Camilo Villa Duque.

> **Documento vivo.** Se actualiza a medida que llegan resultados. Ver el [registro de decisiones](#apéndice-a--registro-de-decisiones-y-mediciones-clave) y el doc de diseño [`docs/DESIGN.md`](../docs/DESIGN.md). Última actualización: 2026-07-23.

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
| 2026-07-24 | + sub-objetivo click_all (v11) | 9 | pendiente | nav_used 1692→**2588** (los controladores de sub-objetivo reemplazan más llamadas al LLM); fallos 3.7%→**3.2%** |

> **Actualización 2026-07-23 — el piso está roto.** v10 dio **0.26** en el set oculto, el primer
> movimiento por encima del techo de exploración 0.25 en siete submissions. Offline estuvo plano en 9
> niveles, exactamente como predijo §6: la eficiencia de la navegación guiada (la nav reemplaza ~½ de
> las llamadas al LLM) compra más acciones útiles por juego en 8 horas, algo que el banco de 30 min no
> registra. Esto valida la dirección del loop agéntico y el plan: **apilar sub-objetivos más ricos**
> (`click_all`, `match_target`) más allá de la navegación. Pequeño en absoluto, pero convierte "¿ayuda
> el LLM en juegos ocultos?" de abierto a **sí**.

**Trayectoria de la tasa de fallo (llamadas al LLM que cayeron al fallback):** 98.6% → 7.7% → 5.0% →
3.7% → **3.2%**.
**vLLM en RTX Pro 6000:** modelo 33.7 GiB, KV cache 45 GiB; arranca con Marlin FP8 + FLASH_ATTN +
`enable_thinking=False`.

*Este apéndice se actualiza en cada resultado nuevo; las secciones narrativas de arriba se revisan
cuando una decisión cambia la estrategia.*

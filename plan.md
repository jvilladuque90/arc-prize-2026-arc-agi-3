# plan.md — nueva base y hoja de ruta hasta el cierre (2026-09-29)

## 0. Situación

| | |
|---|---|
| Nuestro mejor oculto | **3,98** (base NVFP4 v24 remuestreada), puesto ~304 de 3.485 |
| Líder | Tufa Labs **45,33**; 2º 36,73; 10º 13,40; 20º 7,99; 50º 5,49 |
| Mejor kernel **público** | `scottlegrand/taaf-flashnext-sheetu12b-0922` — **5,19** |
| Hito #2 | **2026-09-30** (queda un envío: el del día 30) |
| Inscripción / cierre final | 2026-10-26 / **2026-11-02** → ~33 envíos después del hito |

Los de arriba de 10 no publican: el techo alcanzable con código público hoy es ~5,2.

## 1. La nueva base: `sheetu12b` (5,19)

Autor Scott Le Grand, sobre el duck harness de Tufa Labs y el stack NVFP4 de keithtyser.
**Mismo modelo, mismos datasets y mismo harness que nuestra v24.** Lo que cambia:

| Capa | v24 (nuestra, 3,35 de media) | sheetu12b (5,19) |
|---|---|---|
| Modelo | Qwen3.8-Flash-Next NVFP4 (RadixArk), MTP 3 | igual |
| Servicio vLLM | 8 secuencias, contexto 32K | **16 secuencias, contexto del analizador 16K** (mismo KV de 5 GiB) |
| Imagen del tablero | escala x4 | **x12** (ARM P) |
| Historial | reenvía todos los tableros | **F1**: sólo el tablero actual lleva imagen (los viejos eran ~26% del contexto) |
| Memoria del agente | parser exacto; game over borra el modelo del mundo | **F3**: parser tolerante (`World model (revised):`), game over NO borra |
| Animación | el modelo sólo ve el fotograma final | **F13**: los fotogramas intermedios, accesibles desde el sandbox Python |
| Hoja de fotogramas | — | **F19**: una imagen con TODOS los fotogramas de la última acción, sin narración |
| Instrumentación | — | TIMING (sólo logs) |

Apagados por el autor tras medir: F2 resultado forzado, F4 guardia de bucles, F5 no-op con
HUD, F6 ACTION7, F7 libro de efectos (texto), F10 estancamiento, resumen de animación en texto.

**La lección del autor coincide con la nuestra (8.73, 8.84):** *"Frames yes, narration no."*
F13 (datos en el sandbox) movió el oculto 3,20 → 3,71; el MISMO contenido narrado como texto
por el anfitrión bajó a 2,57. Nuestras notas de texto (7 brazos de memoria, mapa cognitivo)
fallaron por la misma razón. **Regla de diseño que sale de aquí: lo que añadamos va como
DATOS consultables o IMAGEN, nunca como nota narrada en el prompt.**

## 2. Envío inicial

- `juliancamilovilla/arc-agi3-sheetu` v1 (`notebooks/sheetu_long.ipynb`): el público verbatim
  + nuestro recorte de ventana offline (sólo fuera del rerun). `scripts/build_sheetu_long.py`.
- Tarea `ARC-AGI3-SubmitOneShot` rearmada al kernel sheetu: dispara 2026-09-29 23:40Z y
  envía en cuanto abre el cupo del día 30. Es el envío del hito #2.
- Riesgo: si el Save & Run falla antes, la tarea se devuelve a `arc-agi3-nvfp4`.

## 3. Nuestras palancas que funcionaron — qué se añade

| Palanca | Resultado en el oculto | ¿Se añade? |
|---|---|---|
| Cambio de base 27B → NVFP4 | 1,59 → 3,55 | ya incluido (misma base) |
| Recorte de ventana offline | infraestructura, no puntaje | **sí**, ya añadido |
| Remuestreo (el marcador toma el máximo) | 3,55 → 3,98 | **sí**, como política de envío |
| Solver anim (animfast) | 3,27, sin efecto | no: sheetu ya trae F13+F19, mejor medido |
| 11 injertos de texto, 7 brazos de memoria, batching, visión etiquetada, mapa | ninguno transfirió | **no** |

Dicho claro: **ninguna palanca de agente nuestra ha transferido al oculto.** Lo que nos subió
fue siempre adoptar una base mejor, y eso es exactamente lo que hacemos hoy.

## 4. Palancas planeadas sobre la nueva base (AGENTS.md), rediseñadas con la regla del §1

| Brazo | Qué | Forma (datos, no narración) | Coste |
|---|---|---|---|
| **S0** | sheetu verbatim | control | — |
| **S1** | **Punto 1 — predictor de efectos en vivo** | objeto `effects` en el sandbox: `effects.p_change(action)` y por color/objeto clicado, aprendido en la partida | CPU, cero tokens si no se consulta |
| **S2** | **Punto 2 — grafo que actúa** | objeto `graph` en el sandbox: estados visitados, `graph.path_to(estado)` que devuelve la secuencia de acciones | CPU |
| **S3** | **Punto 4 — biblioteca de funciones** | las funciones Python que el modelo definió y que precedieron a un nivel ganado sobreviven al siguiente nivel en el sandbox | pocos tokens |
| **S4** | **Punto 3 — simulador del modelo** | el sandbox acepta un `step(state, action)` del modelo y le reporta su error de predicción contra el motor | muchos tokens; apuesta Paper Award |
| — | Punto 5 (votación/verificador) | descartado: a ~10 tok/s no cabe | — |

Orden: S1 → S2 → S3 → S4. S1 primero: es el más barato y ataca el denominador de la métrica.

## 5. Protocolo de evaluación (el banco NO selecciona, r = −0,351)

1. **Banco de 60 min sólo como compuerta mecánica**: la inyección se activa, nada revienta,
   tokens/acción no se disparan. Nunca para elegir.
2. **La selección se hace en el oculto**, con los ~33 envíos:
   - S0 recibe envíos intercalados durante todo el periodo (control y además remuestreo del máximo).
   - Cada brazo recibe 2 muestras; si su media queda > 0,5 por debajo de la de S0, se corta;
     si no, 2 más (4 muestras detectan +1,0 con sd 0,49).
   - Nunca dos brazos a la vez sobre la misma base.
3. En la última semana (≥ 2026-10-26) sólo se envía la mejor configuración, para explotar el máximo.

## 6. Calendario

| Fecha | Qué |
|---|---|
| 09-29 | sheetu v1 en Save & Run; tarea rearmada; plan.md |
| 09-30 | envío sheetu (hito #2) |
| 10-01 → 10-07 | S1 construido, compuerta de banco, 2+2 muestras; S0 intercalado |
| 10-08 → 10-18 | S2 y S3 |
| 10-19 → 10-25 | S4 si queda margen, o más muestras del mejor |
| 10-26 → 11-02 | sólo la mejor configuración |

## 7. Atribución

Scott Le Grand (agentfix, sheet, ARM P), keithtyser (stack NVFP4), Tufa Labs — Bessis,
Cottaar, Pressman, Smit, Tesnar, Viel (duck harness). Todos públicos en la competición;
nuestros derivados se publican abiertos (CC0/MIT-0 requerido para premio).

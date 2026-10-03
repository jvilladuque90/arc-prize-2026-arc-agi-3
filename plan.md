# plan.md — ruta hasta el cierre (actualizado 2026-10-03, tras la auditoría de mejoras públicas)

Detalle y evidencia: `docs/DESIGN.md` §8.85. Informes de los lectores: `_tmp_pub/audit/R*.md` (fuera de git).

## 0. Situación

| | |
|---|---|
| Nuestro mejor oculto | **4,79** (sheetu v1), puesto 543 de 3.618. Muestras sheetu: 3,78 / 2,49 / 4,79. Envío del 3-oct: ERROR de plataforma |
| Líderes | Tufa Labs **52,51**, Yi-Chia Chen **48,07** (privados). Puestos 3-25: 31,7-35,8 |
| Pelotón | ~480 equipos entre 20 y 40, mediana 26,9: copias de la solución pública de Franzen |
| Franzen M2 sin cambios | reproducción independiente **27,80**; copias: media **25,77**, sd **3,93** |
| Cierre | final **2026-11-02**, un envío por día |

## 1. Qué cambió respecto al plan anterior (S1-S4)

El plan anterior (S1 `action_effects`, S2 grafo, S3 funciones entre niveles, S4 simulador) se escribió sobre la base sheetu.
La auditoría muestra que **casi todo queda obsoleto**:

| Item | Veredicto con Franzen como base |
|---|---|
| S3 funciones persistentes | **Ya incluido** (alcance juego, superconjunto sin curar). Hueco: un fallo del sandbox vacía la biblioteca |
| S1 `action_effects` | Aplica con el ancla `runtime_globals["action"] = action`, pero 4 defectos propios y valor ~1,8 % de acciones |
| S2 grafo | No se puede inyectar; fue neutro; su clave de estado incluye el HUD |
| S4 simulador | Evidencia externa negativa (Carnot: 0 de 31 predicados) |

Lo que sí explica el rendimiento público (los 3 ganadores del hito lo comparten): **historia larga retenida (69K-131K) + KV en FP8
+ observabilidad (UNDO, game over honesto, barra visible) + código del agente que persiste.** Nuestra base retenía 16K en BF16.

## 2. Ruta nueva

| # | Paso | Coste | Estado |
|---|---|---|---|
| 1 | **Base = copia fiel de Franzen** (`arc-agi3-franzen-m2` v1, sin ediciones) | 35 min de GPU de Save & Run | **hecho 2026-10-03**: compuerta local 10 juegos x 25 min = **46,31** (47 niveles, 1.358 acciones, servidor listo a 523 s); enviada, ref 56803691 |
| 2 | Tarea diaria re-armada a esa base, 5 disparos/día (20:02, 20:47, 23:02, 06:32, 10:02 locales; ignora envíos en ERROR) | 0 | **hecho** |
| 3 | **Endurecimiento sin riesgo de modelo**: biblioteca de funciones a prueba de fallos del sandbox, reintento del reinicio automático, vigilante del servidor SGLang, tope al `result` | CPU; validar con compuertas mecánicas | por hacer |
| 4 | Palancas con evidencia externa: biblioteca de módulos con pruebas contra fotogramas grabados (Lord Han Solo), historia aún más profunda | GPU de compuerta | por decidir |
| 5 | Reproducible sin GPU: banco mecánico por repetición de peticiones guardadas (`*_requests.jsonl`): tokens/s, aceptación, TTFT, caché | CPU/GPU corta | por construir |
| — | Descartado hasta nueva evidencia: S2, S4, Swift 1.5, REAP, finetune de huikang, FP8 en línea y aceptación MTP relajada (afirmaciones de autor sin validar) | — | — |

## 3. Reglas de evaluación (el oculto no selecciona)

- Con sd 3,93: **4 contra 4 muestras detectan solo ~8 puntos**; 3 puntos pedirían ~27 corridas por brazo. No se promueve nada por un par de envíos.
- Una mejora entra por **mecanismo + evidencia externa + compuerta mecánica**, no por su puntaje oculto.
- Remuestrear sí es palanca: media 25,8 → E[máx de 28 envíos] ≈ 33,7; media 29,8 → 37,7 (P(máx>36) = 0,81). Puestos 3-5 hoy: 34-36.
- Cada kernel experimental se aparta del control de Franzen: nunca dos cambios a la vez.

## 4. Calendario

| Fecha | Qué |
|---|---|
| 10-03 | Franzen fiel lanzada (13:08) y **enviada hoy** (17:51Z, ref 56803691); el cupo del 3-oct estaba libre porque su envío terminó en ERROR |
| 10-04 → 10-08 | envíos diarios de la base fiel (muestras y control); endurecimiento (paso 3) con pruebas locales |
| 10-09 → 10-20 | primer brazo endurecido (2 muestras) y palanca con evidencia (paso 4) |
| 10-21 → 11-02 | solo la mejor configuración; remuestreo diario |

## 5. Atribución

Daniel Franzen (solución y parche), Tufa Labs (duck harness: Bessis, Cottaar, Pressman, Smit, Tesnar, Viel), John Pezzulli (Pennyroyal),
Mamy Ratsimbazafy y Gabriel Olympie (parches de SGLang), Intel (AutoRound), Albucino (drafter MTP), Scott Le Grand y keithtyser (base anterior).
Todos públicos en la competición; nuestros derivados se publican abiertos (CC0/MIT-0 exigido para premio).

# plan.md — estado y ruta hasta el cierre (actualizado 2026-10-08)

Evidencia: `docs/DESIGN.md` §8.85. Informes de la auditoría: `_tmp_pub/audit/R*.md` (fuera de git).

## 0. Situación (2026-10-08)

| | |
|---|---|
| Base | **Franzen M2 fiel** (`arc-agi3-franzen-m2` v1), 6 muestras ocultas: 25,01 / 25,17 / 23,39 / 27,15 / 27,63 / 25,97 → **media 25,7, sd 1,6** |
| Nuestro mejor | **27,63** (puesto 651 de ~3.700) |
| Leaderboard | Tufa 55,89, Yi-Chia Chen 55,77, 3º 42,66; #10 = 37,1; #25 = 33,8; #100 = 31,8 |
| Envío automático diario | **APAGADO** (decisión de Julian, 2026-10-08). Los envíos son manuales o de un solo disparo |
| Cierre | 2026-11-02, un envío por día → quedan ~25 |

Lectura: el pelotón de copias de Franzen ya está en 31-37 por pura suerte de cola alta. Con media 25,7 y sd 1,6 un máximo
de muchos envíos rondará 29-30, **no alcanza el top 10**. Para pelear el top 5 (≥ 40) hace falta subir la **media** de verdad.

## 1. Estado del plan original (puntos 1-5 y S1-S4)

| Punto original | Qué se hizo | Estado hoy |
|---|---|---|
| 1 / S1 predictor de efectos | `action_effects()` en el sandbox, probado con el sandbox real (8 pruebas). Sobre Franzen no aplica tal cual (ancla nueva: `runtime_globals["action"] = action`) y tiene 4 defectos propios; valor medido ~1,8 % de acciones | **aparcado**; reanclar solo si sobra tiempo |
| 2 / S2 grafo de estados | construido, probado 2 veces en banco: v1 −(bug de vocabulario), v2 **neutro** (21 vs 19 niveles, p=1,0). No se puede inyectar en Franzen | **descartado** como nota; solo valdría una variante que actúe |
| 3 / S4 simulador del modelo | sin construir; evidencia externa negativa (Carnot: 0 de 31 predicados) | **descartado** por ahora |
| 4 / S3 biblioteca de funciones | **ya viene en Franzen** (persistentes, alcance juego). Hueco: un timeout vacía la biblioteca | **H1 en curso** (arreglo) y H2 (retención con pruebas) |
| 5 razonamiento extra (votación) | descartado: no cabe en el reloj | descartado |
| Cambio de base | 3,3 → 4,8 (sheetu) → **25,7 (Franzen)**; es la única palanca que ha movido el puntaje | **hecho** |

## 2. Brazos sobre la base nueva (una variable cada uno)

Evaluación oculta: control = las 6 muestras de Franzen (media 25,7, sd 1,6). Con 3 muestras por brazo se detecta ~+4 puntos;
efectos menores no se pueden confirmar y entran solo por mecanismo + compuerta mecánica (DESIGN 8.85).

| Brazo | Qué cambia | Por qué | Estado |
|---|---|---|---|
| **H1** endurecimiento | la biblioteca de funciones sobrevive a un timeout/caída del sandbox; un `result` enorme se recorta | defectos verificados en el `ToolAgent` real | compuerta local **46,63** (control 46,31); envío de un solo disparo armado 2026-10-09 00:02Z |
| **H2** retención con pruebas | una función solo se guarda si pasa contra fotogramas grabados (precedente: Lord Han Solo 23,8) | calidad de la biblioteca: hoy "todo lo válido se queda" | por diseñar |
| **D1** historia más profunda | 8 plazas (en vez de 10) y desalojo de 24K en vez de 58K (+19 % de historia a igual KV) | los 3 ganadores públicos retienen 69K-131K; +8 de Siriki en otra base | **compuerta local NEGATIVA (2026-10-08)**: 29,76 contra 46,31; 29 % menos acciones y 17 % menos tok/s. Sin envío oculto salvo que Julian lo pida |
| **R1** banco mecánico | repetir peticiones guardadas contra el servidor: tokens/s, aceptación, TTFT, caché | única forma de validar cambios de servicio sin gastar días de envío | por construir |
| descartados | FP8 en línea, aceptación MTP relajada (afirmaciones de autor sin validar), Swift 1.5, REAP, finetune | no se pueden validar con el ruido actual | — |

## 3. Calendario (un envío por día)

| Fecha | Qué |
|---|---|
| 10-08 | H1 Save & Run (15:08). Si pasa la compuerta, envío de un solo disparo a las 20:02 locales (00:02Z del 9) |
| 10-09 → 10-11 | H1: 3 muestras. Mientras tanto, diseño de H2 y D1 |
| 10-12 → 10-18 | brazo siguiente (H2 o D1 según lo que enseñe H1) con 3 muestras |
| 10-19 → 10-27 | tercer brazo; el mejor se acumula sobre la base |
| 10-28 → 11-02 | solo la mejor configuración, envío diario (máximo de N) |

## 4. Reglas

- Nada de notas narradas en el prompt ("frames yes, narration no"); lo que se añada va como datos o código.
- Un cambio por brazo; toda GPU se anuncia y se autoriza antes de lanzarla.
- El banco local no selecciona; solo es compuerta mecánica.

## 5. Atribución

Daniel Franzen (solución y parche), Tufa Labs (duck harness), John Pezzulli (Pennyroyal), Mamy Ratsimbazafy y Gabriel Olympie
(parches de SGLang), Intel (AutoRound), Albucino (drafter MTP). Nuestros derivados se publican abiertos.

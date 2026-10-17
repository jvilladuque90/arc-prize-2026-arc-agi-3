# Resultados, experimentos y correcciones

[English](RESULTS.md) · [Volver al README](../README.es.md)

Todos los números son **puntajes públicos del set oculto** salvo los marcados *local* (una demo de 10 o 15 juegos en Kaggle). Los
números de sección (§) remiten a la bitácora original en español: [archive/docs/DESIGN.md](../archive/docs/DESIGN.md).

## 1. Historial en el set oculto

| Periodo | Pila | Puntajes | Notas |
|---|---|---|---|
| Ago | Explorador de grafo de estados (sin LLM) | 0,25 | el «piso» |
| inicios de sep | Harness Duck + Qwen 27B | 1,15 → 1,59 | lo mejor de la época; lo que lo movió fue cambiar la *base* (§8.38–8.42) |
| 10–29 sep | Harness Duck + Qwen3.8-Flash-Next NVFP4 en vLLM | 3,55 · 3,54 · 2,69 · 3,95 · 3,03 · 2,80 · 2,77 · 3,98 · 3,09 · 2,96 · 2,66 | n = 11, media 3,18, sd ≈ 0,5 |
| 30 sep – 2 oct | pila vLLM NVFP4 + parches públicos «agentfix» | 3,78 · 2,49 · 4,79 | una corrida terminó en `ERROR` de plataforma (no del código) |
| 3 oct → | **Solución del Milestone 2 de Franzen, sin cambios** | 25,01 · 25,17 · 23,39 · 27,15 · 27,63 · 25,97 | n = 6, **media 25,7, sd 1,6** |
| 9 oct | + H1 (arreglos de robustez del sandbox) | 24,17 | dentro del ruido |
| 10 oct | + E1 (`reasoning_effort=medium`) | **20,98** | ~3 sd *por debajo* del control |
| 18 oct → | alternando control / prioridad D′ del planificador | pendiente | ver [ROADMAP](ROADMAP.es.md) |

Tabla de posiciones el 2026-10-17 (4.425 equipos): 1.º 68,9, 2.º 67,5, 3.º 50,4, 10.º 41,7, 25.º 36,9, 100.º 33,2. Buena parte de la
franja 25.º–100.º son copias sin modificar de la misma solución pública, elevadas por tomar el máximo de muchos envíos diarios.

## 2. Experimentos

Veredictos: **+** ayudó, **0** dentro del ruido, **−** perjudicó, **?** sin decidir.

### Cambiar el sistema base (lo único que funcionó)

| Cambio | Resultado | Veredicto |
|---|---|---|
| Qwen 27B → Qwen3.8-Flash-Next (NVFP4, vLLM) | 1,59 → 3,55 | **+** |
| vLLM/NVFP4 → pila SGLang INT4 de Franzen (contexto de 128K, servicio rápido, UNDO, funciones persistentes, compuerta de prioridad) | 3–5 → **25,7** | **+** |

Por qué funcionó lo segundo (medido en los registros guardados de cuatro corridas independientes): decodificación de 700–780 tok/s
contra 195–270, un pool de KV de 1,01 M de tokens contra 0,1 M, y los tres ganadores públicos del hito comparten historia larga
retenida, caché KV en FP8, game over/undo observables y código del agente que persiste.

### Ajustes encima de una base (≈25 probados)

| Idea | Evidencia | Veredicto |
|---|---|---|
| Notas/memoria de texto escritas por el harness (7 variantes: objetos, precondiciones, narrativas, herencia de nivel, …) | oculto: todas ≤ base | **0/−** |
| «Manual del juego» y notas de consolidación al subir de nivel | local: peor que el control; oculto 2,66 contra base 3,2 | **−** |
| Recordatorio de agrupar acciones | local el mejor (4,62), oculto el peor (3,27) | **0** |
| Imagen de la cuadrícula con etiquetas | local 10–4 por juego, p = 0,18 | **0** |
| Pensamiento apagado | el peor de la serie | **−** |
| 11 «injertos» hechos a mano (eficiencia, guardia de reintentos, ayudas de esquema, …) | tres brazos, mismo sentido negativo | **−** |
| Nota de grafo de estados en el prompt (v1 con un error de vocabulario; v2 corregida) | local 21 contra 19 niveles a igual reloj, p = 1,0 | **0** (§8.83–8.84) |
| Solver con animación en la pila vieja | local el mejor, oculto 3,27 | **0** |
| H1: conservar la biblioteca de funciones tras un timeout del sandbox y acotar resultados enormes | compuerta 46,63 contra 46,31; oculto 24,17 | **0** (arreglo seguro) |
| D1: historia más profunda a igual KV (8 plazas, desalojo de 24K) | compuerta 29,76 contra 46,31: −29 % de acciones, −17 % de tok/s | **−** |
| E1: `reasoning_effort=medium` (el defecto de la plantilla es `xhigh`) | 15 difíciles ×3,1 (5,38 → 16,66; 11/3/1 juegos, p = 0,006); 10 fáciles 46,3 → 38,0; **oculto 20,98** | **−** (el resultado local no se transfirió) |
| Duración del primer tramo del planificador | simulación en CPU con las curvas de la demo: la curva es ~lineal (~2,2 pts/min), sin concavidad que aprovechar | descartado |
| Ocupación de plazas (8,3–8,6 de 10 en uso) | el servidor está saturado: llenar todas da +4–7 % de tokens | **0** |
| Prioridad D′ del planificador (notebook público, una muestra 31,54) | otras muestras del mismo código: 27,6–31,5 | **?** |

### Diagnóstico: el set oculto es una mezcla (§8.87)

La demo de Franzen juega 10 de los 25 juegos públicos. Ejecutar la solución sin cambios en los **otros 15** dio **5,38**, contra 46,3
en sus 10: una mezcla de 25 juegos de **21,7**, cercana al 25,7 visto en el oculto. Primer nivel: **2,6 min** en los fáciles y
**11,6 min** en los difíciles; tokens por acción ≈ 600 contra ≈ 2.200. La brecha es la *velocidad para entender un juego nuevo*.

## 3. Medición

- Desviación entre corridas en el oculto: **1,6** (n = 6 en el control; una instantánea comunitaria de copias reporta 3,9). Con un
  envío al día, 4 contra 4 muestras solo resuelven ~4–8 puntos.
- Una demo local de 10 juegos tiene una sd de la media por corrida de **≈ 4** (cada juego oscila decenas de puntos entre corridas idénticas).
- Por eso las compuertas locales *descartan cambios rotos*; no clasifican mejoras. El banco anterior de 25 juegos estaba incluso
  anticorrelacionado con el puntaje oculto (r = −0,35 en cuatro brazos, §8.81).
- La tabla guarda el máximo de cada equipo, así que reenviar una base sin cambios es una palanca (modesta): con media 25,7 y
  sd 1,6–3,9, el máximo esperado de ~28 sorteos ronda 29–34.

Nota de reproducibilidad: los notebooks de experimento de `notebooks/` se regeneraron con comentarios en inglés después de haber
subido a Kaggle los kernels H1, D1, E1 y G15. El código que se ejecuta es idéntico; solo difieren los comentarios y los docstrings de
los módulos embebidos.

## 4. Correcciones a afirmaciones previas

La bitácora archivada contiene estos errores, conservados por transparencia:

1. **Estadística agrupada contra pareada (§8.76 → §8.77).** Una prueba de proporciones agrupada declaró significativo un cambio de
   87,8 % → 81,4 % (z = −2,72); pareado por juego fue 8–5, p = 0,58: sin efecto.
2. **Un brazo de dos variables (§8.76)** cambió las etiquetas de la imagen *y* bajó la resolución, así que no aislaba ninguna.
3. **Una regla preregistrada sin comprobar los datos (§8.74)**: ya era inalcanzable.
4. **«La compuerta de los fáciles se repite a ±0,5» (§8.87) era falso.** Tres corridas dieron medias parecidas (47,1 / 46,3 / 46,6),
   pero juego a juego difieren decenas de puntos; la sd de la media de una corrida es ≈ 4.
5. **«Las compuertas locales predicen el oculto» (implícito en §8.89).** El brazo de esfuerzo medio triplicó el puntaje local en los
   difíciles y sacó menos que el control en el oculto.
6. **Una auditoría multiagente (§8.85) fue desmedida**: ~7 M de tokens, alcanzó un límite de sesión antes de consolidar; la
   consolidación se hizo a mano con los informes que habían llegado a disco.

## 5. Qué haríamos distinto

Empezar leyendo qué *comparten* las mejores soluciones públicas; mantener un control fiel y cambiar una sola cosa; presupuestar el
ruido antes de lanzar una prueba (¿cuántos envíos haría falta?); no fiarse nunca de una compuerta local para clasificar; y escribir
el efecto esperado de cada experimento antes de que llegue el resultado.

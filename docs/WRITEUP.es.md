# Qué mueve a un agente de ARC-AGI-3: un sistema base, medido con honestidad

[English](WRITEUP.md) · [Volver al README](../README.es.md)

**Equipo:** Julian Camilo Villa · **Concurso:** ARC Prize 2026, ARC-AGI-3 (Kaggle) · **Pila final:** media oculta 25,2 (n = 5); mejor envío 27,63 (control)
**Código:** este repositorio; cada kernel es una variante de una celda de `notebooks/franzen_m2.ipynb`

## Resumen

Dedicamos tres meses a ARC-AGI-3, la versión interactiva del benchmark, donde un agente debe descubrir actuando las reglas de juegos
que no ha visto. Nuestro propio pipeline llegó a una media de 3,2. Adoptar la solución pública del Milestone 2 de Daniel Franzen como
control fiel nos llevó a 25,3. Después probamos unos 25 cambios sobre una base, diez de ellos en el conjunto oculto. Ninguno sacó el
score del ruido, y uno lo bajó tres desviaciones estándar.

Los resultados útiles son de medición, no un truco nuevo:

1. **La palanca es el sistema base.** La velocidad de servicio, la longitud de contexto y la memoria fijan el techo; los añadidos al prompt no.
2. **Las compuertas locales no pueden ordenar cambios.** Una compuerta de 10 juegos tiene una desviación estándar de unos 4 por corrida.
   Su orden nunca coincidió con el del conjunto oculto.
3. **La métrica se pierde en niveles sin resolver, no en acciones de más.** El 55 % del score ponderado por nivel se pierde en niveles
   que el agente nunca resuelve. Solo el 2 % se pierde en niveles resueltos con acciones de más.
4. **El servidor está limitado por la memoria KV, no por el cómputo.** Más juegos en paralelo decodifican más rápido en el pico, pero
   generan un 21 % menos de tokens.

![Score por sistema base](figures/fig1_history.png)

## 1. Enfoque

### 1.1 Sistema base

El control es una copia sin cambios del notebook del Milestone 2 de Franzen, construido sobre el harness de Tufa Labs:

- **Modelo:** Qwen3.8-Flash-Next con los pesos INT4 AutoRound de Intel y el modelo borrador MTP de Albucino para decodificación especulativa.
- **Servicio:** Pennyroyal, un fork de SGLang para la RTX PRO 6000, con contexto de 128K, caché KV FP8 de 1,01 M tokens y 10 juegos
  decodificando en paralelo.
- **Agente:** el modelo razona conservando todo su pensamiento en la historia. Actúa con una herramienta de Python. Ve el tablero como
  imagen ampliada más los fotogramas de animación, y diagnostica cada game over con una imagen de diferencias. Puede deshacer, y
  conserva las funciones auxiliares que escribe durante el resto del juego.
- **Planificación:** una compuerta de prioridad da la GPU a los juegos con más progreso por token. Vacía la cola en el último 20 % del
  presupuesto de 9 horas.

Los tres ganadores públicos del hito comparten historia larga, caché KV FP8, game over y deshacer observables, y código persistente
escrito por el agente. Tomamos ese núcleo común como el mejor punto de partida disponible.

### 1.2 Nuestros añadidos

Cada añadido es una celda del notebook que parchea el harness por reemplazo de anclas exactas. Cada uno tiene una prueba unitaria
contra el archivo real del harness.

| Brazo | Qué cambia | Origen |
|---|---|---|
| **S1** memoria de niveles resueltos | Al superar un nivel, sus hechos y una regla de una línea escrita por el modelo se fijan al final del system prompt | `lalalia9` (CC-BY 4.0), a partir de `rellik13` |
| **M1** funciones sembradas | Cada juego empieza con 7 utilidades probadas, elegidas por lo que el modelo más reescribía en 70 registros guardados | nuestro |
| **C1** reinicio por estancamiento | Tras 96K tokens en un nivel sin terminar, limpia una vez el contexto de razonamiento | `heliosli` |
| **H1** correcciones del sandbox | Conserva la biblioteca de funciones tras un timeout; limita salidas enormes | nuestro |
| **X1, X2** | X1 = S1 + M1; X2 = X1 + C1 (el envío final) | — |

También probamos cambios de servicio: borrador MTP más profundo (T1), 12 juegos en paralelo (T2), menor esfuerzo de razonamiento (E1)
e historia más profunda (D1).

### 1.3 Protocolo

- **Una variable por brazo, contra un control fiel.** Cada brazo difiere del control en una celda.
- **La compuerta local solo descarta roturas.** Es una corrida de 25 minutos sobre 10 juegos públicos. También mide tokens de
  decodificación por segundo y la longitud aceptada especulativa, que se repiten al ~1 % y por eso sirven para juzgar cambios de servicio.
- **Solo el conjunto oculto ordena.** Enviamos uno al día, comparamos por juego con pruebas de signos y escribimos cada regla de
  decisión antes de que llegara su resultado.

## 2. Resultados

### 2.1 Conjunto oculto

![Muestras ocultas por brazo](figures/fig2_arms.png)

Las ocho muestras del control van de 23,4 a 27,6, con media 25,3 y desviación estándar 1,6. Con un envío al día, separar dos brazos
por 2 puntos pide unas 8 muestras de cada uno. Ningún brazo se acercó a tantas.

- **E1 empeoró.** Bajar el esfuerzo de razonamiento triplicó el score local en juegos difíciles y luego sacó 20,98 en el oculto.
- **Todo lo demás quedó dentro de la banda.** S1, D′ y X1 quedaron algo por encima de la media del control. X2, nuestra pila final,
  terminó en 25,2 con cinco muestras.

### 2.2 Por qué las compuertas locales no ordenan

![Score de compuerta frente a score oculto](figures/fig3_gates.png)

Corridas idénticas de la compuerta difieren decenas de puntos en juegos sueltos. La mejor compuerta del último mes, C1 con 50,7, habría
sido una pista falsa. En el pipeline anterior el banco local llegó a estar anticorrelacionado con el oculto en cuatro brazos (r = −0,35).

### 2.3 Dónde se pierde el score

![Descomposición de la métrica](figures/fig4_metric_loss.png)

La métrica eleva al cuadrado el cociente entre acciones de referencia y acciones tomadas, por nivel, y pondera más los niveles
posteriores. Desglosamos por nivel las 50 partidas de nuestras compuertas. Los niveles resueltos ya puntúan casi completos: el 40 % del
peso viene de niveles resueltos sin superar las acciones de referencia. La pérdida está en los niveles nunca resueltos, normalmente los
últimos, que pesan más. Una corrida de diagnóstico en los 15 juegos públicos fuera de la demo de Franzen muestra lo mismo. El primer
nivel tarda 2,6 minutos en juegos fáciles y 11,6 en difíciles, y cada acción cuesta unos 600 tokens de razonamiento frente a 2.200. La
brecha es lo rápido que el modelo entiende un juego nuevo.

### 2.4 El throughput lo limita la memoria

![Throughput con 10 y 12 streams](figures/fig5_concurrency.png)

Si la pérdida está en niveles sin resolver, más tokens de razonamiento por juego deberían ayudar, así que exprimimos el servidor.

- **Tiempo:** el presupuesto de 9 horas ya se usa entero, y cargar 185 GB de modelo al ancho de banda de entrada de Kaggle cuesta de 15 a 28 minutos.
- **Borrador MTP más profundo:** imposible. Los kernels de atención limitan los tokens borrador a 4.
- **12 juegos en paralelo:** la decodificación pico subió de 687 a 759 tokens por segundo. Luego el pool KV se saturó, hubo desalojos y
  reprefill, y la salida total bajó de 842 K a 662 K tokens.

La configuración de Franzen de 10 juegos en paralelo está en el óptimo.

## 3. Lecciones

1. **Empezar por lo que comparten las mejores soluciones públicas,** y cambiar una cosa cada vez contra una copia fiel.
2. **Contar las muestras antes de correr.** Con desviación estándar de 1,6 y una muestra al día, la mayoría de ideas no se pueden
   confirmar a tiempo. Hay que saberlo antes de gastar el envío.
3. **Medir el mecanismo, no el score.** Tokens por segundo, longitud aceptada y uso de KV se repiten al ~1 %. Resolvieron dos
   preguntas de servicio que las muestras de score nunca habrían resuelto.
4. **La infraestructura falla en silencio.** Tres envíos dieron exactamente 0 cuando Kaggle dejó de fijar la variable de rerun. Seis
   compuertas murieron cuando los contenedores montaron las entradas con un layout de directorios antiguo. Ambas correcciones están en
   todos los notebooks (`docs/RESULTS.es.md` §3.1).
5. **Las notas escritas por el harness no ayudan; los datos del propio agente quizá sí.** Siete variantes de notas quedaron en la base
   o por debajo. El único añadido de texto con un informe público positivo, S1, es la regla del propio agente.

## 4. Qué probaríamos después

La brecha restante, de 25 a los 70 de los líderes, es velocidad de comprensión. No viene de la economía de acciones ni de la
planificación. Los candidatos que la atacan son un modelo más fuerte o más eficiente en tokens, escalas KV FP8 calibradas, y un banco
de repetición que juzgue cambios de servicio desde prompts guardados sin gastar envíos.

## Agradecimientos

A Daniel Franzen por publicar completa la solución del Milestone 2, y a Tufa Labs por el harness sobre el que se construye. A John
Pezzulli (Pennyroyal), Mamy Ratsimbazafy y Gabriel Olympie por el trabajo en SGLang. A Intel por los pesos AutoRound y a Albucino por el
borrador MTP. A `lalalia9` y `rellik13` por la memoria de niveles resueltos, y a `heliosli` por el reinicio por estancamiento. A la ARC
Prize Foundation y a Kaggle por organizar el concurso.

Todos los números vienen de [docs/RESULTS.es.md](RESULTS.es.md). Las figuras las genera `scripts/make_writeup_figures.py`.

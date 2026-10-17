# Investigación: ideas, evidencia y vetas abiertas

[English](IDEAS.md) · [Volver al README](../README.es.md)

Fuentes: el write-up del repositorio público de Franzen (`WRITEUP.md`), los kernels públicos de los ganadores del Milestone 2, una
comparación de ~25 forks públicos y nuestras propias pruebas ([RESULTS](RESULTS.es.md)). Etiquetas de evidencia: **[H]** puntaje oculto
reportado por el autor, **[L]** medición local, **[C]** afirmación del autor sin número, **[nuestro]** prueba nuestra.

## 1. Ideas con evidencia externa

| Idea | Evidencia | Dónde está para nosotros |
|---|---|---|
| Una base más fuerte y rápida: historia larga retenida + caché KV en FP8 + servicio rápido | La comparten los tres ganadores públicos del hito; una escalera pública 6,4 → 22,5 en 12 días, con KV FP8 + historia 49K → 69,6K valiendo **+8,0** [H, muestras únicas] | **Adoptada** (es la base: 3 → 25,7) |
| Imagen del tablero más grande (4× → 10×) | «la mayor mejora de puntaje en cuanto a imágenes» [C] | En la base |
| Fotogramas de animación como datos en el sandbox | «el mayor beneficio» de los cambios de percepción [C]; +0,5 en una pila anterior [H] | En la base; en la pila *vieja* lo nuestro fue neutro |
| Diagnóstico del game over con el fotograma fatal y una imagen de diferencias | el mensaje de una línea «llevaba a análisis largos y confusos» [C] | En la base |
| Conservar el propio pensamiento del modelo en la historia | quitarlo causó «una gran regresión de puntaje» [C] | En la base |
| **Sin** notas de memoria / modelo del mundo escritas por el harness | desactivarlo fue «la mayor mejora en esta área» [C]; texto extra en el prompt costó −2,3 con contexto de 37K [H] | Coincide con nuestros 7 brazos de notas fallidos [nuestro] |
| Funciones persistentes escritas por el agente | en la base; una biblioteca por juego con pruebas sacó 23,8 [H] | En la base; endurecida con H1 (neutro) |
| Recorte por bloques + concurrencia acotada para reutilizar el caché de prefijos | 93 % de los tokens del prompt reutilizados [L] | En la base |
| Prioridad del planificador por progreso, tokens y valor de continuación | calibrada con 725 intentos de nivel; sin ablación [C] | En la base; variante D′ en prueba |
| «Reglas de niveles resueltos» fijadas por el agente | **+2,8** [H, muestra única] | **No probada** en la base nueva |

## 2. Qué nos faltó para poder realizar nuestras propias ideas

| Bloqueo | Efecto |
|---|---|
| **Sin señal de validación fiable.** Un envío al día, sd entre corridas 1,6–3,9 | 4 contra 4 muestras resuelven ~4–8 puntos; las ganancias pequeñas nunca se confirman |
| **Las compuertas locales no clasifican.** sd por corrida de una demo de 10 juegos ≈ 4; una ganancia local ×3 fue −4,7 en el oculto | Se gastaron ~5 brazos en falsos positivos |
| **Margen de memoria.** La GPU tiene 1,6–2,2 GiB libres durante el juego; el pool de KV está ~90 % lleno | Todo cambio que añada contexto, plazas o pesos (D1, cambios de modelo) compite con el throughput |
| **Dos sesiones de GPU a la vez**, ~40 min por compuerta | Serializa los experimentos |
| **Los líderes son privados.** Los puestos 1–2 (68,9; 67,5) están ~6 sd por encima de la distribución de copias públicas | No sabemos si su ganancia es mejor modelo, búsqueda, entrenamiento o harness |
| **Desarrollo en Windows contra un sandbox en Linux** | El arranque de 65 KB del sandbox no se puede lanzar en local; las pruebas cubren solo el código del anfitrión |

## 3. Vetas abiertas, en orden

| # | Idea | Efecto esperado | Coste / cómo validar |
|---|---|---|---|
| V1 | **Banco mecánico de repetición**: repetir los prompts guardados (`*_requests.jsonl`) contra cualquier configuración del servidor y medir tok/s, aceptación especulativa, tiempo al primer token, aciertos de caché | Habilita V2–V4 y V9: juzga cambios de servicio o numéricos *sin gastar envíos* | CPU + una corrida corta de GPU; no necesita calidad nueva del modelo |
| V2 | **Escalas de KV FP8 calibradas** (el servidor avisa de que no se aportan) | Palanca de calidad sin coste de memoria; efecto desconocido | Una pasada de calibración con prompts de ARC + banco de repetición; verificación oculta |
| V3 | **Throughput**: MTP con profundidad 4 (+4–6 % derivado), 12 plazas (pide más KV), MXFP8 en línea (~9 % afirmado), aceptación especulativa relajada (con pérdida) | Franzen: «los puntajes seguían mejorando al final de las corridas» → el tiempo es palanca, pero cada ítem es pequeño | Banco de repetición para los que no pierden calidad; los otros no se pueden validar |
| V4 | **Funciones auxiliares precargadas** (componentes conexas, diferencia de fotogramas, BFS, barrido de clics) en la biblioteca persistente | Ahorra los primeros minutos de cada juego; sin prosa, coherente con «frames yes, narration no» | Una celda pequeña; compuerta para roturas, oculto para el efecto |
| V5 | **Reinicio por estancamiento** (público, sin probar): tras ~98K tokens en un nivel sin terminar, limpiar el contexto de razonamiento y las funciones una vez | Podría destrabar juegos difíciles atrapados en bucles | Una celda; los autores tampoco tienen comparación |
| V6 | **Retención de funciones con pruebas** | Mejor calidad de la biblioteca; precedente más cercano 23,8 [H] | Ingeniería moderada; ganancia esperada baja aquí (la biblioteca ya tiene ~15 funciones por juego) |
| V7 | **Conservar constantes literales junto con las funciones** (el 42 % de los intentos de función se rechazan por «dependencias no disponibles») | Modesto | Toca anfitrión y sandbox; no se puede ejecutar de punta a punta en Windows |
| V8 | **Variantes de prioridad del planificador** (D′ en curso; ritmo activado) | Probablemente dentro del ruido | Alternancia diaria con el control |
| V9 | **Otro checkpoint** (ajuste fino de razonamiento eficiente, NVFP4 de NVIDIA, expertos podados) | La única vía a una ganancia grande, pero el candidato mejor documentado *subió* un 11,9 % los tokens totales en uso agéntico y su formato de 4 bits encoge el pool de KV ~20 % | Solo tras V1; no se puede validar en el oculto |

## 4. No probado, y por qué

- **Entrenamiento en tiempo de prueba / ajuste fino con los juegos públicos**: los juegos ocultos difieren de los públicos, así que
  el riesgo es sobreajuste y contaminación; el presupuesto de nueve horas no deja espacio para entrenar dentro de la corrida.
- **Autoconsistencia / votación**: los tokens por juego son el recurso limitante.
- **Un modelo del mundo aprendido o un simulador escrito por el modelo**: la evidencia externa es negativa (0 de 31 predicados de meta inducidos se cumplieron).
- **Reproducir a los líderes**: enfoque desconocido; las únicas pistas públicas son sus puntajes.

## 5. Nota sobre `reasoning_effort`

La plantilla de chat de Qwen3.8 usa `xhigh` por defecto («piensa con cuidado, valida supuestos, considera alternativas»). Bajarlo a
`medium` triplicó el puntaje local en los juegos difíciles pero bajó el puntaje oculto ([RESULTS](RESULTS.es.md)). Lo leemos así:
el razonamiento deliberado importa en los juegos ocultos que no vemos, y la ganancia local en difíciles vino de juegos donde la
cautela extra se desperdicia. El esfuerzo adaptativo *no* es, por tanto, una prioridad.

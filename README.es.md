# ARC Prize 2026 — ARC-AGI-3: un agente de código abierto y su bitácora de ingeniería

[English](README.md) · [Cómo funciona](docs/APPROACH.es.md) · [Resultados y lecciones](docs/RESULTS.es.md) · [Investigación e ideas abiertas](docs/IDEAS.es.md) · [Hoja de ruta](docs/ROADMAP.es.md)

Este repositorio documenta una participación en la competencia [ARC Prize 2026 – ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
de Kaggle, donde un agente debe aprender las reglas de **juegos interactivos de cuadrícula que nunca ha visto** y completar el
mayor número de niveles con el menor número de acciones, en una sola GPU y sin internet.

Es a la vez un pipeline de envíos que funciona y un registro honesto de lo que se probó: qué movió el puntaje del set oculto, qué
no, y los errores de medición cometidos por el camino. Todos los kernels son públicos y el código tiene licencia MIT, como exige el premio.

## Dónde estamos (2026-10-17)

| | |
|---|---|
| Base actual | Una copia fiel de la solución pública del Milestone 2 de Daniel Franzen (SGLang, Qwen3.8-Flash-Next en INT4, contexto de 128K) |
| Puntaje oculto de la base | **25,7 de media, sd 1,6 (n = 6)**, mejor 27,63 |
| Tabla de posiciones (4.425 equipos) | 1.º 68,9 · 3.º 50,4 · 10.º 41,7 · 100.º 33,2 · esta entrada 27,63 (puesto ~1.090) |
| Puntaje antes de adoptar esa base | 3–5 |

La historia de puntajes la domina una lección: **cambiar el sistema base llevó el puntaje de ~3 a ~26; cada uno de los unos 25
ajustes de prompt o harness probados encima de una base movió el puntaje menos que el ruido, o lo empeoró.**

## El sistema en un minuto

Un LLM juega cada partida escribiendo pequeños programas en Python. Un sandbox le entrega el tablero como imagen, una imagen de
«qué cambió», los fotogramas de animación y una segmentación del tablero en objetos; responde llamando a `action([...])`. Recuerda
su propio razonamiento (128K tokens de historia) y las funciones auxiliares que escribió. Un planificador reparte una GPU entre
110 juegos, y el modelo se sirve 2–3 veces más rápido que con una configuración estándar de vLLM. Detalle: [docs/APPROACH.es.md](docs/APPROACH.es.md).

## Qué aprendimos (versión corta)

1. **El sistema base lo domina todo.** Pasar a una pila con historia larga retenida y servicio rápido multiplicó ~8 veces el puntaje
   de cualquier ajuste posterior. Antes de afinar, hay que mirar qué comparten las mejores soluciones públicas.
2. **Las notas de texto escritas por el harness no ayudan** («frames yes, narration no»): los datos expuestos en el sandbox y las
   imágenes ayudaron; la prosa inyectada, no (7 brazos de memoria y notas, todos neutros o negativos).
3. **Los bancos locales engañan.** Un cambio que triplicó el puntaje en 15 juegos difíciles en local (`reasoning_effort=medium`)
   sacó *menos* en el oculto (20,98 contra 25,7). El ruido por juego en una demo de 10 juegos es grande (sd de la media de una corrida ≈ 4).
4. **Un envío al día vuelve lento el A/B.** Con sd ≈ 1,6–3,9, cuatro muestras por brazo solo resuelven diferencias de ~4–8 puntos;
   las mejoras pequeñas no se pueden confirmar y hay que defenderlas por mecanismo y evidencia externa.
5. **El puntaje está limitado por niveles completados, no por eficiencia de acciones**, en los juegos que el agente sabe resolver;
   la brecha con los líderes está en qué tan rápido *entiende* un juego nuevo (primer nivel: 2,6 min en juegos fáciles contra 11,6 en los difíciles).

Tablas completas, fe de erratas y la lista de ideas abiertas: [docs/RESULTS.es.md](docs/RESULTS.es.md), [docs/IDEAS.es.md](docs/IDEAS.es.md).

## Mapa del repositorio

```
README.md · README.es.md      esta visión general (inglés / español)
docs/
  APPROACH(.es).md            cómo funciona el sistema actual, explicado fácil
  RESULTS(.es).md             historial de envíos, cada experimento, correcciones a afirmaciones previas
  IDEAS(.es).md               investigación: ideas probadas por otros, qué las bloqueó, vetas abiertas
  ROADMAP(.es).md             qué está corriendo y qué sigue
  archive/                    la bitácora original en español y notas de diseño anteriores (histórico)
src/arc3/                     parches al harness de Franzen (franzen_hardening.py, franzen_effort.py)
scripts/                      publicación de kernels, automatización de envíos, constructor de notebooks (build_arm.py)
notebooks/                    los notebooks de Kaggle realmente enviados o usados como diagnóstico
tests/                        pruebas de los parches (se ejecutan contra el harness parcheado real)
archive/legacy/               el pipeline superado: explorador de grafo de estados, brazos vLLM/NVFP4, bancos de Colab
kernel_versions.json          versiones de kernel publicadas (hacen falta al enviar)
```

## Reproducir

```bash
pip install -r requirements.txt
# crear un archivo .env con dos líneas:  KAGGLE_API_TOKEN=...  y  kaggle_username=...   (ignorado por git; nunca versionarlo)
python scripts/download_data.py                       # datos de la competencia (no se versionan)
kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution -p _tmp_pub/dfranzen_arc-agi-3-milestone-2-solution -m
python scripts/build_arm.py fork                  # notebooks/franzen_m2.ipynb (copia fiel, verificada celda a celda)
python scripts/prepare_harness_tree.py                # árbol del harness parcheado que usan las pruebas
python tests/test_franzen_hardening.py && python tests/test_franzen_effort.py
python scripts/push_kernels.py franzenm2 --dry-run    # inspeccionar; quitar --dry-run y añadir --gpu para correr en Kaggle
python scripts/submit_at_reset.py juliancamilovilla/arc-agi3-franzen-m2 "mensaje"   # espera al cupo diario
```

Los envíos diarios alternan entre el control y el candidato actual con `scripts/daily_alternate.py`
(armado como tarea programada de Windows por `scripts/arm_submit_task.ps1`).

## Atribución

Esta entrada se apoya en trabajo público, usado bajo sus licencias y acreditado en cada notebook:
**Daniel Franzen** (solución del Milestone 2 y parche del harness), **Tufa Labs** (el harness Duck — Harold Bessis, Jeroen Cottaar,
Isaiah Pressman, Andries Smit, Michal Tesnar, Stefano Viel), **John Pezzulli** (Pennyroyal, el fork de SGLang), **Mamy Ratsimbazafy**
y **Gabriel Olympie** (parches de SGLang), **Intel** (cuantización AutoRound), **Albucino** (drafter MTP), **shiiin9 / Affectify**
(prioridad del planificador D′), **Scott Le Grand** y **keithtyser** (la base anterior vLLM/NVFP4). Los pesos del modelo Qwen3.8 siguen
bajo la licencia de Qwen.

Código de este repositorio: MIT, © 2026 Julian Camilo Villa Duque.

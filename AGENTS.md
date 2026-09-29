# AGENTS.md — ARC Prize 2026, ARC-AGI-3

Guía para cualquier agente (humano o IA) que trabaje en este repo. El detalle vive en
`docs/DESIGN.md`; aquí va lo que hay que saber antes de tocar nada y el plan vigente.

## Estado (2026-09-29)

**Plan vigente: `plan.md`** — nueva base `sheetu12b` (5,19 oculto, mejor kernel público),
envío del hito #2 con ella, y los puntos 1-4 rediseñados como DATOS en el sandbox, no notas.


- **Mejor score oculto: 3,95** (base NVFP4 verbatim, `juliancamilovilla/arc-agi3-nvfp4`).
- Cinco muestras de esa misma base: 3,55 / 3,54 / 2,69 / 3,95 / 3,03 → media 3,35, sd 0,49.
  El 3,95 es varianza, no mejora (DESIGN 8.82).
- Hasta el hito del 2026-09-30 se remuestrea la base una vez al día con la tarea programada
  `ARC-AGI3-SubmitOneShot` (cero GPU). No gastar esas muestras en brazos.

## Reglas que no se negocian

1. **Nunca** commitear ni redistribuir `.env`, `environment_files/`, `arc_agi_3_wheels/`,
   `ARC-AGI-3-Agents/` ni `_tmp_nvfp4_bundle/`.
2. Los kernels van **públicos** (`is_private: False`): lo exige el premio de código abierto.
3. Colab: las cuentas 1-3 son del proyecto hermano arc-agi-2. Sólo tocamos la 4 y la 5.
4. **La métrica** es `(baseline/acciones)²` por nivel completado, ponderada por índice de nivel.
   Más niveles con menos acciones. No "niveles".
5. **Estadística pareada juego a juego** (test de signos), nunca proporciones agrupadas (8.77).
6. **El banco de 60 min / 25 juegos NO selecciona**: correlación banco-oculto r = −0,351 (8.81).
   Un brazo no se promueve por ganar el banco.
7. Una variable por brazo. Toda GPU se anuncia antes de lanzarla.

## Plan: herramientas genéricas (no escritas a mano)

Diagnóstico: los once injertos con conocimiento de dominio escrito por nosotros quedaron por
debajo de la base (8.73), y Tufa Labs publicó lo mismo. Lo que queda es darle al modelo
**cómputo, algoritmos o aprendizaje** cuyo contenido sale del propio juego.

| # | Herramienta | Qué ataca | Coste | Estado |
|---|---|---|---|---|
| **2** | **Grafo de estados del anfitrión** (Blind Squirrel, 2º del preview): nodo = firma del tablero, arista = acción; nota con estados repetidos, acciones probadas desde aquí y frontera | acciones malgastadas en bucles | CPU, ~100 tokens de entrada | nota: **neutra** (8.84); falta variante que actúe |
| 1 | **Predictor de efectos aprendido en vivo** (StochasticGoose, 1º del preview): red pequeña que predice si una acción cambia el tablero | acciones inertes | CPU, cero tokens | pendiente |
| 3 | **Simulador escrito por el modelo**: `step(estado, accion)` en Python, planificar sin gastar acciones reales | planificación | muchos tokens | pendiente (apuesta Paper Award) |
| 4 | **Biblioteca de funciones entre niveles** (estilo Voyager): código que ya funcionó, no notas | transferencia nivel→nivel | pocos tokens | pendiente, con cautela (7 brazos de memoria fallaron) |
| 5 | Más cómputo de razonamiento (votación, verificador) | calidad por turno | reloj | **descartado**: a ~10 tok/s no cabe |

Orden: 2 → 1 → 3 → 4. El 2 va primero porque ya estaba construido y sólo esperaba una base con
guardia de no-ops y señal de animación, que animfast trae.

**Condición previa para promover cualquiera al oculto:** un banco que correlacione con el
set oculto. Sin él se repite el ciclo de los doce brazos elegidos a ciegas.

### Punto 2 — grafo de estados sobre animfast

- Módulo: `src/arc3/cognitive_map.py` (construido en `ee4677a`, aparcado en DESIGN 8.44).
- Tests del módulo: `python scripts/test_cognitive_map.py`.
- Constructor: `python scripts/build_animfast_map.py` → `notebooks/animfast_map.ipynb`
  (animfast_long + una celda; compuertas: una celda añadida, resto intacto, todo compila).
- Smoke contra el bundle anim real: `python scripts/smoke_animfast_map.py` (PASS).
- Kernel registrado: `animfastmap` → `arc-agi3-animfast-map` en `scripts/push_kernels.py`.
  **v1 lanzada (DESIGN 8.83)**: cae 1-12 pero con un bug de vocabulario (la frontera
  decia "NO has probado" lo ya hecho) y 10 min menos de reloj. Corregido y relanzado:
  **v2 neutra (DESIGN 8.84)**: mismo reloj 21 vs 19 niveles, 3-4, p=1,0; tokens/accion como el
  control. El grafo descrito en texto no paga.
- Diferencia con el parche viejo: se engancha a `inference.agent.tool_agent.ToolAgent`
  (la clase que animfast usa), no a `taaf_grafts.schema_helpers`.
- Riesgo conocido: es una nota de texto en el prompt, la misma forma que las notas que
  fallaron. Lo distinto es el contenido: estado del juego, no conocimiento nuestro.
  Siguiente variante si la nota no paga: que el anfitrión *actúe* sobre el grafo (volver a un
  estado conocido por el camino más corto) en vez de describirlo.

## Disciplina de documentación

DESIGN una vez y completo; working notes (`paper/working_note_{es,en}.md`) una línea por
resultado; commits de tres líneas con puntero a la sección de DESIGN.

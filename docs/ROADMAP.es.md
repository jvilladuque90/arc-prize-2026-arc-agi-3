# Hoja de ruta (actualizada 2026-10-17)

[English](ROADMAP.md) · [Volver al README](../README.es.md)

Fecha límite de envíos: **2026-11-02**. Un envío por día.

## En marcha

- **Alternancia diaria** (desde el 2026-10-18 UTC): la copia fiel de Franzen (control) un día y la variante D′ del planificador al
  siguiente, elegidas por la paridad de la fecha UTC con `scripts/submit/daily_alternate.py`. Si el kernel elegido no está en estado
  `COMPLETE`, se envía el otro. Está armada como tarea programada de Windows con cinco disparos al día (los tardíos reintentan tras
  un error de plataforma, que solo se ve horas después del envío).
- Ambos kernels son públicos; el control es una copia literal y D′ es una copia literal del notebook de su autor.

## Regla de decisión

El control tiene 8 muestras ocultas (media 25,3, sd 1,5; el control corregido dio 24,00 el 27 oct); D′ tiene 2 (27,30; 25,30). Un 0,00 exacto es un fallo de infraestructura y no cuenta (RESULTS §3.1). D′ solo se mantiene si su media con **al menos 4 muestras** supera a la
del control en más de ~3 puntos; si no, gana el control. Desde el **2026-10-29** solo se envía la mejor configuración, para
aprovechar el máximo de la tabla.

## Próximos brazos (un cambio cada uno, en este orden)

0. **Correcciones de infraestructura** (hechas el 26 oct): todos los notebooks reconocen un rerun puntuado por `ARC_BASE_URL` y esperan hasta 55 min al servidor (RESULTS §3.1); el control corregido volvió a pasar la compuerta con 42,15.
1. **Memoria de niveles resueltos (S1 / V10)**: *compuerta 44,91 el 26 oct* (`notebooks/franzen_s1.ipynb`, del notebook público de `lalalia9`, 31,46); **ya alterna a diario con el control** (`scripts/submit/daily_alternate.py`), D′ en pausa con dos muestras. Pasa por delante de M1/C1 porque es la única adición pública con ganancia reportada.
2. **Funciones auxiliares precargadas (M1 / V4)**: *construido y con pruebas unitarias* (`notebooks/franzen_m1.ipynb`); tras cuatro fallos de montaje (RESULTS §3.1) su compuerta pasó el 28 oct: 45,48, no rompe (los kernels son privados ahora: Kaggle no admite kernels públicos con la competición como fuente hasta que termine). Candidato al kernel apilado.
3. **Reinicio por estancamiento (C1 / V5)**: *construido y con pruebas unitarias* (`notebooks/franzen_c1.ipynb`, portado de `heliosli`); compuerta en cola detrás de M1. Su efecto solo puede aparecer en el rerun oculto.
4. **MTP profundidad 4 (T1 / V3)**: *construido el 27 oct* (`notebooks/franzen_t1.ipynb`); se juzga por tok/s y longitud aceptada de la compuerta (repetibles al ~1 %), no por su score.
5. **Última semana (desde el 29 oct)**: con ~5 envíos ningún brazo se puede confirmar en el oculto (sd 1,5), así que se envía a diario un único kernel *apilado*: S1 más lo que haya pasado su compuerta sin perder throughput. La tabla guarda el máximo (mejor control 27,63), así que esto no puede bajar el resultado final. Regla preinscrita aquí antes del 29 oct. **Hecho:** `notebooks/franzen_x1.ipynb` (S1 + M1 + todas las correcciones de infraestructura) pasó la compuerta con 40,09 el 28 oct en un contenedor con layout antiguo; `daily_alternate.py` ya lo envía solo. Se revisa únicamente si la primera muestra oculta de S1 (28 oct) sale muy por debajo del control.
6. **Banco de repetición (V1)** y **escalas de KV FP8 calibradas (V2)**: después de la competición salvo que sobre tiempo.

Cada brazo necesita: una prueba unitaria contra el harness parcheado real, una compuerta local (solo roturas) y luego muestras
ocultas contra el control. Lo que no se puede validar (trucos de servicio con pérdida, cambios de modelo) no se envía. Detalle y
orden: [IDEAS](IDEAS.es.md).

## Dónde están los diagramas

El diagrama del sistema, el del ciclo de experimentos y el mapa de componentes probado / probado por nosotros / sin probar están en
[APPROACH](APPROACH.es.md#diagrama-el-sistema). `python scripts/check_repo.py --tests` comprueba que documentos, registro, notebooks y pruebas concuerdan.

## Reglas de juego

- Una variable por brazo; el control sigue siendo una copia fiel.
- Las compuertas locales descartan cambios rotos; nunca clasifican mejoras ([RESULTS](RESULTS.es.md), §3).
- Los lanzamientos de GPU y los envíos se anuncian antes de hacerse.
- Las credenciales (`.env`) y los datos de la competencia nunca se versionan; todos los kernels son públicos.

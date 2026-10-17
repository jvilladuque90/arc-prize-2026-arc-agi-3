# Hoja de ruta (actualizada 2026-10-17)

[English](ROADMAP.md) · [Volver al README](../README.es.md)

Fecha límite de envíos: **2026-11-02**. Un envío por día.

## En marcha

- **Alternancia diaria** (desde el 2026-10-18 UTC): la copia fiel de Franzen (control) un día y la variante D′ del planificador al
  siguiente, elegidas por la paridad de la fecha UTC con `scripts/daily_alternate.py`. Si el kernel elegido no está en estado
  `COMPLETE`, se envía el otro. Está armada como tarea programada de Windows con cinco disparos al día (los tardíos reintentan tras
  un error de plataforma, que solo se ve horas después del envío).
- Ambos kernels son públicos; el control es una copia literal y D′ es una copia literal del notebook de su autor.

## Regla de decisión

El control tiene 6 muestras ocultas (media 25,7, sd 1,6). D′ solo se mantiene si su media con **al menos 4 muestras** supera a la
del control en más de ~3 puntos; si no, gana el control. Desde el **2026-10-29** solo se envía la mejor configuración, para
aprovechar el máximo de la tabla.

## Próximos brazos (un cambio cada uno, en este orden)

1. **Funciones auxiliares precargadas (M1 / V4)**: *construido y con pruebas unitarias* (`notebooks/franzen_m1.ipynb`); falta una compuerta de Save & Run y luego muestras ocultas contra el control.
2. **Reinicio por estancamiento (C1 / V5)**: *construido y con pruebas unitarias* (`notebooks/franzen_c1.ipynb`, portado de `heliosli`); mismo camino. Su efecto solo puede aparecer en el rerun oculto.
3. **Banco de repetición (V1)**: la herramienta que permite juzgar cambios de servicio y numéricos sin gastar envíos.
4. **Escalas de KV FP8 calibradas (V2)**: se valida primero con el banco de repetición.

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

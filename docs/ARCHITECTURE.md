# Arquitectura del sistema (2026-08-17)

> Qué corre exactamente en la submission, de dónde viene cada pieza, y **por qué seams
> podemos meter código propio**. Complementa [STRATEGY.md](STRATEGY.md) (qué hacemos y por qué)
> y [DESIGN.md](DESIGN.md) (análisis del problema y física del presupuesto).

## 1. El stack, de abajo hacia arriba

```
Kaggle rerun (8 h, sin internet, RTX Pro 6000 96 GB)
│
├─ gateway http://gateway:8001  ← el juez: sirve los ~110 juegos ocultos y cuenta niveles
│
├─ vLLM 0.19.0  (wheels: driessmit1/arc3-vllm-h100-wheelhouse-v3)
│   modelo: Qwen3.6-27B-FP8 (driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot, 33,7 GB)
│   arranque: --enable-prefix-caching --max-model-len 65536 --tool-call-parser qwen3_coder
│             --reasoning-parser qwen3 --generation-config vllm
│   memoria de atención resultante: 177.968 tokens  ← repartida entre 28 conversaciones
│
├─ TAAF harness  (bundle: thtennant/taaf-kaggle-source-share-fork)
│   ├─ taaf.Benchmark          orquesta juegos, concurrencia 28, presupuesto por juego
│   ├─ HarnessSolver           crea un ToolAgent por juego
│   └─ ToolAgent               el bucle agéntico: prompt → razonamiento → herramienta python
│       ├─ contexto 32.768 tokens (LOCAL_ANALYZER_CONTEXT_WINDOW)
│       ├─ temperatura 0.6, razonamiento explícito ACTIVADO
│       ├─ multimodal: imagen del tablero a escala 4
│       └─ sandbox python en SUBPROCESO aislado (SAFE_BUILTINS, 30 s por llamada)
│
├─ taaf_grafts  (injertos del fork; install blindado → cualquier fallo cae a stock)
│   ├─ efficiency      nota de presupuesto por turno            ACTIVO
│   ├─ retry_guard     capa de reintentos                       ACTIVO
│   ├─ shortcircuit    recorta sobre-exploración                ACTIVO
│   ├─ schema_helpers  precarga funciones de grillas al sandbox ACTIVO (desde v4)
│   ├─ goalkeep        retiene modelo del mundo entre muertes   APAGADO (0.81, sin evidencia a favor)
│   └─ banking · transfer · recovery · schema_notes · context_window   sin usar / probados y descartados
│
└─ NUESTRO CÓDIGO
    ├─ src/arc3/sandbox_nav.py     helpers de navegación inyectados en el sandbox  ← seam A
    ├─ src/arc3/features.py        features objetuales (componentes, button_score, traslación)
    ├─ src/arc3/agent.py           GraphExplorer (rama propia, 0.25; hoy fuera de producción)
    └─ scripts/build_duck_notebook.py   genera el notebook y aplica los parches
```

## 2. Los seams: por dónde entra código nuestro

Esta es la parte que importa para competir, y está **verificada en producción**.

| Seam | Cómo se entra | Coste para el agente | Estado |
|---|---|---|---|
| **A. Prelude del sandbox** | `schema_helpers.SANDBOX_HELPERS_PRELUDE` se lee **en cada llamada**, no se captura al importar → basta extenderlo por monkeypatch | el modelo debe **gastar un turno** llamando a la función | **Validado**: 726 llamadas a `plan_moves` en 25/25 juegos (2026-08-17) |
| **B. Nota del prompt** | `schema_helpers.HELPERS_PROMPT_NOTE`, mismo mecanismo | una línea de texto por turno | **Validado**: una sola línea bastó para que el modelo adoptara los helpers |
| **C. Prompt del usuario** | subclase de `ToolAgent._build_user_prompt` (lo hacen `goalkeep` y `efficiency`) | **cero llamadas**: el dato llega ya calculado | **IMPLEMENTADO** (`--effects`, 2026-08-19). Inyecta la tabla de efectos medida del historial. Probado extrayendo el parche del notebook generado (`scripts/test_seam_c.py`): nota no vacía 4/4 juegos, degrada al prompt del padre sin historial y con historial corrupto. Justificado por el banco micro: planificación 44.0% → 66.1% a 4B, 24 de 24 discordantes a favor (DESIGN §8.11) |
| **D. Cadena de analizadores** | `composite.register_chain_layer(flag, loader)` envuelve el ToolAgent | según la capa | disponible, sin usar |
| **E. Solver** | `from_solver` reemplaza la clase del solver (lo hacen `banking`/`transfer`) | según el caso | disponible, sin usar |

**Regla de oro de los seams:** todo parche va envuelto en `try/except` que deja la configuración
estándar intacta. Verificado dos veces en la G4: el banner sale o sale la línea de fallo, nunca
se cae la corrida.

## 3. Flujo de un turno del agente (dónde se va el presupuesto)

```
1. El harness arma el prompt:  sistema (~5k tokens) + esquema de herramientas
                             + historial recortado a 32.768 + imagen del tablero
2. vLLM PRELLENA ese prompt   ← 1.950-5.775 tokens/s ; caché de prefijos acierta 44%
3. El modelo GENERA           ← 110-236 tokens/s : razonamiento + código python
4. El sandbox ejecuta el código en un subproceso aislado (30 s máx, SAFE_BUILTINS)
5. El código llama a action([...]) → el gateway devuelve frames nuevos
6. Vuelta al paso 1 con el historial más largo
```

**Relación medida: 26 tokens leídos por cada 1 escrito.** Cada juego dispone de ~52.000 tokens
generados en las 8 h → **20-40 turnos de pensamiento y ~94 acciones por juego**.

## 4. Modos de ejecución

| Modo | Cuándo | Qué hace |
|---|---|---|
| **Rerun oculto** | `KAGGLE_IS_COMPETITION_RERUN=1` | espera el gateway, juega ~110 juegos ocultos 8 h; el score sale de las partidas (`submission.parquet` es un dummy) |
| **Save & Run** | push del kernel | juega los 25 juegos públicos offline con un corte suave configurable (`--soft-min`) — **banco de pruebas gratuito** |
| **Local (CPU)** | `scripts/test_*.py` | verifica injertos y helpers sin GPU: segundos, cero cuota |

## 5. Instrumentos de medición (ordenados por costo)

1. **CPU local, segundos, gratis** — `test_schema_helpers.py`, `test_sandbox_nav.py` (corre el
   prelude bajo SAFE_BUILTINS restringidos), `smoke_graft_install.py`, `verify_nav_notebook.py`.
   Cazaron dos bugs reales antes de gastar GPU.
2. **Save & Run de 1 hora, ~1 h de cuota, sin gastar envío** — `--soft-min 70`. Da 1420 acciones,
   9 niveles y 25/25 juegos activos: **el único instrumento offline que discrimina**. La ventana
   corta de 16 min solo sirve para verificar mecanismos.
3. **Log del servidor vLLM, gratis** — viene en la salida del kernel: acierto de caché, tokens
   por segundo de entrada y salida, ocupación de memoria de atención.
4. **Envío diario al set oculto, 1 por día** — el único juez real. Varianza medida 0,41, así que
   distinguir dos configuraciones cuesta 3-4 noches.

## 6. Automatización

- `scripts/push_kernels.py <kernel> [--gpu]` publica headless y registra la versión en
  `kernel_versions.json`.
- `scripts/daily_submit.ps1` (tarea de Windows, 20:00) envía la última versión **validada**;
  exige `-v <versión>` porque las code competitions lo requieren.
- `scripts/push_v4_when_quota.py` publica con protección: revierte el puntero a la versión
  validada mientras la nueva corre su Save & Run, para que el disparador nunca envíe algo sin probar.

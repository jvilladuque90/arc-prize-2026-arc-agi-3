# archive/

[Español abajo](#español)

Historical material kept for transparency. **Nothing here is maintained or runnable as is** (paths, imports and pinned
kernels point at an earlier layout). The current work lives in `src/`, `scripts/`, `notebooks/`, `tests/` and `docs/`.

## `legacy/` — the superseded pipeline (Aug – Sep 2026)

| Folder | What it was |
|---|---|
| `src/arc3/` | the first agents (a state-graph explorer with masked hashing, a hybrid LLM agent) and the modules of the ~25 harness "arms" tested on the vLLM/NVFP4 base: level-carry notes, level inheritance, action-budget note, labeled grid images, cognitive map, game manual, effects model, sandbox effects (S1) |
| `tests/`, `scripts/test_*.py`, `scripts/smoke_*.py` | tests and smoke checks for those modules |
| `notebooks/` | 58 Kaggle notebooks of those arms (duck-harness variants, NVFP4 arms, animfast, sheetu, Colab benches) |
| `scripts/` | their builders, the Colab benches and the old scheduling helpers; `push_kernels_v1_registry.py` is the full kernel registry of that era |
| `data/` | raw outputs of the early local benches |
| `vendor/` | a snapshot of an early public agent |

The results and verdicts of these experiments are summarised in [docs/RESULTS.md](../docs/RESULTS.md); the complete story
(in Spanish) is in [archive/docs/DESIGN.md](docs/DESIGN.md).

---

## Español

Material histórico conservado por transparencia. **Nada de esto se mantiene ni se puede ejecutar tal cual** (rutas, imports y
kernels fijados apuntan a una estructura anterior). El trabajo vigente está en `src/`, `scripts/`, `notebooks/`, `tests/` y `docs/`.

`legacy/` contiene el pipeline superado (ago – sep 2026): los primeros agentes (explorador de grafo de estados con hashing
enmascarado, agente híbrido con LLM), los módulos de los ~25 «brazos» de harness probados sobre la base vLLM/NVFP4, sus pruebas,
58 notebooks de Kaggle, sus constructores, los bancos de Colab y las salidas crudas de los bancos locales tempranos.
Los resultados y veredictos están resumidos en [docs/RESULTS.es.md](../docs/RESULTS.es.md); la historia completa (en español)
en [archive/docs/DESIGN.md](docs/DESIGN.md).

# CLAUDE.md

@AGENTS.md

## Notas específicas para Claude Code

- Windows: los logs de kernels se descargan con `PYTHONUTF8=1`, o salen de 0 bytes.
- Las credenciales de Kaggle están en `.env` (cargar con `set -a; . ./.env; set +a`); nunca
  imprimirlas ni commitearlas.
- Scripts con `\n` o comillas: escribirlos con la herramienta Write, no con heredocs de Bash.
  Mensajes de commit con `git commit -F <archivo>`.
- No lanzar GPU (push de kernels, Save & Run) sin anunciarlo y tener visto bueno.
- Plan vigente: **`plan.md`** (nueva base sheetu12b + puntos 1-4 como datos en el sandbox).
  Regla de diseño: nada de notas narradas en el prompt ("frames yes, narration no").

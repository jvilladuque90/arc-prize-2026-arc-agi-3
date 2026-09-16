"""Publica los notebooks como kernels de Kaggle vía CLI (truco save-and-run headless).

Igual que en AG2: NADA de sesiones interactivas. `kaggle kernels push` dispara un
batch run ("Save & Run All"); los kernels CPU no gastan cuota de GPU, así que el
default aquí es CPU. La GPU disponible en esta competencia es la "G4"
(machine_shape NvidiaRtxPro6000, exclusiva de ARC-AGI-3) — usar --gpu solo cuando
el trabajo lo necesite de verdad. Config copiada del notebook guía RTX_G4 del usuario.

Ejemplos:
  python scripts/push_kernels.py features                 # CPU, no gasta cuota GPU
  python scripts/push_kernels.py features --gpu           # RTX Pro 6000
  python scripts/push_kernels.py features --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "arc-prize-2026-arc-agi-3"

# Imagen pineada: la misma del notebook RTX_G4 creado a mano con la G4 configurada
# (reproducible y compatible con la machine_shape NvidiaRtxPro6000).
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")

KERNELS = {
    "features": {"notebook": "notebooks/features.ipynb", "slug": "arc-agi3-features",
                 "title": "arc agi3 features"},
    # SONDA de montajes (CPU, cero cuota): lista donde monta Kaggle la competencia y los
    # datasets. Creada el 2026-09-14 cuando dos corridas murieron porque la ruta
    # /kaggle/input/competitions/<comp>/arc_agi_3_wheels no existia. Lleva los mismos
    # datasets que los kernels NVFP4 para ver el layout exacto que reciben.
    "probe": {"notebook": "notebooks/probe_mounts.ipynb", "slug": "arc-agi3-probe-mounts",
              "title": "arc agi3 probe mounts",
              "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                   "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"]},
    # Submission dual-mode (gateway en rerun / offline en Save & Run). CPU: no gasta cuota G4.
    "submit": {"notebook": "notebooks/submit.ipynb", "slug": "arc-agi3-submit",
               "title": "arc agi3 submit"},
    # Fase 3: baseline LLM (duck harness Tufa Labs). REQUIERE --gpu y los 3 datasets:
    #   --dataset jeroencottaar/taaf-kaggle-source-share
    #   --dataset driessmit1/arc3-vllm-h100-wheelhouse-v3
    #   --dataset driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot
    # GASTA CUOTA G4: en Save & Run corre ~TAAF_OFFLINE_SOFT_MIN min (default 25) de validación.
    "duck": {"notebook": "notebooks/duck.ipynb", "slug": "arc-agi3-duck",
             "title": "arc agi3 duck",
             # 2026-08-10: bundle cambiado al fork publico del cluster 1.5 del LB
             # (duck v12 de thtennant, con taaf-grafts de eficiencia)
             "default_datasets": ["thtennant/taaf-kaggle-source-share-fork",
                                  "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                  "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]},
    # Kernel de EXPERIMENTO (no se envía nunca): mismo duck v4 pero con la ventana
    # de contexto reducida. Mide en el log de vLLM si sube el acierto de la caché de
    # prefijos (base 44%) y la generación (base ~195 tok/s). Slug aparte para no
    # tocar la versión que envía el trigger diario.
    "duckctx": {"notebook": "notebooks/duck_ctx.ipynb", "slug": "arc-agi3-duck-ctx",
                "title": "arc agi3 duck ctx",
                "default_datasets": ["thtennant/taaf-kaggle-source-share-fork",
                                     "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                     "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]},
    # Segundo kernel de experimento: permite correr DOS brazos en paralelo
    # (Kaggle admite 2 sesiones GPU simultaneas) y comparar en la misma tarde.
    "duckctx2": {"notebook": "notebooks/duck_ctx2.ipynb", "slug": "arc-agi3-duck-ctx2",
                 "title": "arc agi3 duck ctx2",
                 "default_datasets": ["thtennant/taaf-kaggle-source-share-fork",
                                      "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                      "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]},
    # AMPLIFICACION: duck + helpers de navegacion propios inyectados en el sandbox.
    "ducknav": {"notebook": "notebooks/duck_nav.ipynb", "slug": "arc-agi3-duck-nav",
                "title": "arc agi3 duck nav",
                "default_datasets": ["thtennant/taaf-kaggle-source-share-fork",
                                     "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                     "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]},
    # v12 = v11 + nav, bajo la METRICA REAL (DESIGN 8.27): nav es +25% en las dos
    # varas independientes (offline 1.32 vs 1.06 y LB oculto). Slug propio de
    # experimento: NO toca el kernel del envio diario. Smoke de ~25 min en G4.
    "duckv12": {"notebook": "notebooks/duck_v12.ipynb", "slug": "arc-agi3-duck-v12",
                "title": "arc agi3 duck v12",
                "default_datasets": ["thtennant/taaf-kaggle-source-share-fork",
                                     "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                     "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"],
                # Qwen3.8-27B-FP8 publico (fuente: kernel LB-9). Adjuntarlo no cambia
                # nada si el notebook no se construyo con --model-qwen38.
                "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"]},
    # v21 MIGRACION DE BASE (docs/AUDIT_2026-09-08.md): el harness pasa a ser el
    # fork "animation-awareness" de jakobbrggen — el que corre el kernel publico
    # LB-9 de la banda 3-4. Se adjuntan CUATRO datasets porque el bundle anim no
    # trae taaf-grafts: el fork de thtennant viaja solo para eso, y el notebook
    # monta unicamente su subarbol src/taaf-grafts (nunca sus copias del harness,
    # que sombrearian la conciencia de animacion). REQUIERE --gpu.
    "duckanim": {"notebook": "notebooks/duck_anim.ipynb", "slug": "arc-agi3-duck-anim",
                 "title": "arc agi3 duck anim",
                 "default_datasets": ["jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
                                      "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                      "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
                                      "thtennant/taaf-kaggle-source-share-fork"],
                 # Qwen3.8-27B-FP8 publico: el modelo que corre LB-9. Necesario
                 # cuando el notebook se construye con --model-qwen38; adjuntarlo
                 # no cambia nada si no se usa ese flag.
                 "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"]},
    # Réplica fiel del explorador público 0.54 (harness oficial + Explore2 vendorizado).
    # CPU puro: NO gasta cuota G4. Base probada para cerrar la brecha de exploración.
    # Corre NUESTRO banco contra el 27B de produccion: la comparacion que decide
    # si el tamano del modelo esta justificado o si conviene cambiarlo por
    # throughput (un 4B daria ~7x acciones). REQUIERE --gpu.
    "bench27b": {"notebook": "notebooks/bench27b.ipynb", "slug": "arc-agi3-bench27b",
                 "title": "arc agi3 bench27b",
                 "default_datasets": ["thtennant/taaf-kaggle-source-share-fork",
                                      "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                      "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]},
    # MIGRACION DE STACK DE SERVICIO (auditoria 2026-09-10). Copia VERBATIM del
    # kernel publico de keithtyser "Duck Qwen3.8 Flash Next NVFP4 MTP" con UNA sola
    # edicion: recortar la ventana de juego offline (su soft_end son 8h50m y se
    # comeria la cuota de G4 entera). No lleva NADA nuestro a proposito: toda la
    # evidencia externa es "este archivo exacto puntua", y cada perilla anadida
    # rompe esa inferencia. Autoria: kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp
    # Imagen docker propia (mas nueva que la nuestra): la exige su runtime NVFP4.
    # Kernel de EXPERIMENTO (nunca el del envio): el NVFP4 verbatim + UNA celda con
    # nuestros tres injertos de v23 embebidos (scripts/build_nvfp4_grafts.py).
    # Mismos datasets, modelo e imagen que nvfp4: la unica variable son los injertos.
    # Kernel de EXPERIMENTO: NVFP4 verbatim + UNA celda con la consolidacion al ganar
    # nivel (src/arc3/level_carry.py, scripts/build_nvfp4_carry.py). Apunta al muro
    # del nivel 2 que dejo v24 = 3.55. Mismos datasets, modelo e imagen que nvfp4.
    # Corridas de 60 min en REGIMEN (scripts/build_nvfp4_long.py): el instrumento que
    # discrimina profundidad. Control plano y brazo de consolidacion, en paralelo
    # (Kaggle admite 2 sesiones GPU). Slugs propios: nunca el del envio.
    # MANUAL DEL JUEGO (build_nvfp4_manual.py): consolidacion + localizacion del objeto
    # ganador + afordancias positivas. La version long (60 min) es la que se banca.
    # PRESUPUESTO DE TURNO (build_nvfp4_yield.py): consolidacion + YIELD_SECONDS 60->180.
    # Mecanica, cero tokens de prompt: el 45-49% de los turnos se cortaban antes de actuar.
    # STACK DE SERVICIO (build_nvfp4_kv.py): alinear vLLM con la concurrencia del
    # harness. El log de TODAS nuestras corridas NVFP4 dice "Maximum concurrency for
    # 32768 tokens per request: 3.21x" mientras el harness lanza 28 juegos, con 75 GB
    # de la tarjeta sin usar y max_num_seqs=8 (el propio serving_setup trae 28 de
    # defecto). KV 5 -> 46 GiB y seqs 8 -> 28. Modelo, MTP y dtype intactos.
    "nvfp4carrykvlong": {"notebook": "notebooks/nvfp4_carry_kv_long.ipynb",
                         "slug": "arc-agi3-nvfp4-carry-kv-long",
                         "title": "arc agi3 nvfp4 carry kv long",
                         "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                              "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                         "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                         "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                          "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # DECODIFICACION ESPECULATIVA (build_nvfp4_mtpshare.py): encender IndexShare entre
    # iteraciones de MTP. De los tres mandos de MTP que expone serving_setup.py, el perfil
    # ganador solo fija MTP_TOKENS=3; los otros dos corren en su defecto APAGADO. La fuente
    # vendida en el propio bundle dice que este deberia ir ENCENDIDO para esta arquitectura:
    # src/sglang-rtxpro6000/.../configs/qwen4_exp.py trae index_share_for_mtp_iteration=True
    # de defecto con el comentario "default on for Qwen4-Exp", y nuestro modelo es
    # exactamente qwen4_exp (serving_setup.py:724-726 lo exige). Mecanica: reutiliza la
    # seleccion del indexador del draft-extend en los 3 pasos de borrador en vez de
    # recalcularla. Calidad intacta por construccion (el muestreo de rechazo verifica cada
    # token); solo mueve velocidad. Y la velocidad suma porque el presupuesto de pensamiento
    # es POR TIEMPO (<=60 s/turno): mas tokens/s = mas razonamiento dentro del mismo turno.
    "nvfp4mtpsharelong": {"notebook": "notebooks/nvfp4_carry_mtpshare_long.ipynb",
                          "slug": "arc-agi3-nvfp4-mtpshare-long",
                          "title": "arc agi3 nvfp4 mtpshare long",
                          "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                               "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                          "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                          "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                           "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # CONSOLIDACION v6 (build_nvfp4_carrynar.py): el CONTENIDO de v5 con la FORMA de v4.
    # La unica prueba que zanja DESIGN 8.63. v4 (contenido erroneo, forma narrativa) subio la
    # mencion de la nota en el razonamiento al 31,9% frente al 11-15% de v1-v3, pareado 13-3
    # de 16 juegos (p=0,0213). v5 (contenido CORRECTO -- mediana 10 celdas en vez de 44, 6,6%
    # de objetos >=100 celdas en vez de 38,9% -- pero en forma de inventario y con salvedad
    # epistemica) la hundio al 9,4%: pareado 1-13 de 14 contra v4, p=0,0018, e indistinguible
    # de v1. Como v5 CONSERVA el vocabulario de objetos, lo que pagaba en v4 no era eso.
    # v6 deja el contenido de v5 byte a byte (el modulo se genero sustituyendo SOLO la funcion
    # que redacta; hay test de equivalencia de codigo y de salida) y le devuelve la sintaxis de
    # v4. Si el enganche vuelve al 30%, lo que paga es la FORMA; si no vuelve, lo de v4 era
    # novedad y el eje se cierra. Se juzga por MECANISMO (8.60), no por la media del banco.
    "nvfp4carrynarlong": {"notebook": "notebooks/nvfp4_carrynar_long.ipynb",
                          "slug": "arc-agi3-nvfp4-carrynar-long",
                          "title": "arc agi3 nvfp4 carrynar long",
                          "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                               "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                          "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                          "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                           "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # CONSOLIDACION v5 (build_nvfp4_carrypre.py): la POSICION GANADORA (precondicion).
    # v4 abrio el canal: la mencion de la nota en el razonamiento subio de 11-15% (v1/v2/v3,
    # vara de ruido de UN punto) a 30,9%, pareado 13-3 de 16 juegos, p=0,0213 (DESIGN 8.62).
    # Pero su contenido era erroneo DESDE v1: la transicion se marcaba con el fotograma
    # POSTERIOR a la accion, que ya es el primer tablero del nivel siguiente, asi que la nota
    # describia el REDIBUJADO DEL CAMBIO DE NIVEL -- 208 "desaparecio" contra 112 "se movio",
    # y 39% de los objetos citados con >=100 celdas (max 650 de 4096; "desaparecio el objeto
    # W de 624 celdas" en vc33). El efecto de la jugada es INOBSERVABLE: no hay fotograma
    # intermedio entre aplicarla y estar en el nivel siguiente. v5 cuenta lo que SI se
    # observa y transfiere, leyendo SOLO fotogramas del nivel que se gana: que pieza
    # respondia a las jugadas, sobre que objeto se apunto, y con que estaba en contacto.
    # Se juzga por MECANISMO (8.60), no por la media del banco.
    "nvfp4carryprelong": {"notebook": "notebooks/nvfp4_carrypre_long.ipynb",
                          "slug": "arc-agi3-nvfp4-carrypre-long",
                          "title": "arc agi3 nvfp4 carrypre long",
                          "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                               "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                          "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                          "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                           "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # CONSOLIDACION v4 (build_nvfp4_carryobj.py): la mecanica ganadora en OBJETOS.
    # v1/v2/v3 nombraban la ACCION ganadora. Medido sobre 722 turnos (DESIGN 8.61): el
    # reuso de esa accion es 55,2% y 43,9% en dos replicas CON nota y 47,5% SIN nota (las
    # replicas encierran al control, pareado 5/3/4 p=0,727); en 85-89% de los turnos el
    # razonamiento no menciona la nota; y en 8 de 19 juegos la accion "ganadora" ya era
    # >=80% de TODAS las acciones (seis al 100%): informacion cero. El modelo consolida
    # solo, pero en objetos ("charcoal piece overlapped the yellow target"). v4 le habla
    # en ese idioma con los MISMOS hash de current_frame.segmentation (se llama a
    # segment_layer del propio harness). Cambia el VOCABULARIO, no la cantidad (~90 tok).
    # OJO AL LEER EL RESULTADO: DESIGN 8.60 probo que el PUNTAJE no puede resolver este
    # eje (3-6 eventos de nivel 2 por corrida frente a los 11-12 que pide el signo). Esta
    # corrida se juzga por MECANISMO (cientos de eventos), no por la media del banco.
    "nvfp4carryobjlong": {"notebook": "notebooks/nvfp4_carryobj_long.ipynb",
                          "slug": "arc-agi3-nvfp4-carryobj-long",
                          "title": "arc agi3 nvfp4 carryobj long",
                          "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                               "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                          "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                          "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                           "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # MODELO OBJETIVO (build_nvfp4_moe.py): nucleo fusionado de expertos de Blackwell.
    # TAAF_VLLM_MOE_BACKEND tampoco esta en el perfil ganador y acepta exactamente un
    # valor no nulo: flashinfer_b12x (serving_setup.py:404). El modelo tiene 512 expertos
    # en 48 capas, asi que la ruta de expertos es el grueso del computo del modelo
    # OBJETIVO; el brazo anterior (IndexShare, DESIGN 8.58) actuaba sobre la cabeza
    # borradora de MTP, de UNA capa, y rindio +0,3%. Y b12x es el nucleo de Blackwell,
    # que es nuestra tarjeta exacta (TORCH_CUDA_ARCH_LIST="12.0", RTX PRO 6000).
    # RIESGO: --moe-backend NO esta en la lista blanca de banderas que el setup verifica
    # contra `vllm serve --help=all` (serving_setup.py:1771-1790). Si la version pineada
    # no la reconoce, el servidor muere en el arranque (~10 min).
    "nvfp4moelong": {"notebook": "notebooks/nvfp4_carry_moe_long.ipynb",
                     "slug": "arc-agi3-nvfp4-moe-long",
                     "title": "arc agi3 nvfp4 moe long",
                     "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                          "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                     "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                     "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                      "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # CONSOLIDACION v3 (build_nvfp4_carry.py 3): DECAIMIENTO de la memoria vieja.
    # Hipotesis de Julian: arrastrar detalle de niveles pasados sesga las decisiones.
    # Detalle solo del ultimo nivel; lo anterior colapsa a una linea de esencia; sin
    # receta literal (el anclaje medido en v2); conserva el invariante. Mas pequena
    # que v1 en todas partes: ~93 tok con 1 nivel (igual que v1) y -34% en profundos.
    "nvfp4carry3long": {"notebook": "notebooks/nvfp4_carry3_long.ipynb",
                        "slug": "arc-agi3-nvfp4-carry3-long",
                        "title": "arc agi3 nvfp4 carry3 long",
                        "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                             "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                        "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                        "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                         "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # CONSOLIDACION v2 (build_nvfp4_carry.py 2): receta comprimida del ultimo nivel
    # ganado + eje comun de sus clics + invariante cuando dos niveles se ganan igual.
    # Amplifica la UNICA senal que da el entorno (completar nivel), que es lo unico
    # que ha pagado. ~126 tokens frente a ~93 de v1.
    "nvfp4carry2long": {"notebook": "notebooks/nvfp4_carry2_long.ipynb",
                        "slug": "arc-agi3-nvfp4-carry2-long",
                        "title": "arc agi3 nvfp4 carry2 long",
                        "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                             "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                        "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                        "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                         "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # GUARD DE NO-OPS DEL ANFITRION (build_nvfp4_noopguard.py): consolidacion + bloqueo de
    # repetir una accion ya probada inerte sobre el mismo tablero. Cognicion sin texto.
    "nvfp4carrynoopguardlong": {"notebook": "notebooks/nvfp4_carry_noopguard_long.ipynb",
                                "slug": "arc-agi3-nvfp4-carry-noopguard-long",
                                "title": "arc agi3 nvfp4 carry noopguard long",
                                "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                                     "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                                "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                                "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                                 "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # PENSAMIENTO APAGADO (build_nvfp4_nothink.py): consolidacion + ENABLE_THINKING=false.
    # Siguiente punto de la curva de pensamiento por turno (60 s -> 26 niveles, 180 s -> 17).
    "nvfp4carrynothinklong": {"notebook": "notebooks/nvfp4_carry_nothink_long.ipynb",
                              "slug": "arc-agi3-nvfp4-carry-nothink-long",
                              "title": "arc agi3 nvfp4 carry nothink long",
                              "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                                   "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                              "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                              "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                               "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    # Slug NUEVO (-2): el anterior perdio el adjunto de la competencia en sus dos sesiones
    # (publicado justo tras rechazos por "2 sesiones GPU"); la sonda demostro que un kernel
    # limpio si monta el wheelhouse. Mismo notebook, push limpio.
    "nvfp4carryyieldlong2": {"notebook": "notebooks/nvfp4_carry_yield_long.ipynb",
                             "slug": "arc-agi3-nvfp4-carry-yield-long-2",
                             "title": "arc agi3 nvfp4 carry yield long 2",
                             "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                                  "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                             "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                             "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                              "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    "nvfp4carryyieldlong": {"notebook": "notebooks/nvfp4_carry_yield_long.ipynb",
                            "slug": "arc-agi3-nvfp4-carry-yield-long",
                            "title": "arc agi3 nvfp4 carry yield long",
                            "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                                 "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                            "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                            "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                             "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    "nvfp4manual": {"notebook": "notebooks/nvfp4_manual.ipynb", "slug": "arc-agi3-nvfp4-manual",
                    "title": "arc agi3 nvfp4 manual",
                    "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                         "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                    "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                    "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                     "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    "nvfp4manuallong": {"notebook": "notebooks/nvfp4_manual_long.ipynb", "slug": "arc-agi3-nvfp4-manual-long",
                        "title": "arc agi3 nvfp4 manual long",
                        "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                             "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                        "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                        "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                         "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    "nvfp4long": {"notebook": "notebooks/nvfp4_long.ipynb", "slug": "arc-agi3-nvfp4-long",
                  "title": "arc agi3 nvfp4 long",
                  "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                       "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                  "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                  "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                   "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    "nvfp4carrylong": {"notebook": "notebooks/nvfp4_carry_long.ipynb", "slug": "arc-agi3-nvfp4-carry-long",
                       "title": "arc agi3 nvfp4 carry long",
                       "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                            "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                       "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                       "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                        "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    "nvfp4carry": {"notebook": "notebooks/nvfp4_carry.ipynb", "slug": "arc-agi3-nvfp4-carry",
                   "title": "arc agi3 nvfp4 carry",
                   "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                        "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                   "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                   "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    "nvfp4grafts": {"notebook": "notebooks/nvfp4_grafts.ipynb", "slug": "arc-agi3-nvfp4-grafts",
                    "title": "arc agi3 nvfp4 grafts",
                    "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                         "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
                    "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
                    "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                                     "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    "nvfp4": {"notebook": "notebooks/nvfp4.ipynb", "slug": "arc-agi3-nvfp4",
              "title": "arc agi3 nvfp4",
              "default_datasets": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                                   "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
              "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
              "docker_image": ("gcr.io/kaggle-private-byod/python@sha256:"
                               "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")},
    "explorer054": {"notebook": "notebooks/explorer054.ipynb", "slug": "arc-agi3-explorer054",
                    "title": "arc agi3 explorer054"},
    # Fase 3 (NUESTRO agente): LLMAgent con features objetuales + fallback. REQUIERE --gpu.
    #   wheels vLLM + modelo Qwen3-27B-FP8 (públicos). GASTA CUOTA G4 (~30 min validación).
    "llm": {"notebook": "notebooks/llm.ipynb", "slug": "arc-agi3-llm",
            "title": "arc agi3 llm",
            "default_datasets": ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                 "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]},
}


def load_env(env_path: Path) -> None:
    # .env como fuente de verdad: sobreescribe el entorno (evita tokens obsoletos).
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", choices=list(KERNELS))
    ap.add_argument("--gpu", action="store_true",
                    help="usa la G4 (NvidiaRtxPro6000); default CPU para no gastar cuota")
    ap.add_argument("--dataset", action="append", default=[],
                    help="dataset adicional user/slug; repetible")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")

    cfg = KERNELS[args.kernel]
    datasets = args.dataset or cfg.get("default_datasets", [])
    meta = {
        "id": f"{user}/{cfg['slug']}",
        "title": cfg["title"],
        "code_file": Path(cfg["notebook"]).name,
        "language": "python",
        "kernel_type": "notebook",
        # PUBLICO: las reglas del ARC Prize exigen "all code and methods must be open
        # sourced to be eligible for prizes" (milestone #2: 2026-09-30). Kernels publicos
        # + repo con licencia MIT = cumplimiento.
        "is_private": False,
        "enable_gpu": bool(args.gpu),
        "enable_internet": False,  # obligatorio en evaluación; igual que RTX_G4
        "dataset_sources": datasets,
        "competition_sources": [COMP],
        "model_sources": cfg.get("model_sources", []),
        "kernel_sources": [],
        # Algunos kernels traen su propio runtime y exigen otra imagen.
        "docker_image": cfg.get("docker_image", DOCKER_IMAGE),
    }
    if args.gpu:
        meta["machine_shape"] = "NvidiaRtxPro6000"

    tmp = Path(tempfile.mkdtemp(prefix="arc3_kernel_"))
    shutil.copy(ROOT / cfg["notebook"], tmp / meta["code_file"])
    (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("metadata:\n" + json.dumps(meta, indent=2))

    if args.dry_run:
        print(f"\n[dry-run] carpeta lista en {tmp}")
        return 0

    print(f"\nPublicando kernel '{meta['id']}' ...")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(tmp)],
                       capture_output=True, text=True)
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="")

    # Registrar la version publicada: las code competitions exigen -v en el submit
    # (daily_submit.ps1 lee este archivo). Sin esto el trigger diario falla con
    # "Code competition submissions require both the output file name and the
    # version number".
    m = re.search(r"[Kk]ernel version (\d+)", r.stdout or "")
    if r.returncode == 0 and m:
        vfile = ROOT / "kernel_versions.json"
        versions = json.loads(vfile.read_text(encoding="utf-8")) if vfile.exists() else {}
        versions[meta["id"]] = int(m.group(1))
        vfile.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
        print(f"kernel_versions.json: {meta['id']} -> v{m.group(1)}")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())

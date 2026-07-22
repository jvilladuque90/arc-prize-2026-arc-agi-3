"""Construcción de prompt e interpretación de respuesta para el agente VLM.

Diferenciador propio (ninguno del top lo hace): además de la imagen del frame, se
inyecta en el prompt una descripción TEXTUAL de la estructura del frame calculada con
nuestras features objetuales (`arc3.features`): objetos (color, tamaño, bbox, "botón-idad"),
fondo, y el efecto numérico de la última acción (píxeles cambiados, vector de movimiento).
El VLM razona sobre datos duros en vez de solo píxeles, que es donde alucina.

Todo aquí es puro (numpy + str): testeable sin GPU.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Optional

import numpy as np

from .features import connected_components, transition_features

# Paleta ARC de 16 colores (RGB) — la misma del visor oficial.
ARC_PALETTE = np.array([
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
    (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37),
    (100, 70, 30), (140, 100, 60), (90, 90, 90), (30, 30, 90), (200, 200, 255),
    (255, 255, 255),
], dtype=np.uint8)

# Nombres semánticos de acciones para el LLM (mapeo id -> nombre y viceversa).
ACTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right", 5: "action5",
                6: "click", 7: "action7"}
NAME_TO_ID = {v: k for k, v in ACTION_NAMES.items()}
NAME_TO_ID.update({"a1": 1, "a2": 2, "a3": 3, "a4": 4, "a5": 5, "a6": 6, "a7": 7,
                   "reset": 0})


def render_frame_png(grid: np.ndarray, scale: int = 8) -> bytes:
    """Grid 64x64 -> PNG RGB escalado (para el canal visual del VLM)."""
    from PIL import Image

    rgb = ARC_PALETTE[np.clip(grid, 0, 15)]
    img = Image.fromarray(rgb, "RGB").resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def frame_png_data_uri(grid: np.ndarray, scale: int = 8) -> str:
    return "data:image/png;base64," + base64.b64encode(render_frame_png(grid, scale)).decode()


def describe_objects(grid: np.ndarray, max_objects: int = 12) -> str:
    """Descripción textual compacta de los objetos del frame (nuestro diferenciador)."""
    counts = np.bincount(grid.ravel(), minlength=16)
    background = int(counts.argmax())
    objs = connected_components(grid, background)
    total = grid.size
    lines = [f"background_color={background}, distinct_colors={int((counts > 0).sum())}, "
             f"objects={len(objs)}"]
    for i, o in enumerate(objs[:max_objects]):
        y0, x0, y1, x1 = o["bbox"]
        cy, cx = o["centroid"]
        rarity = 1.0 - counts[o["color"]] / total
        area = (y1 - y0 + 1) * (x1 - x0 + 1)
        buttonness = (0.5 * rarity + 0.5 * (o["size"] / area)) * (1.0 if o["size"] <= 64 else 0.3)
        lines.append(
            f"  obj{i}: color={o['color']} size={o['size']} "
            f"bbox=(x{x0}-{x1},y{y0}-{y1}) center=(x{int(cx)},y{int(cy)}) "
            f"button_score={buttonness:.2f}")
    return "\n".join(lines)


def describe_last_transition(prev: Optional[np.ndarray], cur: np.ndarray) -> str:
    if prev is None:
        return "last_action_effect: (none, first frame)"
    tf = transition_features(prev, cur)
    mv = ""
    if tf["move_score"] > 0.6 and (tf["move_dy"] or tf["move_dx"]):
        mv = f", object_moved=(dy{tf['move_dy']},dx{tf['move_dx']})"
    return (f"last_action_effect: pixels_changed={tf['n_changed']}, "
            f"colors_gained={tf['colors_gained']}, colors_lost={tf['colors_lost']}{mv}")


SYSTEM_PROMPT = (
    "You are an agent playing an interactive puzzle game on a 64x64 colored grid "
    "(colors 0-15, coordinates x=0..63 left-to-right, y=0..63 top-to-bottom). "
    "You explore to discover the rules, then act to complete levels. "
    "Trust the numeric STRUCTURE and TRANSITION data over the image when they disagree. "
    "Respond ONLY with a JSON object: {\"reasoning\": \"...\", \"actions\": [ ... ]} where "
    "each action is either {\"name\": \"up|down|left|right|action5|action7\"} or "
    "{\"name\": \"click\", \"x\": <0-63>, \"y\": <0-63>}. Plan 1-3 actions. Prefer actions "
    "not marked ineffective. To find interactive elements, click objects with high button_score."
)


def build_user_text(
    grid: np.ndarray,
    prev_grid: Optional[np.ndarray],
    available_actions: list[int],
    levels_completed: int,
    ineffective: Optional[list[str]] = None,
    memory: Optional[str] = None,
) -> str:
    """Texto del turno: acciones legales + estructura de objetos + efecto de la última acción."""
    legal = [ACTION_NAMES[a] for a in available_actions if a in ACTION_NAMES] or \
        list(ACTION_NAMES.values())
    parts = [
        f"levels_completed={levels_completed}",
        f"legal_actions={legal}",
        "FRAME STRUCTURE:",
        describe_objects(grid),
        describe_last_transition(prev_grid, grid),
    ]
    if ineffective:
        parts.append(f"ineffective_in_this_state={ineffective[:20]}")
    if memory:
        parts.append(f"MEMORY:\n{memory}")
    parts.append("Return your JSON now.")
    return "\n".join(parts)


REFLECT_SYSTEM = (
    "You are analyzing your own play of an interactive 64x64 grid puzzle to build a memory "
    "that will guide future actions. From the transition history, infer the game's rules, the "
    "likely goal, what progress looks like, and which actions to avoid. Be concrete and concise. "
    "Respond ONLY in this markdown format (each section <=3 short bullet points, total <1500 chars):\n"
    "# Memory\n## Rules\n- ...\n## Goal\n- ...\n## Progress\n- ...\n## Avoid\n- ..."
)


def build_reflection_text(history: list[str], memory: Optional[str], levels: int) -> str:
    """Prompt de reflexión: resume el historial de transiciones en memoria accionable."""
    parts = [f"levels_completed_so_far={levels}"]
    if memory:
        parts.append("PREVIOUS MEMORY (revise if evidence contradicts it):\n" + memory)
    parts.append("RECENT TRANSITIONS (action -> numeric effect):")
    parts.extend(history[-40:])
    parts.append("Write the updated # Memory now.")
    return "\n".join(parts)


def parse_actions(text: str) -> list[dict[str, Any]]:
    """Extrae la lista de acciones del texto del LLM de forma robusta.

    Escanea todos los objetos JSON del texto y prioriza el que tenga clave 'actions'.
    Devuelve lista de dicts normalizados: {'id': int, 'x': int?, 'y': int?}.
    """
    obj = _extract_json_with_actions(text)
    raw_actions = []
    if obj and isinstance(obj.get("actions"), list):
        raw_actions = obj["actions"]
    elif obj and "name" in obj:  # el LLM devolvió una sola acción suelta
        raw_actions = [obj]
    out: list[dict[str, Any]] = []
    for a in raw_actions:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name", a.get("action", ""))).strip().lower()
        aid = NAME_TO_ID.get(name)
        if aid is None:
            continue
        entry: dict[str, Any] = {"id": aid}
        if aid == 6:
            try:
                entry["x"] = max(0, min(63, int(a.get("x", 32))))
                entry["y"] = max(0, min(63, int(a.get("y", 32))))
            except (TypeError, ValueError):
                entry["x"] = entry["y"] = 32
        out.append(entry)
    return out


def _extract_json_with_actions(text: str) -> Optional[dict[str, Any]]:
    dec = json.JSONDecoder()
    best: Optional[dict[str, Any]] = None
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            if "actions" in obj:
                return obj
            if best is None and "name" in obj:
                best = obj
        i = end
    return best

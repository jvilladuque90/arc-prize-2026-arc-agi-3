"""Tests de las partes puras del agente LLM (sin GPU): prompt, parser, fallback."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc3.llm_prompt import (  # noqa: E402
    build_user_text, describe_objects, parse_actions, render_frame_png,
)
from arc3.llm_agent import HybridAgent, LLMAgent  # noqa: E402


def _grid_with_button():
    g = np.zeros((64, 64), dtype=np.int8)
    g[10:14, 20:24] = 2   # bloque rojo (posible botón)
    g[40, 40] = 5         # pixel raro
    return g


def test_describe_objects_lists_button():
    g = _grid_with_button()
    desc = describe_objects(g)
    assert "background_color=0" in desc
    assert "button_score=" in desc
    assert "color=2" in desc


def test_render_png_nonempty():
    png = render_frame_png(_grid_with_button())
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 100


def test_parse_actions_variants():
    # objeto con 'actions'
    r = parse_actions('bla {"reasoning":"x","actions":[{"name":"up"},{"name":"click","x":5,"y":9}]} end')
    assert r == [{"id": 1}, {"id": 6, "x": 5, "y": 9}]
    # acción suelta
    assert parse_actions('{"name":"left"}') == [{"id": 3}]
    # basura -> vacío
    assert parse_actions("no json here") == []
    # click fuera de rango se recorta
    assert parse_actions('{"actions":[{"name":"click","x":999,"y":-4}]}') == [{"id": 6, "x": 63, "y": 0}]


def test_build_user_text_has_structure_and_legal():
    g = _grid_with_button()
    txt = build_user_text(g, None, [1, 2, 6], 0, ineffective=["a1"])
    assert "legal_actions=" in txt and "FRAME STRUCTURE" in txt
    assert "ineffective_in_this_state=['a1']" in txt


def test_llm_agent_uses_llm_then_falls_back():
    calls = {"n": 0}

    def chat_ok(system, user, image):
        calls["n"] += 1
        return '{"reasoning":"go","actions":[{"name":"right"}]}'

    ag = LLMAgent("g", chat_ok)
    g = _grid_with_button()
    aid, x, y = ag.choose(g, "NOT_FINISHED", 0, [1, 2, 3, 4, 6])
    assert aid == 4 and calls["n"] == 1  # tomó la acción del LLM

    def chat_bad(system, user, image):
        raise RuntimeError("llm down")

    ag2 = LLMAgent("g", chat_bad)
    aid2, _, _ = ag2.choose(g, "NOT_FINISHED", 0, [1, 2, 3, 4, 6])
    assert aid2 in (0, 1, 2, 3, 4, 5, 6, 7)  # fallback dio una acción válida
    assert ag2._llm_fails == 1


def test_llm_agent_plan_queue():
    def chat_plan(system, user, image):
        return '{"actions":[{"name":"up"},{"name":"down"},{"name":"left"}]}'

    ag = LLMAgent("g", chat_plan, plan_max=3)
    g = np.zeros((64, 64), dtype=np.int8)
    a1 = ag.choose(g, "NOT_FINISHED", 0, [1, 2, 3])[0]
    # segundo/tercer paso salen de la cola sin re-llamar (cambiamos el grid para que no aborte)
    g2 = g.copy(); g2[0, 0] = 1
    a2 = ag.choose(g2, "NOT_FINISHED", 0, [1, 2, 3])[0]
    assert a1 == 1 and a2 == 2


def test_reflection_updates_and_injects_memory():
    seen = {"reflected": False, "mem_in_prompt": False}

    def chat(system, user, image):
        if system.startswith("You are analyzing"):   # llamada de reflexión
            seen["reflected"] = True
            return "# Memory\n## Rules\n- click red to score\n## Goal\n- reach top\n## Avoid\n- edges"
        if "MEMORY:" in user:
            seen["mem_in_prompt"] = True
        return '{"actions":[{"name":"right"}]}'

    ag = LLMAgent("g", chat)
    g = np.zeros((64, 64), dtype=np.int8)
    # alterna el grid para acumular historial y disparar reflexión tras REFLECT_EVERY pasos
    for i in range(40):
        g2 = g.copy(); g2[0, i % 64] = (i % 5) + 1
        ag.choose(g2, "NOT_FINISHED", 0, [1, 2, 3, 4])
    assert seen["reflected"] and ag._memory.startswith("# Memory")
    assert seen["mem_in_prompt"]


def test_parse_goal():
    from arc3.llm_prompt import parse_goal
    assert parse_goal('{"reasoning":"x","goal":{"x":10,"y":20},"actions":[{"name":"up"}]}') == (10, 20)
    assert parse_goal('{"actions":[{"name":"up"}]}') is None
    assert parse_goal('{"goal":{"x":999,"y":-5},"actions":[]}') == (63, 0)


def test_guided_navigation_uses_motion_model():
    # el LLM fija un goal y devuelve movimientos; tras aprender el modelo de movimiento,
    # el agente debe navegar solo (nav_used > 0).
    def chat(system, user, image):
        if system.startswith("You are analyzing"):
            return "# Memory\n## Rules\n- move\n## Goal\n- reach\n## Avoid\n- x"
        return '{"goal":{"x":60,"y":30},"actions":[{"name":"right"}]}'

    ag = LLMAgent("g", chat)
    # simula frames donde ACTION4(right) traslada un objeto hacia +x de forma coherente
    base = np.zeros((64, 64), dtype=np.int8)
    for step in range(20):
        g = base.copy()
        x = min(10 + step, 62)
        g[30:33, x:x + 3] = 2   # objeto que se mueve a la derecha
        ag.choose(g, "NOT_FINISHED", 0, [1, 2, 3, 4])
    assert ag._nav_used >= 1   # navegó con el modelo aprendido en algún paso


def test_hybrid_starts_explorer_then_switches_to_llm():
    llm_used = {"n": 0}

    def chat(system, user, image):
        llm_used["n"] += 1
        return '{"actions":[{"name":"up"}]}'

    ag = HybridAgent("g", chat, stuck_actions=3)
    g = np.zeros((64, 64), dtype=np.int8)
    # sin progreso de nivel: tras stuck_actions pasos debe cambiar al LLM
    for _ in range(6):
        ag.choose(g, "NOT_FINISHED", 0, [1, 2, 3, 6])
    assert ag._using_llm is True and llm_used["n"] >= 1
    # expone atributos que el runner espera
    assert hasattr(ag, "_nodes") and isinstance(ag._llm_calls, int)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

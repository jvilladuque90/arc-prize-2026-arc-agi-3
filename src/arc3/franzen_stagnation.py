"""C1: one fresh reasoning context after a very expensive unfinished level (context management).

The policy and the helper module below are from the public Kaggle notebook `heliosli/try-merging-it-you-might-get-lucky-and-win-gold`
("V4: a fresh review after prolonged level stagnation"); credit to its author. We port them with three differences:

  * the anchors are line-ending independent and each must appear EXACTLY once (otherwise nothing is touched);
  * they apply to Franzen's already-patched `tool_agent.py` without the author's SHA-1 pin;
  * the patch is applied in Save & Run as well as in the competition rerun, so a broken patch is seen at the gate
    instead of only inside a submission (the original cell is guarded by `if TRUE_SUBMISSION`).

Policy: after 96 * 1024 generated tokens (cumulative since the level started) on a level that is still unfinished, clear the
model's reasoning history and its retained functions ONCE per level and tell it so in the next opener. The game state, the
action history, token accounting and the runtime budget are untouched.

Status: built and unit-tested; NOT validated on the hidden set. A 25-minute gate almost never reaches 96K tokens on one level,
so only the hidden rerun can exercise it. See docs/IDEAS.md (V5).
"""

from __future__ import annotations

HELPER_PATH = "inference/utils/stagnation_restart.py"

HELPER_SOURCE = r'''"""One fresh reasoning context after a very expensive unfinished level.

Only observed level progress and actual generated tokens are used. The game,
action history, death ledger, model sampling and accounting are untouched.
(Policy and code: heliosli, public Kaggle notebook; ported in arc3/franzen_stagnation.py.)
"""
import logging

LEVEL_TOKEN_LIMIT = 96 * 1024
log = logging.getLogger(__name__)


def maybe_restart(agent, frame):
    if frame is None or int(frame.level) <= 0:
        return ""
    level = int(frame.level)
    generated = int(agent._session_generated_tokens)
    pending = getattr(agent, "_stagnation_pending_notice", None)
    if pending is not None:
        pending_level, at_tokens, notice = pending
        if pending_level == level and generated <= at_tokens:
            # A failed HTTP request must not lose the explanatory fresh opener.
            agent._resume_after_yield = False
            agent._resume_reason = ""
            return notice
        agent._stagnation_pending_notice = None
    seen = getattr(agent, "_stagnation_restarted_levels", set())
    if level in seen or (agent._last_step_summary or {}).get("run_complete"):
        return ""
    spent = max(0, generated - int(agent._tokens_at_level_start))
    if spent < LEVEL_TOKEN_LIMIT:
        return ""

    # Run only between analyzer invocations, after outstanding tool calls finish.
    # The next request is a complete current-board opener, never a yield resume
    # asking the model to continue reasoning that is no longer in its context.
    old_messages = len(agent._history_messages)
    agent._history_messages = []
    agent._kept_functions = {}
    agent._retained_clear_notice = ""
    agent._resume_after_yield = False
    agent._resume_reason = ""
    agent._text_chars_per_token = None
    agent._note_history_evicted()
    agent._stagnation_restarted_levels = seen | {level}
    notice = (
        f"Fresh review of the current level: {spent} tokens of earlier reasoning "
        "have not completed this level. That reasoning and retained Python functions "
        "have been cleared once to reconsider the problem from its observations. "
        "No game action or game reset was performed by this context change: the "
        "current board, completed levels, remaining action budget, and recorded "
        "history are unchanged. Python history, transitions and death_ledger still "
        "contain the actual observations. Re-establish what counts as success and "
        "what the controls do from the board and those observations; treat earlier "
        "interpretations as hypotheses. Check a concrete prediction with a safe "
        "action before committing to another long search.\n\n"
    )
    agent._stagnation_pending_notice = (level, generated, notice)
    log.warning("stagnation_context_restart level=%d step=%d tokens=%d old_messages=%d",
                level, int(frame.step), spent, old_messages)
    return notice
'''

# (anchor, replacement) pairs on tool_agent.py, written with "\n" and converted to the file's line endings.
EDITS = (
    ("from inference.agent.python_tool_sandbox import run_sandboxed_python\n",
     "from inference.agent.python_tool_sandbox import run_sandboxed_python\n"
     "from inference.utils.stagnation_restart import maybe_restart\n"),
    ("            self._history_messages = []\n            self._session_total_tokens = 0\n",
     "            self._stagnation_restarted_levels = set()\n"
     "            self._stagnation_pending_notice = None\n"
     "            self._history_messages = []\n            self._session_total_tokens = 0\n"),
    ("        current_frame, history_entries = load_runtime_state(state_path)\n        user_prompt = self._build_user_prompt(\n",
     "        current_frame, history_entries = load_runtime_state(state_path)\n"
     "        restart_notice = maybe_restart(self, current_frame)\n"
     "        user_prompt = restart_notice + self._build_user_prompt(\n"),
)
MARKER = "maybe_restart"


def apply_to_source(src: str) -> str:
    """Return tool_agent.py with the restart policy wired in. Raises if any anchor is not unique."""
    if MARKER in src:
        return src
    nl = "\r\n" if "\r\n" in src else "\n"
    out = src
    for before, after in EDITS:
        b, a = before.replace("\n", nl), after.replace("\n", nl)
        n = out.count(b)
        if n != 1:
            raise RuntimeError(f"anchor found {n} times, expected 1: {before.strip().splitlines()[0]!r}")
        out = out.replace(b, a, 1)
    return out


def apply_to_tree(root) -> None:
    """Write the helper module and patch tool_agent.py under `root` (= .../ARC3-Inference), preserving line endings."""
    from pathlib import Path
    root = Path(root)
    ta = root / "inference" / "agent" / "tool_agent.py"
    with open(ta, "r", encoding="utf-8", newline="") as fh:
        before = fh.read()
    after = apply_to_source(before)
    compile(after, str(ta), "exec")
    with open(ta, "w", encoding="utf-8", newline="") as fh:
        fh.write(after)
    helper = root / HELPER_PATH
    compile(HELPER_SOURCE, str(helper), "exec")
    with open(helper, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(HELPER_SOURCE)

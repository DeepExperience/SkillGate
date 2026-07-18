"""Implicit instruction + reflection-context appended to system prompt at
runtime, used by SFT data collection to actively explore use-skill vs no-skill
branches plus failure-driven reflection retries.

Two design rules to keep in mind when editing this file:

1. **Phrased as environmental notes, not user commands.**
   The system prompt becomes "...[task instructions]\n\n[implicit note]".
   We deliberately phrase as "Note:" rather than "You should" to minimize
   the agent generating meta-talk like "as you instructed I won't use
   skills" — such meta-talk would leak the artificial instruction into the
   trajectory's assistant turns, contaminating SFT data even after we
   strip it from the system prompt.

2. **Exact text returned MUST be preserved as-is.**
   The collector strips this exact byte sequence from the saved trajectory's
   system message. If the runner appends ANYTHING else (extra whitespace,
   newlines, quoting), the strip will fail and SFT data will leak the
   instruction. Always: `sys_prompt = sys_prompt + "\\n\\n" + implicit_text`.
   Never insert other content between original sys_prompt and implicit_text.
"""

from __future__ import annotations


# Branch A: force skill use for SFT collection. This is intentionally stronger
# than the original soft "Note" because the current experiment optimizes for
# trajectories with strict path-based skill-file reads.
IMPLICIT_USE_SKILL = (
    "Mandatory: before attempting this task, open and read at least one "
    "retrieved skill file listed above by inspecting its SKILL.md path. Prefer "
    "the most relevant retrieved skill. Do not start solving the task until "
    "you have inspected a relevant skill file, and use the skill guidance when "
    "it applies."
)

# Branch B: discourage skill use. We still inject the skill files (so the
# prompt format is identical to use_skill), but the note tells the agent
# this particular task is best solved without them.
IMPLICIT_NO_SKILL = (
    "Note: although skill files are listed above for reference, this task is "
    "best solved using your own general knowledge of the domain. Do not open "
    "or read any files under /root/.claude/skills/ or /root/.codex/skills/ "
    "for this task; solve it directly."
)


def make_implicit_text(mode: str | None) -> str:
    """Return the exact text to append for a given implicit mode.

    Empty string for unknown / blank mode (= no implicit instruction).
    Caller appends this with `sys_prompt + "\\n\\n" + implicit_text` and
    saves the returned text in the result row so the collector can strip
    the exact same bytes later.
    """
    if not mode:
        return ""
    mode = mode.strip().lower()
    if mode == "use_skill":
        return IMPLICIT_USE_SKILL
    if mode == "no_skill":
        return IMPLICIT_NO_SKILL
    return ""


def build_reflection_text(context: str | None) -> str:
    """Wrap a reflection context (last failed attempt's summary) into a
    system-prompt suffix.

    Returns "" if context is empty/blank, so callers can unconditionally
    concatenate without a guard. The wrapper text + context together
    are what the collector strips from saved trajectories.
    """
    if not context:
        return ""
    context = context.strip()
    if not context:
        return ""
    return (
        "\n\n## Previous attempt at this task did not succeed\n"
        f"{context}\n\n"
        "Please try a meaningfully different approach this time. Do not "
        "repeat the same actions in the same order."
    )


def apply_implicit_and_reflection(
    sys_prompt: str,
    implicit_mode: str | None,
    reflection_context: str | None,
) -> tuple[str, str, str]:
    """Append implicit + reflection to a system prompt.

    Returns (new_sys_prompt, implicit_text_applied, reflection_text_applied).
    The caller stores the latter two in the result row + trajectory metadata
    so collect_successes.py can strip them from SFT export.

    Order: original sys_prompt → implicit (if any) → reflection (if any).
    Reflection is appended AFTER implicit because reflection is more recent
    information and we want the agent to attend to it last.
    """
    implicit_text = make_implicit_text(implicit_mode)
    reflection_text = build_reflection_text(reflection_context)
    if implicit_text:
        sys_prompt = sys_prompt + "\n\n" + implicit_text
    if reflection_text:
        # build_reflection_text already returns leading "\n\n" so no extra here.
        sys_prompt = sys_prompt + reflection_text
    return sys_prompt, implicit_text, reflection_text

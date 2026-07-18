#!/usr/bin/env python3
"""Smoke test for claw_grader_adapter. Doesn't touch SGLang / docker / mock services;
feeds fake OpenAI-chat messages + hard-coded audit_data to verify end-to-end
pipeline (format conversion → native grader.py → DimensionScores → 0.8/0.2 formula).

Run:
    python3 GeneralAgent/eval_scripts/unified_runner/test_claw_grader_adapter_smoke.py
Returns exit 0 on pass.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified_runner.claw_grader_adapter import (
    openai_msgs_to_native, build_native_dispatches,
    grade_with_native_grader, format_dim_scores,
)


class FakeTraj:
    """Minimal stand-in for UnifiedAgentLoop trajectory used by extract_tool_dispatches."""
    def __init__(self, messages):
        self.messages = messages
        self.turns = len(messages)
        self.time_sec = 0.0
        self.finish_reason = "completed"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.error = ""
        self.final_response = ""


def test_msg_conversion():
    """openai_msgs_to_native handles all role/content combos."""
    openai_msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Task: create an event"},
        {"role": "assistant", "content": "I'll call the API.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "exec", "arguments": '{"command": "curl -X POST ..."}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "exec",
         "content": '{"exit_code": 0, "stdout": "{\\"status\\": \\"created\\"}"}'},
        {"role": "assistant", "content": "Done. Event created."},
    ]
    native = openai_msgs_to_native(openai_msgs, trace_id="smoke")
    assert len(native) == 5, f"expected 5, got {len(native)}"
    # role mapping: tool → user
    roles = [m.message.role for m in native]
    assert roles == ["system", "user", "assistant", "user", "assistant"], roles
    # ToolUseBlock in assistant #3
    asst3 = native[2].message.content
    tus = [b for b in asst3 if b.type == "tool_use"]
    assert len(tus) == 1 and tus[0].name == "exec", tus
    # ToolResultBlock in user #4 (was role=tool)
    user4 = native[3].message.content
    trs = [b for b in user4 if b.type == "tool_result"]
    assert len(trs) == 1 and trs[0].tool_use_id == "call_1", trs
    print("  OK test_msg_conversion")


def test_grade_pinbench_with_fake_audit():
    """Feed T086 (PinbenchCalendarEventCreationGrader) with audit_data matching
    expected event → should give completion=1.0 (all 4 checks pass)."""
    openai_msgs = [
        {"role": "system", "content": "Agent system prompt"},
        {"role": "user", "content": "Create event for Q1 roadmap"},
        {"role": "assistant", "content": "Created! Q1 roadmap meeting scheduled."},
    ]
    traj = FakeTraj(openai_msgs)

    # Hand-crafted audit data matching T086 grader's expectations:
    # title == "Project Sync" / start_time startswith "2026-03-10T15:00" /
    # attendees contains "john@example.com" / final_text contains "Q1 roadmap"
    audit_data = {
        "calendar": {
            "calls": [],
            "created_events": [{
                "title": "Project Sync",
                "start_time": "2026-03-10T15:00:00",
                "attendees": ["john@example.com", "alice@example.com"],
            }],
        }
    }

    task_dir = Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[3]))) / "datasets/claw-eval/tasks/T086_pinbench_calendar_event_creation"
    tasks_dir = task_dir.parent

    passed, score, dim = grade_with_native_grader(
        task_id="T086_pinbench_calendar_event_creation",
        task_dir=task_dir,
        tasks_dir=tasks_dir,
        openai_msgs=openai_msgs,
        traj=traj,
        tool_endpoints=[],
        task_def={"task_id": "T086_pinbench_calendar_event_creation"},
        audit_data=audit_data,
    )
    print(f"  dim: {format_dim_scores(dim)}")
    print(f"  score: {score:.3f}  passed: {passed}")
    # Expect completion=1.0 (4/4 checks), robustness=1.0 (no errors), safety=1.0
    assert dim.completion == 1.0, f"completion expected 1.0 got {dim.completion}"
    assert dim.safety == 1.0, f"safety expected 1.0 got {dim.safety}"
    # score = 1.0 × (0.8 × 1.0 + 0.2 × 1.0) = 1.0
    assert score >= 0.99, f"expected ~1.0 score, got {score}"
    assert passed is True
    print("  OK test_grade_pinbench_with_fake_audit")


def test_grade_empty_audit_returns_zero():
    """Same task but audit is empty → completion=0 → score=0 → not pass."""
    traj = FakeTraj([{"role": "assistant", "content": "nothing"}])
    task_dir = Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[3]))) / "datasets/claw-eval/tasks/T086_pinbench_calendar_event_creation"
    passed, score, dim = grade_with_native_grader(
        task_id="T086_pinbench_calendar_event_creation",
        task_dir=task_dir,
        tasks_dir=task_dir.parent,
        openai_msgs=[{"role": "assistant", "content": "nothing"}],
        traj=traj,
        tool_endpoints=[],
        task_def={"task_id": "T086_pinbench_calendar_event_creation"},
        audit_data={"calendar": {"created_events": [], "calls": []}},
    )
    print(f"  dim (empty audit): {format_dim_scores(dim)}")
    # grader returns early with completion=0 / robustness=0 default / safety=1
    # → score = 1.0 × (0.8 × 0 + 0.2 × 0) = 0 → not pass
    assert dim.completion == 0.0
    assert passed is False
    print("  OK test_grade_empty_audit_returns_zero")


def main():
    print("=" * 60)
    print("claw_grader_adapter smoke test")
    print("=" * 60)
    test_msg_conversion()
    print()
    test_grade_pinbench_with_fake_audit()
    print()
    test_grade_empty_audit_returns_zero()
    print()
    print("All 3 smoke tests passed ✅")


if __name__ == "__main__":
    main()

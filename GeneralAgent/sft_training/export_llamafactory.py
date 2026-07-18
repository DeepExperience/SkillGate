#!/usr/bin/env python3
"""Export collected agent SFT messages to LLaMA-Factory OpenAI format.

Input is `sft_messages.jsonl` from collect_successes.py:
  {"messages": [...OpenAI-style...], "metadata": {...}}

Output is a JSON list plus dataset_info.json suitable for LLaMA-Factory:
  {"messages": [{"role":"system|user|assistant|tool", "content": "..."}], "metadata": {...}}

We intentionally preserve assistant.content verbatim, including OpenClaw-style
<tool_call> text. If an assistant turn has only structured `tool_calls` and no
textual tool call block, we synthesize the same XML protocol that eval parses.

The exporter writes LLaMA-Factory's `formatting: openai` format instead of
ShareGPT because agent trajectories can contain multiple consecutive tool
observations after one assistant turn. LLaMA-Factory's OpenAI converter merges
those observations correctly; the ShareGPT converter rejects them as abnormal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GeneralAgent.task_exclusions import is_bad_task


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


HIDDEN_INSTRUCTION_LEAK_MARKERS = (
    "Mandatory: before attempting this task, open and read at least one "
    "retrieved skill file listed above by inspecting its SKILL.md path.",
    "Note: although skill files are listed above for reference, this task is "
    "best solved using your own general knowledge of the domain.",
    "## Previous attempt at this task did not succeed",
    "Please try a meaningfully different approach this time. Do not repeat "
    "the same actions in the same order.",
)


def has_text_tool_call(content: str) -> bool:
    return "<tool_call>" in content and "</tool_call>" in content


def parameter_value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def tool_calls_to_text(tool_calls: Any) -> str:
    """Fallback text for assistant turns with tool_calls but no XML block.

    Unified eval parses Qwen/OpenClaw XML in the form:

        <tool_call>
        <function=exec>
        <parameter=command>...</parameter>
        </function>
        </tool_call>

    Older exporter code appended raw JSON inside `<function=...>`, which the
    parser did not interpret as arguments and, worse, duplicated tool calls when
    assistant.content already contained a textual XML block.
    """
    if not tool_calls:
        return ""
    blocks: list[str] = []
    for call in tool_calls:
        function = (call or {}).get("function", {}) if isinstance(call, dict) else {}
        name = function.get("name", "")
        if not name:
            continue
        arguments = function.get("arguments", "")
        try:
            parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except Exception:
            parsed_args = arguments
        if isinstance(parsed_args, dict):
            param_blocks = [
                f"<parameter={key}>\n{parameter_value_to_text(value)}\n</parameter>"
                for key, value in parsed_args.items()
                if isinstance(key, str) and key.isidentifier()
            ]
            if not param_blocks and parsed_args:
                param_blocks = [
                    "<parameter=arguments>\n"
                    f"{json.dumps(parsed_args, ensure_ascii=False)}\n"
                    "</parameter>"
                ]
        else:
            param_blocks = [
                "<parameter=arguments>\n"
                f"{parameter_value_to_text(parsed_args)}\n"
                "</parameter>"
            ]
        blocks.append(
            "<tool_call>\n"
            f"<function={name}>\n"
            + "\n".join(param_blocks)
            + "\n"
            f"</function>\n"
            "</tool_call>"
        )
    return "\n".join(blocks)


def merge_consecutive_tool_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    pending_tools: list[str] = []

    def flush_tools() -> None:
        if pending_tools:
            merged.append(
                {
                    "role": "tool",
                    "content": "\n</tool_response>\n<tool_response>\n".join(pending_tools),
                }
            )
            pending_tools.clear()

    for message in messages:
        if message["role"] == "tool":
            if message["content"]:
                pending_tools.append(message["content"])
            continue
        flush_tools()
        merged.append(message)

    flush_tools()
    return merged


def merge_consecutive_same_side_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize occasional tool/user or assistant/assistant adjacency.

    Agent trajectories can contain runtime reminder user messages immediately
    after a tool observation. LLaMA-Factory's OpenAI converter expects strict
    user-or-tool / assistant alternation, so we merge adjacent messages from
    the same side instead of dropping content.
    """
    normalized: list[dict[str, str]] = []
    odd_roles = {"user", "tool"}
    for message in messages:
        if (
            normalized
            and message["role"] in odd_roles
            and normalized[-1]["role"] in odd_roles
        ):
            normalized[-1]["content"] += "\n\n" + message["content"]
        elif (
            normalized
            and message["role"] == "assistant"
            and normalized[-1]["role"] == "assistant"
        ):
            normalized[-1]["content"] += "\n\n" + message["content"]
        else:
            normalized.append(dict(message))
    return normalized


def convert_record(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    messages = record.get("messages") or []
    metadata = record.get("metadata") or {}
    if not messages:
        return None, "no_messages"
    raw_text = "\n".join(stringify_content(message.get("content")) for message in messages)
    if any(marker in raw_text for marker in HIDDEN_INSTRUCTION_LEAK_MARKERS):
        return None, "hidden_instruction_leak"

    system_parts: list[str] = []
    non_system_messages: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role")
        content = stringify_content(message.get("content"))
        if role == "system":
            if content:
                system_parts.append(content)
        elif role == "user":
            if content:
                non_system_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tool_calls_text = ""
            if not has_text_tool_call(content):
                tool_calls_text = tool_calls_to_text(message.get("tool_calls"))
            if tool_calls_text:
                content = (content + "\n\n" + tool_calls_text) if content else tool_calls_text
            if content:
                non_system_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            if content:
                non_system_messages.append({"role": "tool", "content": content})
        else:
            return None, f"unknown_role:{role}"

    conversations = merge_consecutive_same_side_messages(
        merge_consecutive_tool_messages(non_system_messages)
    )
    while conversations and conversations[-1]["role"] != "assistant":
        conversations.pop()

    if not conversations:
        return None, "empty_conversations"
    if conversations[0]["role"] not in {"user", "tool"}:
        return None, "starts_with_assistant"
    if conversations[-1]["role"] != "assistant":
        return None, "last_not_assistant"

    odd_roles = {"user", "tool"}
    even_roles = {"assistant"}
    for index, turn in enumerate(conversations):
        expected = odd_roles if index % 2 == 0 else even_roles
        if turn["role"] not in expected:
            return None, f"bad_role_order:{index}:{turn['role']}"

    exported_messages: list[dict[str, str]] = []
    if system_parts:
        exported_messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    exported_messages.extend(conversations)

    return {
        "messages": exported_messages,
        "metadata": metadata,
    }, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="sft_messages.jsonl from collect_successes.py")
    parser.add_argument("--out-dir", default="GeneralAgent/sft_training/llamafactory_data")
    parser.add_argument("--dataset-name", default="agent_sft_pilot")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--include-known-bad-tasks",
        action="store_true",
        help=(
            "preserve historical records that are now on the central Docker-task "
            "exclusion list; use only for exact reconstruction of frozen datasets"
        ),
    )
    args = parser.parse_args()

    input_path = repo_path(args.input)
    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for raw_line in input_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        metadata = payload.get("metadata") or {}
        if (
            not args.include_known_bad_tasks
            and is_bad_task(metadata.get("bench", ""), metadata.get("task_id", ""))
        ):
            counts["known_bad_docker_task"] += 1
            continue
        converted, reason = convert_record(payload)
        counts[reason] += 1
        if converted is not None:
            exported.append(converted)
            if args.max_examples > 0 and len(exported) >= args.max_examples:
                break

    data_file = out_dir / f"{args.dataset_name}.json"
    data_file.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset_info = {
        args.dataset_name: {
            "file_name": data_file.name,
            "formatting": "openai",
            "columns": {
                "messages": "messages",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "tool",
                "function_tag": "function",
                "system_tag": "system",
            },
        }
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"input={input_path}")
    print(f"exported={len(exported)}")
    print(f"data_file={data_file}")
    print(f"dataset_info={out_dir / 'dataset_info.json'}")
    print("reasons=" + json.dumps(dict(counts), ensure_ascii=False, sort_keys=True))
    if not exported:
        raise SystemExit("no examples exported")


if __name__ == "__main__":
    main()

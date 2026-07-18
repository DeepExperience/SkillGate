#!/usr/bin/env python3
"""Convert collected SFT messages to OpenClaw-compatible prompt/tool format.

The converter is intentionally conservative:
  - rewrites the system prompt to the same OpenClaw-style builder used by
    unified_runner inference;
  - rewrites retrieved skills into OpenClaw's <available_skills> XML format;
  - injects the Qwen chat-template tool schema block generated from the same
    OpenClaw-compatible tool_schemas.py used by inference manual_schema mode;
  - converts legacy tool parameters where possible;
  - drops records containing non-OpenClaw tool calls instead of teaching the
    model impossible deployment-time calls.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "GeneralAgent" / "eval_scripts"))

from unified_runner.agent_loop import UnifiedAgentLoop
from unified_runner.bench_workspace_files import (
    build_workspace_files_for_bench,
    extract_claw_http_endpoints,
    extract_swe_repo_state,
    extract_user_runtime_context_tail,
    strip_runtime_context_from_user_msg,
)
from unified_runner.openclaw_compat import (
    SkillPromptEntry,
    build_openclaw_system_prompt,
    format_skills_for_openclaw,
)
from unified_runner.prompt_profiles import (
    LEGACY_11_PROFILE,
    OPENCLAW_GATED_PROFILE,
    normalize_prompt_profile,
)
from unified_runner.tool_schemas import get_default_tool_names, get_tools


DEFAULT_INPUT = (
    "GeneralAgent/sft_training/datasets/"
    "20260506_sft_campaign_1667_replace_run05_hindsight_en/sft_messages.jsonl"
)
DEFAULT_OUTPUT = (
    "GeneralAgent/sft_training/datasets/"
    "20260507_sft_campaign_1667_openclaw_full/sft_messages.jsonl"
)
DEFAULT_TOKENIZER = (
    os.environ.get("SKILLRL_ROOT", "/path/to/skillRL") + "/models/Qwen3.5-9B"
)

SHELL_MIGRATED_TOOLS = {"apply_patch", "grep", "find", "ls"}

HIDDEN_MARKERS = (
    "Mandatory: before attempting this task, open and read at least one retrieved skill file",
    "Note: although skill files are listed above for reference, this task is best solved",
    "## Previous attempt at this task did not succeed",
)

SCHEMA_END_RE = re.compile(r"</IMPORTANT>\s*", re.S)
MARKDOWN_SKILL_RE = re.compile(
    r"- \*\*`(?P<name>[^`]+)`\*\* at `(?P<path>[^`]+)`\s*\n\s*(?P<desc>[^\n]*)"
)
XML_SKILL_RE = re.compile(
    r"<skill>\s*<name>(?P<name>.*?)</name>\s*"
    r"<description>(?P<desc>.*?)</description>\s*"
    r"<location>(?P<loc>.*?)</location>\s*</skill>",
    re.S,
)
TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>"
    r"(?P<body>.*?)</function>\s*</tool_call>",
    re.S,
)
ANY_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)
FUNCTION_TAG_RE = re.compile(r"<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>")
PARAM_RE = re.compile(
    r"<parameter=(?P<key>[A-Za-z_][A-Za-z0-9_]*)>\s*(?P<value>.*?)\s*</parameter>",
    re.S,
)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def strip_schema_block(system: str) -> str:
    if system.lstrip().startswith("# Tools"):
        match = SCHEMA_END_RE.search(system)
        if match:
            return system[match.end():].lstrip()
        end = system.find("</tools>")
        if end >= 0:
            return system[end + len("</tools>"):].lstrip()
    return system


def split_old_system(system: str) -> tuple[str, str]:
    system = strip_schema_block(system)
    cut_points = [
        idx
        for idx in (
            system.find("\n**Skills available"),
            system.find("\nAvailable skills"),
            system.find("\nThe following skills provide"),
            system.find("\n## Skills (mandatory)"),
        )
        if idx >= 0
    ]
    if not cut_points:
        return system.strip(), ""
    cut = min(cut_points)
    return system[:cut].strip(), system[cut:].strip()


def normalize_skill_location(path: str) -> str:
    path = path.strip()
    if path.endswith("/"):
        path = path[:-1]
    if path.endswith("/SKILL.md") or path.endswith("SKILL.md"):
        return path
    return path + "/SKILL.md"


def extract_skills(old_system: str) -> str:
    _, skill_part = split_old_system(old_system)
    entries: list[SkillPromptEntry] = []
    for match in XML_SKILL_RE.finditer(skill_part):
        entries.append(
            SkillPromptEntry(
                name=match.group("name").strip(),
                description=match.group("desc").strip(),
                location=normalize_skill_location(match.group("loc")),
            )
        )
    if not entries:
        for match in MARKDOWN_SKILL_RE.finditer(skill_part):
            entries.append(
                SkillPromptEntry(
                    name=match.group("name").strip(),
                    description=match.group("desc").strip(),
                    location=normalize_skill_location(match.group("path")),
                )
            )
    return format_skills_for_openclaw(entries)


def workspace_for_record(metadata: dict[str, Any]) -> str:
    bench = metadata.get("bench")
    if bench == "swe_lite":
        return "/testbed"
    if bench == "claw":
        return "/workspace"
    return "/root"


def runtime_label_for_record(metadata: dict[str, Any]) -> str:
    bench = metadata.get("bench") or "unknown"
    return f"unified_runner.{bench}"


def runtime_metadata_for_record(metadata: dict[str, Any]) -> dict[str, str]:
    """Build the OpenClaw `## Runtime` metadata line from record metadata.

    Mirrors probe T086's pipe-separated format. Per-record values are
    benchmark-aware (workspace path varies); model + host fields default to
    the SFT-collection runtime so trained-time prompt is internally
    consistent with the SGLang-served qwen3.5-27b teacher.
    """
    bench = metadata.get("bench") or "unknown"
    workspace = workspace_for_record(metadata)
    return {
        "agent": "main",
        # FROZEN PROMPT CONSTANT — do not rename. This exact host string is
        # baked into the system prompts of the released SFT dataset and hence
        # into the trained models' prompt distribution. Renaming it breaks
        # byte-identical dataset reconstruction (tools/rebuild_final_sft.py
        # --compare-canonical) and creates a train/infer prompt mismatch. It
        # is a container hostname with no credential value; replace only in
        # lockstep with regenerating the dataset and retraining.
        "host": "bsud-quicksilver-vtuk-1",
        "repo": workspace,
        "os": "Linux 5.15.0-124-generic (x64)",
        "node": "v24.15.0",
        "model": "sglang/qwen3.5-27b",
        "default_model": "sglang/qwen3.5-27b",
        "shell": "bash",
        "thinking": "off",
        # Append benchmark for traceability (does not appear in probe T086 but
        # adds zero behavioral risk and lets us track which bench produced
        # this trajectory at debug time).
        "bench": bench,
    }


def build_new_system(
    old_system: str,
    metadata: dict[str, Any],
    schema_block: str,
    *,
    runtime_context_tail: str = "",
) -> str:
    skills_prompt = extract_skills(old_system)
    bench = metadata.get("bench") or "unknown"
    repo_path = ""
    repo_listing = ""
    git_log = ""
    http_endpoints = ""
    if bench == "swe_lite" and runtime_context_tail:
        repo_path, repo_listing, git_log = extract_swe_repo_state(runtime_context_tail)
    elif bench == "claw" and runtime_context_tail:
        http_endpoints = extract_claw_http_endpoints(runtime_context_tail)
    workspace_files = build_workspace_files_for_bench(
        bench,
        repo_path=repo_path,
        repo_listing=repo_listing,
        git_log=git_log,
        http_endpoints=http_endpoints,
    )
    # openclaw_full: emit all 21 OpenClaw sections for byte-alignment with
    # probe deployments. legacy_11: callee handles the legacy short prompt.
    system = build_openclaw_system_prompt(
        workspace_dir=workspace_for_record(metadata),
        skills_prompt=skills_prompt,
        sandboxed=True,
        runtime_label=runtime_label_for_record(metadata),
        runtime_metadata=runtime_metadata_for_record(metadata),
        workspace_files=workspace_files,
    )
    if schema_block:
        system = schema_block.rstrip() + "\n\n" + system
    return system


def task_context_from_old_system(old_system: str) -> str:
    context, _ = split_old_system(old_system)
    context = "\n".join(
        line for line in context.splitlines()
        if not any(marker in line for marker in HIDDEN_MARKERS)
    ).strip()
    if "You are a personal assistant running inside OpenClaw." in context:
        return ""
    if not context:
        return ""
    return (
        "## Benchmark Runtime Context\n"
        "The following benchmark-specific context was provided by the original runner. "
        "It is task context, not a replacement for the OpenClaw system prompt.\n\n"
        + context
    )


def parse_xml_params(body: str) -> dict[str, str]:
    return {m.group("key"): m.group("value").strip() for m in PARAM_RE.finditer(body)}


def shell_join(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if part is not None)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def heredoc_delimiter(text: str, base: str = "PATCH") -> str:
    delimiter = base
    suffix = 0
    while f"\n{delimiter}\n" in f"\n{text}\n":
        suffix += 1
        delimiter = f"{base}_{suffix}"
    return delimiter


def render_xml_tool_call(name: str, args: dict[str, Any]) -> str:
    parts = ["<tool_call>", f"<function={name}>"]
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = "" if value is None else str(value)
        parts.extend([f"<parameter={key}>", text, "</parameter>"])
    parts.extend([f"</function>", "</tool_call>"])
    return "\n".join(parts)


def migrate_ls_to_exec(args: dict[str, Any]) -> dict[str, Any]:
    path = args.get("path") or "."
    flag = "-la" if truthy(args.get("all")) else "-l"
    return {"command": shell_join(["ls", flag, path])}


def migrate_grep_to_exec(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return None, "grep_unmappable"
    path = args.get("path") or "."
    glob = args.get("glob")
    output_mode = str(args.get("output_mode") or "files_with_matches")
    context_lines = int_or_none(args.get("context_lines"))

    rg_parts: list[Any] = ["rg"]
    grep_parts: list[Any] = ["grep", "-RIn"]
    if truthy(args.get("case_insensitive")):
        rg_parts.append("-i")
        grep_parts.append("-i")
    if output_mode == "files_with_matches":
        rg_parts.append("-l")
        grep_parts = ["grep", "-RIl"]
    elif output_mode == "count":
        rg_parts.append("-c")
        grep_parts = ["grep", "-RIc"]
    if context_lines is not None and context_lines > 0 and output_mode == "content":
        rg_parts.extend(["-C", context_lines])
        grep_parts.extend(["-C", context_lines])
    if isinstance(glob, str) and glob.strip():
        rg_parts.extend(["-g", glob])
        grep_parts.append(f"--include={glob.split('/')[-1] or glob}")
    rg_parts.extend([pattern, path])
    grep_parts.extend([pattern, path])
    command = (
        "if command -v rg >/dev/null 2>&1; then "
        f"{shell_join(rg_parts)}; else {shell_join(grep_parts)}; fi"
    )
    return {"command": command}, None


def migrate_find_to_exec(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return None, "find_unmappable"
    path = args.get("path") or "."
    name_part = pattern.split("/")[-1] or pattern
    rg_cmd = shell_join(["rg", "--files", path, "-g", pattern])
    find_cmd = shell_join(["find", path, "-type", "f", "-name", name_part])
    command = (
        "if command -v rg >/dev/null 2>&1; then "
        f"{rg_cmd} | head -200; else {find_cmd} 2>/dev/null | head -200; fi"
    )
    return {"command": command}, None


def migrate_apply_patch_to_exec(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    patch = args.get("input")
    if patch is None:
        patch = args.get("patch")
    if not isinstance(patch, str) or not patch.strip():
        return None, "apply_patch_unmappable"
    delimiter = heredoc_delimiter(patch)
    command = f"apply_patch <<'{delimiter}'\n{patch.rstrip()}\n{delimiter}"
    return {"command": command}, None


def transform_tool_args(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    if normalize_prompt_profile() == LEGACY_11_PROFILE:
        if name not in set(get_default_tool_names()):
            return name, args, f"non_profile_tool:{name}"
        return name, args, None

    if name == "ls":
        return "exec", migrate_ls_to_exec(args), None
    if name == "grep":
        new_args, err = migrate_grep_to_exec(args)
        return ("exec", new_args or args, err)
    if name == "find":
        new_args, err = migrate_find_to_exec(args)
        return ("exec", new_args or args, err)
    if name == "apply_patch":
        new_args, err = migrate_apply_patch_to_exec(args)
        return ("exec", new_args or args, err)
    if name not in set(get_default_tool_names()):
        return name, args, f"non_openclaw_tool:{name}"
    if name == "edit":
        if "edits" not in args:
            old = args.get("old_string")
            new = args.get("new_string")
            if not isinstance(old, str) or not isinstance(new, str):
                return name, args, "edit_unmappable"
            edit: dict[str, Any] = {"oldText": old, "newText": new}
            if args.get("replace_all"):
                edit["replaceAll"] = True
            args = {"path": args.get("path"), "edits": [edit]}
    elif name == "process":
        action = args.get("action")
        args = dict(args)
        if action == "read":
            args["action"] = "log"
        elif action == "signal":
            args["action"] = "kill"
        if "pid" in args and "sessionId" not in args:
            args["sessionId"] = str(args.pop("pid"))
        args.pop("signal", None)
    return name, args, None


def transform_assistant_content(content: str) -> tuple[str, list[str]]:
    errors: list[str] = []
    allowed_tools = set(get_default_tool_names())
    migratable_tools = set() if normalize_prompt_profile() == LEGACY_11_PROFILE else SHELL_MIGRATED_TOOLS

    def repl(match: re.Match[str]) -> str:
        name = match.group("name")
        args = parse_xml_params(match.group("body"))
        new_name, new_args, err = transform_tool_args(name, args)
        if err:
            errors.append(err)
            return match.group(0)
        return render_xml_tool_call(new_name, new_args)

    transformed = TOOL_CALL_XML_RE.sub(repl, content)
    for block in ANY_TOOL_CALL_RE.findall(transformed):
        if not TOOL_CALL_XML_RE.fullmatch(block.strip()):
            # Qwen JSON tool_call blocks are okay if they name an allowed or
            # safely migratable tool; malformed XML/function names are not safe
            # for OpenClaw SFT.
            try:
                raw = re.sub(r"^<tool_call>|</tool_call>$", "", block.strip(), flags=re.S).strip()
                obj = json.loads(raw)
                name = obj.get("name")
                if name not in allowed_tools and name not in migratable_tools:
                    errors.append(f"non_profile_tool:{name}")
            except Exception:
                errors.append("malformed_tool_call_xml")
    for name in FUNCTION_TAG_RE.findall(transformed):
        if name not in allowed_tools:
            errors.append(f"non_profile_tool:{name}")
    return transformed, errors


def transform_message(message: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    message = dict(message)
    errors: list[str] = []
    if message.get("role") == "assistant":
        content = message.get("content") or ""
        if content:
            message["content"], content_errors = transform_assistant_content(content)
            errors.extend(content_errors)
        new_calls = []
        for call in message.get("tool_calls") or []:
            call = dict(call)
            function = dict(call.get("function") or {})
            name = function.get("name")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except Exception:
                args = {}
            new_name, new_args, err = transform_tool_args(name, args)
            if err:
                errors.append(err)
                continue
            function["name"] = new_name
            function["arguments"] = json.dumps(new_args, ensure_ascii=False)
            call["function"] = function
            new_calls.append(call)
        if "tool_calls" in message:
            message["tool_calls"] = new_calls
    elif message.get("role") == "tool":
        name = message.get("name")
        if normalize_prompt_profile() != LEGACY_11_PROFILE and name in SHELL_MIGRATED_TOOLS:
            message["name"] = "exec"
        elif name and name not in set(get_default_tool_names()):
            errors.append(f"non_profile_tool:{name}")
    return message, errors


def convert_record(record: dict[str, Any], schema_block: str) -> tuple[dict[str, Any] | None, str]:
    messages = record.get("messages") or []
    metadata = dict(record.get("metadata") or {})
    if not messages:
        return None, "no_messages"
    raw_text = "\n".join(str(m.get("content") or "") for m in messages)
    if any(marker in raw_text for marker in HIDDEN_MARKERS):
        return None, "hidden_instruction_leak"
    old_system = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
    # Pull the legacy "## Benchmark Runtime Context" body off the OLD SYSTEM
    # message (where phase1/phase2 collection put it). Routes per-bench info
    # (SWE repo state, Claw HTTP endpoint list, harbor generic guidance) into
    # Project Context AGENTS.md/TOOLS.md instead of the user prompt.
    # Profile-conditional: legacy_11 keeps everything in user msg.
    runtime_context_tail = ""
    is_openclaw_full = normalize_prompt_profile() != LEGACY_11_PROFILE
    if is_openclaw_full:
        # split_old_system returns (context_part, skills_part). The context
        # part holds the bench-specific runtime guidance for SWE/Claw/Harbor.
        context_part, _ = split_old_system(old_system)
        runtime_context_tail = context_part

    new_messages: list[dict[str, Any]] = []
    errors: list[str] = []
    user_context_added = False
    # Legacy task_context (claw HTTP context that pre-dates Benchmark Runtime
    # Context heading); only used by legacy_11 path.
    task_context = task_context_from_old_system(old_system) if not is_openclaw_full else ""
    user_msg_stripped = False
    for message in messages:
        role = message.get("role")
        if role == "system":
            if not any(m.get("role") == "system" for m in new_messages):
                new_messages.append(
                    {
                        "role": "system",
                        "content": build_new_system(
                            old_system,
                            metadata,
                            schema_block,
                            runtime_context_tail=runtime_context_tail,
                        ),
                    }
                )
            continue
        if role == "user" and not user_msg_stripped and is_openclaw_full:
            # First user message: strip the legacy "## Benchmark/Repository
            # Runtime Context" tail (now lives in Project Context).
            message = dict(message)
            message["content"] = strip_runtime_context_from_user_msg(
                message.get("content") or ""
            )
            user_msg_stripped = True
        if role == "user" and not user_context_added and task_context:
            message = dict(message)
            message["content"] = (message.get("content") or "").rstrip() + "\n\n" + task_context
            user_context_added = True
        new_message, msg_errors = transform_message(message)
        errors.extend(msg_errors)
        content = new_message.get("content") or ""
        if normalize_prompt_profile() != LEGACY_11_PROFILE and any(
            marker in content
            for marker in ("<parameter=old_string>", "<parameter=new_string>", "<parameter=patch>")
        ):
            errors.append("legacy_tool_parameter_leftover")
        new_messages.append(new_message)
    if errors:
        return None, errors[0]
    profile = normalize_prompt_profile()
    if profile == LEGACY_11_PROFILE:
        compat_version = "20260506_legacy_11"
    else:
        # openclaw_full: byte-aligned to probe T086 reference plus web_search
        # (28 tools, 21 rendered sections). All non-legacy profiles produce
        # the same shape.
        compat_version = "20260507_openclaw_full"
    metadata.update(
        {
            "openclaw_compat": profile != LEGACY_11_PROFILE,
            "openclaw_compat_version": compat_version,
            "prompt_profile": profile,
            "tools_schema_injected": bool(schema_block),
            "tool_schema_source": "GeneralAgent/eval_scripts/unified_runner/tool_schemas.py",
        }
    )
    return {"messages": new_messages, "metadata": metadata}, "ok"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--no-schema-block", action="store_true")
    args = parser.parse_args()

    input_path = repo_path(args.input)
    output_path = repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_block = ""
    if not args.no_schema_block:
        schema_block = UnifiedAgentLoop._render_tools_schema_block(
            args.tokenizer_path,
            get_tools(),
        )
        if not schema_block:
            raise RuntimeError("failed to render tools schema block")

    reason_counts: Counter[str] = Counter()
    bench_counts: Counter[str] = Counter()
    converted = []
    for record in read_jsonl(input_path):
        new_record, reason = convert_record(record, schema_block)
        reason_counts[reason] += 1
        if new_record is not None:
            converted.append(new_record)
            bench_counts[new_record.get("metadata", {}).get("bench", "?")] += 1

    with output_path.open("w", encoding="utf-8") as fh:
        for record in converted:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "prompt_profile": normalize_prompt_profile(),
        "input": str(input_path),
        "output": str(output_path),
        "input_records": sum(reason_counts.values()),
        "output_records": len(converted),
        "reasons": dict(reason_counts),
        "bench_counts": dict(bench_counts),
        "schema_block_chars": len(schema_block),
    }
    report_path = output_path.parent / "openclaw_compat_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

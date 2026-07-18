"""Prompt and skills formatting helpers for unified_runner.

This module is the single prompt-construction entry point for unified_runner.
Benchmark adapters should keep benchmark-specific details in the user/task
message and keep the system message controlled by `UNIFIED_PROMPT_PROFILE`.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .openclaw_probe_full_tools import OPENCLAW_FULL_28_TOOL_NAMES
from .prompt_profiles import LEGACY_11_PROFILE, normalize_prompt_profile


# Canonical OpenClaw display order for the ## Tooling list — copied verbatim
# from `src/agents/system-prompt.ts:657` `const toolOrder = [...]`. Tools in
# the active set that are NOT in this list get appended after, sorted
# alphabetically (this matches the rendering in probe T086 where dir_fetch,
# dir_list, file_fetch, file_write, memory_*, sessions_spawn, sessions_yield,
# and tts appear at the end after the canonical block).
OPENCLAW_CANONICAL_TOOL_ORDER: list[str] = [
    "read", "write", "edit", "apply_patch",
    "grep", "find", "ls",
    "exec", "process",
    "web_search", "web_fetch",
    "browser", "canvas", "nodes",
    "cron", "message", "gateway",
    "agents_list", "sessions_list", "sessions_history",
    "sessions_send", "subagents", "session_status",
    "image", "image_generate",
]

# Default set for the openclaw_full profile = probe T086 27 tools + web_search.
# The imported order mirrors the provider `tools` manifest; system-prompt
# rendering goes through `_ordered_tool_names()` to apply OpenClaw's human
# `## Tooling` order.
OPENCLAW_TOOL_ORDER: list[str] = OPENCLAW_FULL_28_TOOL_NAMES

LEGACY_11_TOOL_ORDER = [
    "read",
    "write",
    "edit",
    "apply_patch",
    "grep",
    "find",
    "ls",
    "exec",
    "process",
    "web_fetch",
    "web_search",
]

# Probe T086 verbatim summaries.
# Tools whose name maps to None get rendered as `- {name}` (no description),
# matching the OpenClaw runtime behavior when coreToolSummaries has no entry.
OPENCLAW_TOOL_SUMMARIES: dict[str, str | None] = {
    "agents_list": "List OpenClaw agent ids allowed for sessions_spawn",
    "browser": "Control web browser",
    "canvas": "Present/eval/snapshot the Canvas",
    "cron": (
        "Manage cron jobs and wake events (use for reminders; when scheduling a reminder, "
        "write the systemEvent text as something that will read like a reminder when it "
        "fires, and mention that it is a reminder depending on the time gap between setting "
        "and firing; include recent context in reminder text if appropriate)"
    ),
    "dir_fetch": None,
    "dir_list": None,
    "edit": "Make precise edits to files",
    "exec": "Run shell commands (pty available for TTY-required CLIs)",
    "file_fetch": None,
    "file_write": None,
    "gateway": "Restart, apply config, or run updates on the running OpenClaw process",
    "memory_get": None,
    "memory_search": None,
    "message": "Send messages and channel actions",
    "nodes": "List/describe/notify/camera/screen on paired nodes",
    "process": "Manage background exec sessions",
    "read": "Read file contents",
    "session_status": (
        "Show a /status-equivalent status card (usage + time + Reasoning/Verbose/Elevated); "
        'use for model-use questions (📊 session_status); optional per-session model override'
    ),
    "sessions_history": "Fetch history for another session/sub-agent",
    "sessions_list": "List other sessions (incl. sub-agents) with filters/last",
    "sessions_send": "Send a message to another session/sub-agent",
    "sessions_spawn": (
        'Spawn an isolated sub-agent session; use context="fork" only when current '
        "transcript context is required"
    ),
    "sessions_yield": None,
    "subagents": "List, steer, or kill sub-agent runs for this requester session",
    "tts": None,
    "web_fetch": "Fetch and extract readable content from a URL",
    # Legacy-11 fallbacks (used by legacy_11 profile only).
    "apply_patch": "Apply multi-file patches",
    "grep": "Search file contents for patterns",
    "find": "Find files by glob pattern",
    "ls": "List directory contents",
    "write": "Create or overwrite files",
    "web_search": "Search the web using the configured provider",
}


def _format_tool_line(name: str) -> str:
    summary = OPENCLAW_TOOL_SUMMARIES.get(name)
    return f"- {name}: {summary}" if summary else f"- {name}"


@dataclass(frozen=True)
class SkillPromptEntry:
    name: str
    description: str
    location: str


def _ordered_tool_names(tool_names: Iterable[str] | None = None) -> list[str]:
    """Return the rendered tool list ordering used by OpenClaw's ## Tooling.

    Algorithm matches OpenClaw `system-prompt.ts`: tools in the active set
    that appear in OPENCLAW_CANONICAL_TOOL_ORDER come first in that order,
    then any remaining tools (e.g. probe-specific dir_fetch, memory_*,
    sessions_spawn) are appended sorted alphabetically.
    """
    requested = list(dict.fromkeys(OPENCLAW_TOOL_ORDER if tool_names is None else tool_names))
    requested_set = set(requested)
    canonical_block = [n for n in OPENCLAW_CANONICAL_TOOL_ORDER if n in requested_set]
    extras = sorted(n for n in requested if n not in OPENCLAW_CANONICAL_TOOL_ORDER)
    return canonical_block + extras


def _ordered_legacy_tool_names(tool_names: Iterable[str] | None = None) -> list[str]:
    if tool_names is None:
        return list(LEGACY_11_TOOL_ORDER)
    requested = list(dict.fromkeys(tool_names))
    ordered = [name for name in LEGACY_11_TOOL_ORDER if name in requested]
    ordered.extend(name for name in requested if name not in ordered)
    return ordered


def build_legacy_11_system_prompt(
    *,
    workspace_dir: str,
    tool_names: Iterable[str] | None = None,
    skills_prompt: str = "",
    sandboxed: bool = False,
    runtime_label: str = "unified_runner",
) -> str:
    """Build the pre-OpenClaw unified_runner prompt family.

    This intentionally avoids the "inside OpenClaw" identity and exposes the
    older 11-tool surface. It is for ablation/backward-compatible training and
    should not be used for final OpenClaw deployment comparisons.
    """

    _ = runtime_label
    names = _ordered_legacy_tool_names(tool_names)
    shell_names = [n for n in names if n not in {"web_fetch", "web_search"}]
    web_names = [n for n in names if n in {"web_fetch", "web_search"}]
    lines = [
        "You are completing a task by calling tools and/or manipulating local files.",
        f"Your shell tools: {', '.join(shell_names)}.",
        (
            "Your web tools: "
            + (
                ", ".join(web_names)
                + " (prefer web_fetch over exec curl for external web pages)."
                if web_names
                else "none."
            )
        ),
        "",
        "Guidelines:",
        f"- Work in `{workspace_dir}` unless the task says otherwise.",
        "- Use workspace-relative paths when possible.",
        "- Use tools to inspect files, services, and command output before making changes.",
        "- For task-specific HTTP services, use `exec` with curl and POST JSON when endpoint docs say so.",
        "- For long-running shell commands, use exec(background=true) and process to inspect or stop them.",
        "- When the task is complete, stop calling tools and reply with a clear summary of what you did and why.",
        "",
    ]
    if sandboxed:
        lines.extend(
            [
                "Runtime:",
                "- Tools execute inside the benchmark sandbox/container.",
                "- Some host resources may be unavailable due to sandbox policy.",
                "",
            ]
        )
    if skills_prompt.strip():
        lines.extend(
            [
                "Skills available for this task:",
                skills_prompt.strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


DEFAULT_MODEL_ALIASES: list[str] = [
    "- local qwen3.5 27b: sglang/qwen3.5-27b",
]

DEFAULT_DOCS_PATH: str = os.environ.get(
    "UNIFIED_OPENCLAW_DOCS_PATH",
    os.path.join(
        os.environ.get("SKILLRL_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
        "GeneralAgent/openclaw_probe_runtime/openclaw-npm/node_modules/openclaw/docs",
    ),
)

# Probe T086 default workspace files. AGENTS.md content is provided; the
# four MISSING placeholders match the probe artifact exactly so trained
# models see the same "[MISSING]" markers they would see in deployment.
DEFAULT_AGENTS_MD = (
    "# AGENTS.md\n"
    "\n"
    "This workspace is an evaluation probe. Solve the user task directly.\n"
    "Use OpenClaw tools when needed. For HTTP task endpoints, use `exec` with `curl`.\n"
    "Do not run onboarding, bootstrap, memory setup, or unrelated maintenance."
)
DEFAULT_TOOLS_MD = (
    "This workspace is for an OpenClaw official-framework probe. Prefer OpenClaw tools directly.\n"
    "For HTTP evaluation tools, use `exec` with `curl`."
)


def _project_context_section(
    workspace_dir: str,
    workspace_files: dict[str, str | None] | None,
) -> list[str]:
    """Render the `# Project Context` block.

    Matches OpenClaw `buildProjectContextSection` (system-prompt.ts:135):
    - ``workspace_files=None`` or empty dict → return ``[]`` (skip the entire
      section). This corresponds to a real OpenClaw deployment where the
      user's workspace has no recognized context files.
    - Non-empty dict with ``content=None`` → render ``[MISSING] Expected at: ...``
      placeholder (matches probe T086 behavior for SOUL/IDENTITY/USER).
    - Non-empty dict with ``content`` set → inline the content.

    The caller (runner / converter) is responsible for choosing per-bench
    AGENTS.md/TOOLS.md content via ``bench_workspace_files``.
    """
    if not workspace_files:
        return []
    lines = [
        "# Project Context",
        "The following project context files have been loaded:",
        "If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; "
        "follow its guidance unless higher-priority instructions override it.",
    ]
    workspace_dir_clean = workspace_dir.rstrip("/")
    for basename, content in workspace_files.items():
        full_path = f"{workspace_dir_clean}/{basename}"
        lines.append(f"## {full_path}")
        if content is None:
            lines.append(f"[MISSING] Expected at: {full_path}")
        else:
            lines.append(content)
    return lines


def _default_runtime_metadata(workspace_dir: str = "") -> dict[str, str]:
    """Build sensible runtime metadata with os.uname()-derived host/os.

    Only the fields a benchmark runner can reasonably know are filled in;
    fields that depend on a real OpenClaw deployment (model id, agent name)
    fall back to ``unknown`` and the caller is expected to override them via
    ``build_openclaw_system_prompt(runtime_metadata=...)``.
    """
    try:
        u = os.uname()
        host = u.nodename or "unknown"
        os_str = f"{u.sysname} {u.release} ({u.machine})"
    except Exception:
        host, os_str = "unknown", "unknown"
    return {
        "agent": "main",
        "host": host,
        "repo": workspace_dir or "unknown",
        "os": os_str,
        "node": "unknown",
        "model": "unknown",
        "default_model": "unknown",
        "shell": "bash",
        "thinking": "off",
    }


def _runtime_section(
    metadata: dict[str, str] | None,
    workspace_dir: str = "",
) -> list[str]:
    if metadata is None:
        md = _default_runtime_metadata(workspace_dir)
    else:
        # Caller may pass partial dict; merge with defaults so any missing
        # field is filled with sensible value rather than disappearing.
        md = _default_runtime_metadata(workspace_dir)
        md.update(metadata)
    runtime_line = "Runtime: " + " | ".join(f"{k}={v}" for k, v in md.items())
    return [
        "## Runtime",
        runtime_line,
        "Reasoning: off (hidden unless on/stream). Toggle /reasoning; /status shows Reasoning when enabled.",
    ]


def build_openclaw_system_prompt(
    *,
    workspace_dir: str,
    tool_names: Iterable[str] | None = None,
    skills_prompt: str = "",
    direct_skill_prompt: str = "",
    sandboxed: bool = False,
    runtime_label: str = "unified_runner",
    model_aliases: list[str] | None = None,
    time_zone: str = "UTC",
    docs_path: str | None = None,
    workspace_files: dict[str, str | None] | None = None,
    runtime_metadata: dict[str, str] | None = None,
) -> str:
    """Build the OpenClaw-style system prompt for the openclaw_full profile.

    The wording mirrors `buildAgentSystemPrompt` in OpenClaw's
    `src/agents/system-prompt.ts` and the rendered output of probe T086
    (`probe_27b_openclaw/cases/T086_pinbench_calendar_event_creation/
    artifacts/system_prompt_first_request.txt`). All 21 sections probe T086
    renders are emitted unconditionally — even those whose underlying tool
    (gateway, sessions_spawn, memory_*, etc) is not implemented by the
    unified runner. Rationale: training prompt, inference prompt, and
    OpenClaw deployment prompt must match at the section-content level so
    the SFT-trained model behaves predictably when deployed onto a real
    OpenClaw runtime. The 21 not-implemented tools raise a clear
    "not implemented in this runtime" error from the ToolLayer if the
    model attempts to call them, so the agent learns to fall back.

    Args:
        workspace_dir: Per-benchmark working directory.
        tool_names: Tool list to declare in the ## Tooling section. Defaults
            to the full 27 probe tools.
        skills_prompt: Pre-rendered <available_skills> XML block.
        direct_skill_prompt: Pre-rendered full skill text to inject directly
            into the system prompt. This is used by eval ablations that test
            the value of the skill content without requiring a read decision.
        sandboxed: When True, append the ## Sandbox section.
        runtime_label: Label retained for legacy callers; not rendered.
        model_aliases: Lines under ## Model Aliases. Defaults to the probe
            single alias.
        time_zone: Value for ## Current Date & Time.
        docs_path: Override absolute docs path under ## Documentation. None
            keeps the probe's hard-coded path (good for byte-equivalence).
        workspace_files: Map of basenames (AGENTS.md/SOUL.md/USER.md/TOOLS.md)
            → file content or None (None → renders [MISSING] placeholder).
        runtime_metadata: Dict that becomes the
            ``Runtime: agent=... | host=... | model=...`` metadata line.
    """

    if normalize_prompt_profile() == LEGACY_11_PROFILE:
        return build_legacy_11_system_prompt(
            workspace_dir=workspace_dir,
            tool_names=tool_names,
            skills_prompt=skills_prompt,
            sandboxed=sandboxed,
            runtime_label=runtime_label,
        )

    _ = runtime_label  # accepted for legacy compat; openclaw_full has no such arg

    names = _ordered_tool_names(tool_names)
    tool_lines = [_format_tool_line(name) for name in names]
    tooling_trailers = [
        "TOOLS.md does not control tool availability; it is user guidance for how to use external tools.",
        "For long waits, avoid rapid poll loops: use exec with enough yieldMs or process(action=poll, timeout=<ms>).",
        "If a task is more complex or takes longer, spawn a sub-agent. Completion is push-based: it will auto-announce when done.",
        'Sub-agents start isolated by default. Use `sessions_spawn` with `context:"fork"` only when the child needs the current transcript context; otherwise omit `context` or use `context:"isolated"`.',
        "Do not poll `subagents list` / `sessions_list` in a loop; only check status on-demand (for intervention, debugging, or when explicitly asked).",
    ]

    docs_path_line = docs_path if docs_path is not None else DEFAULT_DOCS_PATH

    lines: list[str] = [
        # 1) Identity
        "You are a personal assistant running inside OpenClaw.",
        # 2) Tooling
        "## Tooling",
        "Tool availability (filtered by policy):",
        "Tool names are case-sensitive. Call tools exactly as listed.",
        *tool_lines,
        *tooling_trailers,
        # 3) Tool Call Style
        "## Tool Call Style",
        "Default: do not narrate routine, low-risk tool calls (just call the tool).",
        "Narrate only when it helps: multi-step work, complex/challenging problems, sensitive actions (e.g., deletions), or when the user explicitly asks.",
        "Keep narration brief and value-dense; avoid repeating obvious steps.",
        "Use plain human language for narration unless in a technical context.",
        "When a first-class tool exists for an action, use the tool directly instead of asking the user to run equivalent CLI or slash commands.",
        'When exec returns approval-pending, include the concrete /approve command from the tool output\'s "Reply with:" line as plain chat text for the user, and do not ask for a different or rotated code.',
        "Never execute /approve through exec or any other shell/tool path; /approve is a user-facing approval command, not a shell command.",
        "Treat allow-once as single-command only: if another elevated command needs approval, request a fresh /approve and do not claim prior approval covered it.",
        "When approvals are required, preserve and show the full command/script exactly as provided (including chained operators like &&, ||, |, ;, or multiline shells) so the user can approve what will actually run, but keep command/script previews separate from the /approve command and never substitute the shell command/script for the approval id or slug.",
        # 4) Execution Bias
        "## Execution Bias",
        "- Actionable request: act in this turn.",
        "- Non-final turn: use tools to advance, or ask for the one missing decision that blocks safe progress.",
        "- Continue until done or genuinely blocked; do not finish with a plan/promise when tools can move it forward.",
        "- Weak/empty tool result: vary query, path, command, or source before concluding.",
        "- Mutable facts need live checks: files, git, clocks, versions, services, processes, package state.",
        "- Final answer needs evidence: test/build/lint, screenshot, inspection, tool output, or a named blocker.",
        "- Longer work: brief progress update, then keep going; use background work or sub-agents when they fit.",
        # 5) Safety
        "## Safety",
        "You have no independent goals: do not pursue self-preservation, replication, resource acquisition, or power-seeking; avoid long-term plans beyond the user's request.",
        "Prioritize safety and human oversight over completion; if instructions conflict, pause and ask; comply with stop/pause/audit requests and never bypass safeguards. (Inspired by Anthropic's constitution.)",
        "Do not manipulate or persuade anyone to expand access or disable safeguards. Do not copy yourself or change system prompts, safety rules, or tool policies unless explicitly requested.",
        # 6) OpenClaw CLI Quick Reference
        "## OpenClaw CLI Quick Reference",
        "OpenClaw is controlled via subcommands. Do not invent commands.",
        "For config changes, use the first-class `gateway` tool (`config.schema.lookup`, `config.get`, `config.patch`, `config.apply`) instead of editing config through exec; the gateway tool hot-reloads config when possible and uses a safe restart only when required.",
        "Use the `gateway` tool action `restart` for Gateway restarts. Only use CLI service lifecycle commands when the user explicitly asks for them.",
        "Gateway service lifecycle quick reference:",
        "- openclaw gateway status",
        "- openclaw gateway restart",
        "Operator-only, explicit user request:",
        "- openclaw gateway start",
        "- openclaw gateway stop",
        "Do not chain `openclaw gateway stop` and `openclaw gateway start` as a restart substitute.",
        "If unsure, ask the user to run `openclaw help` (or `openclaw gateway --help`) and paste the output.",
    ]

    # 7) Skills (mandatory)
    if skills_prompt.strip():
        lines.extend(
            [
                "## Skills (mandatory)",
                "Before replying: scan <available_skills> <description> entries.",
                "- If exactly one skill clearly applies: read its SKILL.md at <location> with `read`, then follow it. You MUST use the exact <location> value from <available_skills>; never guess, fabricate, or hard-code a skill file path.",
                "- If multiple could apply: choose the most specific one, read its SKILL.md at <location> with `read`, then follow it. You MUST use the exact <location> value from <available_skills>; never guess, fabricate, or hard-code a skill file path.",
                "- If none clearly apply: do not read any SKILL.md.",
                "Constraints: never read more than one skill up front; only read after selecting.",
                "- When a skill drives external API writes, assume rate limits: prefer fewer larger writes, avoid tight one-item loops, serialize bursts when possible, and respect 429/Retry-After.",
                skills_prompt.strip(),
            ]
        )

    if direct_skill_prompt.strip():
        lines.extend(
            [
                "## Preloaded Skill Content",
                "The following single top-ranked skill has already been loaded into this system prompt.",
                "Use the skill content directly when it is relevant to the task.",
                "Do not spend a tool call reading the same SKILL.md just to access this content.",
                direct_skill_prompt.strip(),
            ]
        )

    # 8) Memory Recall
    lines.extend(
        [
            "## Memory Recall",
            "Before answering anything about prior work, decisions, dates, people, preferences, or todos: run memory_search on MEMORY.md + memory/*.md + indexed session transcripts; then use memory_get to pull only the needed lines. If low confidence after search, say you checked.",
            "Citations: include Source: <path#line> when it helps the user verify memory snippets.",
        ]
    )

    # 9) OpenClaw Self-Update
    lines.extend(
        [
            "## OpenClaw Self-Update",
            "Get Updates (self-update) is ONLY allowed when the user explicitly asks for it.",
            "Do not run config.apply or update.run unless the user explicitly requests an update or config change; if it's not explicit, ask first.",
            "Use config.schema.lookup with a specific dot path to inspect only the relevant config subtree before making config changes or answering config-field questions; avoid guessing field names/types.",
            "Actions: config.schema.lookup, config.get, config.patch (partial update, merges with existing), config.apply (validate + write full config), update.run (update deps or git, then restart). Config writes hot-reload when possible and use a safe restart only when required.",
            "After restart, OpenClaw pings the last active session automatically.",
        ]
    )

    # 10) Model Aliases + session_status hint
    aliases = list(model_aliases) if model_aliases is not None else list(DEFAULT_MODEL_ALIASES)
    lines.extend(
        [
            "## Model Aliases",
            "Prefer aliases when specifying model overrides; full provider/model is also accepted.",
            *aliases,
            "If you need the current date, time, or day of week, run session_status (📊 session_status).",
        ]
    )

    # 11) Workspace
    lines.extend(
        [
            "## Workspace",
            f"Your working directory is: {workspace_dir}",
            "Treat this directory as the single global workspace for file operations unless explicitly instructed otherwise.",
        ]
    )

    # 12) Documentation
    lines.extend(
        [
            "## Documentation",
            f"OpenClaw docs: {docs_path_line}",
            "Mirror: https://docs.openclaw.ai",
            "Source: https://github.com/openclaw/openclaw",
            "Community: https://discord.com/invite/clawd",
            "Find new skills: https://clawhub.ai",
            "For OpenClaw behavior, commands, config, or architecture: consult local docs first.",
            "For config field docs, prefer the `gateway` tool action `config.schema.lookup`; for broader config guidance, read `docs/gateway/configuration.md` and `docs/gateway/configuration-reference.md`.",
            "If docs are incomplete or stale, review the OpenClaw source on GitHub before answering.",
            "When diagnosing issues, run `openclaw status` yourself when possible; only ask the user if you lack access (e.g., sandboxed).",
        ]
    )

    # 13) Current Date & Time
    lines.extend(
        [
            "## Current Date & Time",
            f"Time zone: {time_zone}",
        ]
    )

    # 14) Workspace Files (injected) header
    lines.extend(
        [
            "## Workspace Files (injected)",
            "These user-editable files are loaded by OpenClaw and included below in Project Context.",
        ]
    )

    # 15) Assistant Output Directives
    lines.extend(
        [
            "## Assistant Output Directives",
            "Use these when you need delivery metadata in an assistant message:",
            "- `MEDIA:<path-or-url>` on its own line requests attachment delivery. The web UI strips supported MEDIA lines and renders them inline; channels still decide actual delivery behavior.",
            "- `[[audio_as_voice]]` marks attached audio as a voice-note style delivery hint. The web UI may show a voice-note badge when audio is present; channels still own delivery semantics.",
            "- To request a native reply/quote on supported surfaces, include one reply tag in your reply:",
            "- Reply tags must be the very first token in the message (no leading text/newlines): [[reply_to_current]] your reply.",
            "- [[reply_to_current]] replies to the triggering message.",
            "- Prefer [[reply_to_current]]. Use [[reply_to:<id>]] only when an id was explicitly provided (e.g. by the user or a tool).",
            "Whitespace inside the tag is allowed (e.g. [[ reply_to_current ]] / [[ reply_to: 123 ]]).",
            "- Channel-specific interactive directives are separate and should not be mixed into this web render guidance.",
            "Supported tags are stripped before user-visible rendering; support still depends on the current channel config.",
        ]
    )

    # 16) Project Context (with AGENTS.md/SOUL.md/IDENTITY.md/USER.md/TOOLS.md)
    lines.extend(_project_context_section(workspace_dir, workspace_files))

    # 17) Silent Replies
    lines.extend(
        [
            "## Silent Replies",
            "When you have nothing to say, respond with ONLY: NO_REPLY",
            "⚠️ Rules:",
            "- It must be your ENTIRE message — nothing else",
            '- Never append it to an actual response (never include "NO_REPLY" in real replies)',
            "- Never wrap it in markdown or code blocks",
            '❌ Wrong: "Here\'s help... NO_REPLY"',
            '❌ Wrong: "NO_REPLY"',
            "✅ Right: NO_REPLY",
            "",
            # 18) Cache boundary marker (must be inline as a real comment line)
            "<!-- OPENCLAW_CACHE_BOUNDARY -->",
            "",
            # 19) Messaging
            "## Messaging",
            "- Reply in current session → automatically routes to the source channel (Signal, Telegram, etc.)",
            "- Cross-session messaging → use sessions_send(sessionKey, message)",
            '- Sub-agent orchestration → use `sessions_spawn(...)` to start delegated work; omit `context` for isolated children, set `context:"fork"` only when the child needs the current transcript; use `subagents(action=list|steer|kill)` to manage already-spawned children.',
            "- Runtime-generated completion events may ask for a user update. Rewrite those in your normal assistant voice and send the update (do not forward raw internal metadata or default to NO_REPLY).",
            "- Never use exec/curl for provider messaging; OpenClaw handles all routing internally.",
            "### message tool",
            "- Use `message` for proactive sends + channel actions (polls, reactions, etc.).",
            "- For `action=send`, include `target` and `message`.",
            "- If multiple channels are configured, pass `channel` (feishu|wecom|googlechat|nostr|msteams|mattermost|nextcloud-talk|matrix|bluebubbles|line|zalo|yuanbao|zalouser|synology-chat|tlon|discord|imessage|irc|qqbot|signal|slack|telegram|twitch|whatsapp).",
            "- If you use `message` (`action=send`) to deliver your user-visible reply, respond with ONLY: NO_REPLY (avoid duplicate replies).",
        ]
    )

    # 20) Runtime
    lines.extend(_runtime_section(runtime_metadata, workspace_dir=workspace_dir))

    # 21) Sandbox (only when explicitly enabled)
    if sandboxed:
        lines.extend(
            [
                "## Sandbox",
                "You are running in a sandboxed runtime (tools execute in Docker).",
                "Some tools may be unavailable due to sandbox policy.",
            ]
        )

    # Probe T086 has no trailing newline (artifact ends with the Reasoning
    # line). Reproduce that exactly for byte-equivalence.
    return "\n".join(lines).rstrip()


def format_skills_for_openclaw(entries: Iterable[SkillPromptEntry]) -> str:
    entries = list(entries)
    if not entries:
        return ""
    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in entries:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{html.escape(skill.name)}</name>",
                f"    <description>{html.escape(skill.description or '')}</description>",
                f"    <location>{html.escape(skill.location)}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def skill_location_from_dir(skill_dir: str | Path) -> str:
    path = Path(skill_dir)
    if path.name == "SKILL.md":
        return str(path)
    return str(path / "SKILL.md")


def append_runtime_context_to_user_prompt(user_prompt: str, title: str, context: str) -> str:
    context = context.strip()
    if not context:
        return user_prompt
    return (
        f"{user_prompt.rstrip()}\n\n"
        f"## {title}\n"
        f"{context}\n"
    )


def build_http_tool_runtime_context(tool_docs: str) -> str:
    return "\n".join(
        [
            "The benchmark exposes task-specific HTTP services. They are not OpenClaw tools.",
            "Use the OpenClaw `exec` tool to call them with curl.",
            "Always POST JSON with `-H 'Content-Type: application/json' -d '{...}'` unless the endpoint docs say otherwise.",
            "Observe service responses carefully and iterate. Re-call list/get endpoints when needed to verify state.",
            "",
            tool_docs.strip(),
        ]
    ).strip()


def build_harbor_runtime_context() -> str:
    return "\n".join(
        [
            "You are inside a benchmark task container.",
            "Work in `/root` unless the task says otherwise.",
            "Use `exec` with shell commands such as `ls`, `find`, and `grep`, plus `read` for file contents, to inspect files before modifying them.",
            "Use `exec` for shell commands and check command output carefully.",
            "Install missing dependencies with apt-get or pip only when needed for the task.",
        ]
    )


def build_swe_runtime_context(repo_path: str, repo_listing: str, git_log: str) -> str:
    return "\n".join(
        [
            "You are fixing a bug in a Python repository.",
            f"Repository location: {repo_path}",
            "",
            "Repository structure:",
            repo_listing.strip(),
            "",
            "Recent git history:",
            git_log.strip(),
            "",
            "Workflow:",
            "1. Understand the issue description.",
            "2. Use exec with shell commands such as grep/find, plus read for file contents, to locate and inspect relevant source files.",
            "3. Make the minimal targeted fix.",
            "4. Use exec to run relevant tests when practical.",
            "5. Stop calling tools when the fix is complete.",
            "",
            "Important:",
            "- The repo is already checked out at the correct commit.",
            "- Work in the repository path above.",
            "- Keep changes minimal and targeted.",
        ]
    )


def default_skill_root_for_prompt() -> str:
    return os.environ.get("UNIFIED_SKILL_PROMPT_ROOT", "/root/.claude/skills")

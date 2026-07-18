"""Tool schemas in OpenAI function calling format.

Two prompt/tool profiles are supported through `UNIFIED_PROMPT_PROFILE`.

`openclaw_full` OpenClaw deployment-compatible profile:
the 27 tools captured from a real OpenClaw provider request plus `web_search`
in the provider `tools` manifest order. The system prompt's human `## Tooling`
section separately renders `web_search` before `web_fetch`, matching
OpenClaw's `system-prompt.ts` order. The unified runner
implements read, write, edit, exec, process, web_search, and web_fetch; GUI,
messaging, session, media, cron, gateway, and memory tools are advertised for
prompt/deployment alignment but return clear "not implemented" errors in this
benchmark runtime. Convenience tools from older unified_runner data
(apply_patch, grep, find, ls) remain in TOOL_REGISTRY for backward-compatible
trajectory migration, but are not exposed by default.

`legacy_11` pre-OpenClaw unified_runner profile:
read, write, edit, apply_patch, grep, find, ls, exec, process, web_fetch,
web_search, with legacy edit/process/exec schemas.

Usage:
    from unified_runner.tool_schemas import get_tools
    tools = get_tools()                      # active profile default
    tools = get_tools(include=["exec","read"])  # subset
"""

from __future__ import annotations

from typing import Any

from .openclaw_probe_full_tools import (
    OPENCLAW_FULL_28_TOOLS,
    OPENCLAW_FULL_28_TOOL_NAMES,
)
from .prompt_profiles import (
    LEGACY_11_PROFILE,
    OPENCLAW_FULL_PROFILE,
    OPENCLAW_GATED_PROFILE,  # back-compat alias for OPENCLAW_FULL_PROFILE
    normalize_prompt_profile,
)

# --- read -------------------------------------------------------------------

READ_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read",
        "description": (
            "Read file contents. Returns numbered lines (cat -n style). "
            "Supports offset/limit for partial reads."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read.",
                },
            },
            "required": ["path"],
        },
    },
}

# --- write ------------------------------------------------------------------

WRITE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write",
        "description": (
            "Write full content to a file. Creates parent directories if needed. "
            "Overwrites the existing file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
}

# --- edit -------------------------------------------------------------------

EDIT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Make precise edits to a file. Provide one or more exact text "
            "replacements as edits=[{oldText,newText}]."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Exact text to find.",
                            },
                            "newText": {
                                "type": "string",
                                "description": "Replacement text.",
                            },
                        },
                        "required": ["oldText", "newText"],
                    },
                    "description": "One or more exact replacements to apply in order.",
                },
            },
            "required": ["path", "edits"],
        },
    },
}

# --- apply_patch ------------------------------------------------------------

APPLY_PATCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "apply_patch",
        "description": (
            "Apply a patch to one or more files using the OpenClaw "
            "*** Begin Patch / *** End Patch format."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Patch content using the *** Begin Patch/End Patch format.",
                },
            },
            "required": ["input"],
        },
    },
}

# --- grep -------------------------------------------------------------------

GREP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search file contents using regex patterns. "
            "Returns matching lines or file paths."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to working directory.",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter (e.g. '*.py').",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search.",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode. Default: files_with_matches.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines before and after each match.",
                },
            },
            "required": ["pattern"],
        },
    },
}

# --- find -------------------------------------------------------------------

FIND_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "find",
        "description": (
            "Find files by glob pattern. "
            "Returns matching file paths sorted by modification time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.ts').",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to working directory.",
                },
            },
            "required": ["pattern"],
        },
    },
}

# --- ls ---------------------------------------------------------------------

LS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ls",
        "description": "List directory contents with file sizes and types.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path. Defaults to working directory.",
                },
                "all": {
                    "type": "boolean",
                    "description": "Include hidden files (default false).",
                },
            },
            "required": [],
        },
    },
}

# --- exec -------------------------------------------------------------------

EXEC_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "exec",
        "description": (
            "Execute a shell command and return its output. "
            "Working directory persists between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default 120).",
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for this command (defaults to current cwd).",
                },
                "yieldMs": {
                    "type": "number",
                    "description": "Milliseconds to wait before backgrounding (accepted for OpenClaw compatibility).",
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background immediately.",
                },
                "pty": {
                    "type": "boolean",
                    "description": "Run in a pseudo-terminal when available (accepted for OpenClaw compatibility).",
                },
                "elevated": {
                    "type": "boolean",
                    "description": "Run elevated if allowed (not enabled in unified runner).",
                },
            },
            "required": ["command"],
        },
    },
}

# --- process ----------------------------------------------------------------

PROCESS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "process",
        "description": (
            "Manage background exec sessions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "poll", "log", "write", "send-keys", "submit", "paste", "kill", "clear", "remove"],
                    "description": "Process action.",
                },
                "sessionId": {
                    "type": "string",
                    "description": "Session id for actions other than list.",
                },
                "data": {
                    "type": "string",
                    "description": "Data to write for write.",
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key tokens to send for send-keys.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to paste for paste.",
                },
                "offset": {
                    "type": "number",
                    "description": "Log offset.",
                },
                "limit": {
                    "type": "number",
                    "description": "Log length.",
                },
                "timeout": {
                    "type": "number",
                    "description": "For poll: wait up to this many milliseconds before returning.",
                },
            },
            "required": ["action"],
        },
    },
}

LEGACY_EDIT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Perform exact string replacement in a file. Fails if old_string "
            "is not found or not unique (unless replace_all=true)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to find and replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default false).",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}

LEGACY_EXEC_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "exec",
        "description": "Execute a shell command and return its output. Working directory persists between calls.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120).",
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background (default false).",
                },
            },
            "required": ["command"],
        },
    },
}

LEGACY_PROCESS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "process",
        "description": "Manage long-running processes. List running processes, read their output, or send signals.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "read", "signal"],
                    "description": "Action to perform on processes.",
                },
                "pid": {
                    "type": "integer",
                    "description": "Process ID (required for read/signal).",
                },
                "signal": {
                    "type": "string",
                    "description": "Signal name for signal action (e.g. SIGTERM, SIGKILL).",
                },
            },
            "required": ["action"],
        },
    },
}

# --- web_fetch --------------------------------------------------------------

WEB_FETCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch a URL and return its text content. By default routes through "
            "r.jina.ai reader to get clean markdown-like text. Use this instead "
            "of `exec curl <url>` when you need readable page content for research."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute URL to fetch (http:// or https://).",
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "If true (default), route through r.jina.ai "
                                   "reader for clean text. Set false to get raw HTML.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max chars to return (default 8000).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 60).",
                },
            },
            "required": ["url"],
        },
    },
}

# --- web_search -------------------------------------------------------------

WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web and return a list of results (title, url, snippet). "
            "Requires one of: EXA_API_KEY, GOOGLE_CSE_API_KEY+GOOGLE_CSE_ID. "
            "If no key configured, returns an error suggesting web_fetch instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                },
            },
            "required": ["query"],
        },
    },
}

# --- registry ---------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "read": READ_SCHEMA,
    "write": WRITE_SCHEMA,
    "edit": EDIT_SCHEMA,
    "apply_patch": APPLY_PATCH_SCHEMA,
    "grep": GREP_SCHEMA,
    "find": FIND_SCHEMA,
    "ls": LS_SCHEMA,
    "exec": EXEC_SCHEMA,
    "process": PROCESS_SCHEMA,
    "web_fetch": WEB_FETCH_SCHEMA,
    "web_search": WEB_SEARCH_SCHEMA,
}

LEGACY_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    **TOOL_REGISTRY,
    "edit": LEGACY_EDIT_SCHEMA,
    "exec": LEGACY_EXEC_SCHEMA,
    "process": LEGACY_PROCESS_SCHEMA,
}

OPENCLAW_DEPLOY_TOOL_NAMES: list[str] = [
    "read",
    "write",
    "edit",
    "exec",
    "process",
    "web_fetch",
    "web_search",
]

LEGACY_11_TOOL_NAMES: list[str] = [
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

EXTENDED_UNIFIED_TOOL_NAMES: list[str] = list(TOOL_REGISTRY.keys())

# Implemented 7-tool subset retained for host-mode smoke tests and callers that
# need only tools the unified runner can execute directly.
CORE_TOOLS: list[dict[str, Any]] = [
    TOOL_REGISTRY[name] for name in OPENCLAW_DEPLOY_TOOL_NAMES
]

# openclaw_full profile: 27 probe tools + web_search = 28. The 27 are
# verbatim from probe T086; web_search is added because OpenClaw enables it
# whenever a provider key is configured (i.e. any non-probe deployment).
# Models trained on this list will see web_search in the schema block; if a
# specific deployment has it disabled, the prompt-conditional rendering at
# OpenClaw's side will simply omit it from the system text.
OPENCLAW_FULL_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    t["function"]["name"]: t for t in OPENCLAW_FULL_28_TOOLS
}

PROMPT_PROFILE_TOOL_NAMES: dict[str, list[str]] = {
    OPENCLAW_FULL_PROFILE: list(OPENCLAW_FULL_28_TOOL_NAMES),
    LEGACY_11_PROFILE: LEGACY_11_TOOL_NAMES,
}


def get_default_tool_names(profile: str | None = None) -> list[str]:
    resolved = normalize_prompt_profile(profile)
    return list(PROMPT_PROFILE_TOOL_NAMES[resolved])


def get_tool_registry(profile: str | None = None) -> dict[str, dict[str, Any]]:
    resolved = normalize_prompt_profile(profile)
    if resolved == LEGACY_11_PROFILE:
        return LEGACY_TOOL_REGISTRY
    if resolved == OPENCLAW_FULL_PROFILE:
        return OPENCLAW_FULL_TOOL_REGISTRY
    return TOOL_REGISTRY  # safety fallback (should be unreachable)


def get_tools(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return tool schemas in the active profile, optionally filtered.

    For openclaw_full profile, schemas are returned in the OpenClaw provider
    `tools` manifest order so the SGLang chat-template-rendered schema block
    matches OpenClaw deployment requests.
    """
    profile = normalize_prompt_profile()
    registry = get_tool_registry(profile)
    default_names = get_default_tool_names(profile)
    requested_names = set(default_names if include is None else registry.keys())
    if include is not None:
        requested_names &= set(include)
    if exclude is not None:
        requested_names -= set(exclude)
    # Preserve profile-defined order, not dict iteration order.
    ordered = (
        default_names if include is None else [n for n in registry if n in requested_names]
    )
    return [registry[n] for n in ordered if n in requested_names]


# Mapping from OpenClaw names to Claw-Eval (Claude Code) names
OPENCLAW_TO_CLAW_EVAL: dict[str, str] = {
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "grep": "Grep",
    "find": "Glob",
    "ls": "Bash",      # ls → Bash("ls ...")
    "exec": "Bash",
    "process": "Bash",  # process → Bash("ps ...")
    "apply_patch": "Bash",  # apply_patch → Bash("patch ...")
    "web_fetch": "WebFetch",
    "web_search": "WebSearch",
}

# Mapping from SWE-Gym tool names to OpenClaw names
SWE_TO_OPENCLAW: dict[str, str] = {
    "read_file": "read",
    "write_file": "write",
    "run_command": "exec",
    "str_replace": "edit",
    "view": "read",
    "create": "write",
    "insert": "edit",
    "search_tool": "grep",
    "execute_bash": "exec",
}

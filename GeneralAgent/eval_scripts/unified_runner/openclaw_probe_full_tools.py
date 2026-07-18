"""OpenClaw tool schemas for our SFT/eval prompts.

Two flavors are exposed:

- `OPENCLAW_PROBE_27_TOOLS` — verbatim copy of probe T086's
  `tools_first_request.json`. Use this for byte-equivalence regression tests
  against the probe artifact. **Does NOT include web_search** because the
  probe runtime did not have a web-search provider configured.

- `OPENCLAW_FULL_28_TOOLS` — probe 27 + `web_search`. This is the
  "OpenClaw default deployment" target: web_search is enabled by default in
  OpenClaw whenever a provider API key is present, so any real user
  deployment is a strict superset of probe T086 *plus* web_search. This is
  what training data targets.

To refresh after an OpenClaw runtime/plugin change: re-extract a probe
trajectory artifact, overwrite `openclaw_probe_27_tools.json`, regenerate
`openclaw_full_28_tools.json` from it (re-add web_search), and restart any
process that imported this module.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROBE_JSON = _HERE / "openclaw_probe_27_tools.json"
_FULL_JSON = _HERE / "openclaw_full_28_tools.json"

OPENCLAW_PROBE_27_TOOLS: list[dict] = json.loads(_PROBE_JSON.read_text(encoding="utf-8"))
OPENCLAW_FULL_28_TOOLS: list[dict] = json.loads(_FULL_JSON.read_text(encoding="utf-8"))

# Probe T086 tool names in the order OpenClaw renders them in the
# `## Tooling` section: canonical OpenClaw display order first
# (system-prompt.ts:657 toolOrder constant), then alphabetical extras.
# Matches the probe artifact line-for-line.
OPENCLAW_PROBE_27_TOOL_NAMES: list[str] = [
    "read", "write", "edit",
    "exec", "process",
    "web_fetch",
    "browser", "canvas", "nodes",
    "cron", "message", "gateway",
    "agents_list", "sessions_list", "sessions_history",
    "sessions_send", "subagents", "session_status",
    "dir_fetch", "dir_list", "file_fetch", "file_write",
    "memory_get", "memory_search",
    "sessions_spawn", "sessions_yield", "tts",
]

# Full-deployment provider-tool list: probe 27 + web_search inserted where an
# enabled OpenClaw web-search tool appears in the provider `tools` manifest.
# This order is intentionally the provider manifest order, not the human
# `## Tooling` rendering order; `openclaw_compat.py` separately applies
# system-prompt.ts's display ordering for the system prompt text.
OPENCLAW_FULL_28_TOOL_NAMES: list[str] = [
    "agents_list", "browser", "canvas", "cron",
    "dir_fetch", "dir_list",
    "edit", "exec",
    "file_fetch", "file_write",
    "gateway",
    "memory_get", "memory_search",
    "message", "nodes",
    "process", "read",
    "session_status",
    "sessions_history", "sessions_list", "sessions_send",
    "sessions_spawn", "sessions_yield",
    "subagents", "tts",
    "web_fetch", "web_search",
    "write",
]

# Sanity: the curated orderings must equal the schema-name sets verbatim.
_probe_names = {t["function"]["name"] for t in OPENCLAW_PROBE_27_TOOLS}
assert _probe_names == set(OPENCLAW_PROBE_27_TOOL_NAMES), (
    f"OPENCLAW_PROBE_27_TOOL_NAMES drift vs JSON: "
    f"missing={set(OPENCLAW_PROBE_27_TOOL_NAMES) - _probe_names}, "
    f"extra={_probe_names - set(OPENCLAW_PROBE_27_TOOL_NAMES)}"
)
_full_names = {t["function"]["name"] for t in OPENCLAW_FULL_28_TOOLS}
assert _full_names == set(OPENCLAW_FULL_28_TOOL_NAMES), (
    f"OPENCLAW_FULL_28_TOOL_NAMES drift vs JSON: "
    f"missing={set(OPENCLAW_FULL_28_TOOL_NAMES) - _full_names}, "
    f"extra={_full_names - set(OPENCLAW_FULL_28_TOOL_NAMES)}"
)

"""Central prompt/tool profile selection for unified_runner.

`UNIFIED_PROMPT_PROFILE` controls the two supported SFT/eval prompt families:

- openclaw_full: byte-aligned OpenClaw deployment-style prompt and the full
  28 tool schemas (probe 27 plus web_search when the deployment has search
  configured). Matches what a real OpenClaw runtime sends to the model. The
  unified runner only implements 7 of those 28 (read, write, edit, exec,
  process, web_search, web_fetch); the other tools (browser, canvas, gateway,
  sessions_*, memory_*, etc) raise a clear "not implemented in this runtime"
  error so the agent learns to fall back to exec/curl. Use this for any
  train/eval intended to deploy onto an OpenClaw runtime.
- legacy_11: pre-OpenClaw unified_runner prompt and 11 tools (read, write,
  edit, apply_patch, grep, find, ls, exec, process, web_fetch, web_search).
  Kept for backward compatibility with v1-v4 SFT data only.

Keep this as the only profile switch so prompt text and tool schemas do not
silently diverge.
"""

from __future__ import annotations

import os

PROMPT_PROFILE_ENV = "UNIFIED_PROMPT_PROFILE"
OPENCLAW_FULL_PROFILE = "openclaw_full"
LEGACY_11_PROFILE = "legacy_11"
# Back-compat alias retained so callers still importing the old constant
# do not break. The underlying profile is now openclaw_full.
OPENCLAW_GATED_PROFILE = OPENCLAW_FULL_PROFILE
SUPPORTED_PROMPT_PROFILES = (OPENCLAW_FULL_PROFILE, LEGACY_11_PROFILE)

_ALIASES = {
    "": OPENCLAW_FULL_PROFILE,
    "openclaw": OPENCLAW_FULL_PROFILE,
    "openclaw_full": OPENCLAW_FULL_PROFILE,
    "openclaw-full": OPENCLAW_FULL_PROFILE,
    "full": OPENCLAW_FULL_PROFILE,
    # Legacy aliases that used to map to the 7-tool gated profile now map
    # to the new full profile, since the gated profile has been removed.
    "openclaw_gated": OPENCLAW_FULL_PROFILE,
    "openclaw-gated": OPENCLAW_FULL_PROFILE,
    "gated": OPENCLAW_FULL_PROFILE,
    "legacy": LEGACY_11_PROFILE,
    "legacy_11": LEGACY_11_PROFILE,
    "legacy-11": LEGACY_11_PROFILE,
    "old": LEGACY_11_PROFILE,
    "old_11": LEGACY_11_PROFILE,
}


def normalize_prompt_profile(value: str | None = None) -> str:
    raw = os.environ.get(PROMPT_PROFILE_ENV, "") if value is None else value
    key = raw.strip().lower()
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"{PROMPT_PROFILE_ENV}={raw!r} invalid; "
            f"expected one of {', '.join(SUPPORTED_PROMPT_PROFILES)}"
        ) from exc


def is_legacy_11_profile(value: str | None = None) -> bool:
    return normalize_prompt_profile(value) == LEGACY_11_PROFILE

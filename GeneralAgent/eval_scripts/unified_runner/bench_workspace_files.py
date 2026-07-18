"""Per-bench Project Context file generators.

OpenClaw renders the ``# Project Context`` section by inlining the content of
context files (AGENTS.md, TOOLS.md, SOUL.md, IDENTITY.md, USER.md) found in
the workspace at startup. Each user has different content; the model trained
on probe-specific hardcoded content would be brittle.

Instead, we generate per-bench AGENTS.md + TOOLS.md content that:

1. accurately describes each benchmark's task setup (so the prompt is
   semantically correct for that bench);
2. keeps the user message clean (just the task description, not a wall of
   "Benchmark Runtime Context" boilerplate);
3. varies across benches at training time so the SFT model learns to adapt
   to whatever content shows up in a real OpenClaw deployment.

Helpers:
  - ``build_workspace_files_for_bench`` — main entry point; returns
    ``{basename: content}`` dict for the runner / converter to pass to
    ``build_openclaw_system_prompt(workspace_files=...)``.
  - ``strip_runtime_context_from_user_msg`` — removes the legacy ``## Benchmark
    Runtime Context`` / ``## Repository Runtime Context`` tails so the user
    message is just the task prompt.
  - ``extract_swe_repo_state`` / ``extract_claw_http_endpoints`` — parse
    legacy user-msg tails into per-bench content (used by the converter to
    reuse historical SFT data).
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# AGENTS.md
# ---------------------------------------------------------------------------


def build_agents_md(
    bench: str,
    *,
    repo_path: str = "",
    repo_listing: str = "",
    git_log: str = "",
) -> str:
    """Return inlined AGENTS.md content. The leading ``# AGENTS.md`` heading
    matches the probe convention (its real on-disk AGENTS.md starts with that
    markdown header)."""
    header = "# AGENTS.md\n\n"
    if bench == "claw":
        body = (
            "This workspace is a Claw-Eval benchmark task running on the host.\n"
            "\n"
            "Solve the user task directly using the HTTP services exposed by the "
            "benchmark (see TOOLS.md for endpoint conventions and the per-task "
            "service list).\n"
            "\n"
            "Do not run onboarding, bootstrap, memory setup, or unrelated maintenance."
        )
        return header + body
    if bench in ("tb2", "sb_ns", "seta_synth"):
        body = (
            "This workspace is a benchmark task running inside a Docker container.\n"
            "\n"
            "Solve the task directly. Work in `/root` unless the task description "
            "says otherwise. Do not run onboarding, bootstrap, memory setup, or "
            "unrelated maintenance."
        )
        return header + body
    if bench == "swe_lite":
        sections = [
            "This workspace is a SWE-Bench bug fix task.",
            "",
            f"Repository location: {repo_path or '/testbed'}",
            "The repository is already checked out at the correct commit.",
            "Make minimal, targeted changes to fix the issue described in the user message.",
            "",
            "Workflow:",
            "1. Understand the issue description.",
            "2. Use exec with shell commands such as grep/find, plus read for file contents, to locate and inspect relevant source files.",
            "3. Make the minimal targeted fix.",
            "4. Use exec to run relevant tests when practical.",
            "5. Stop calling tools when the fix is complete.",
        ]
        if repo_listing:
            sections.extend(["", "Repository structure:", repo_listing.strip()])
        if git_log:
            sections.extend(["", "Recent git history:", git_log.strip()])
        return header + "\n".join(sections)
    # Generic fallback for any unknown bench.
    body = (
        f"This workspace is a {bench} benchmark task. "
        "Solve the user task directly without onboarding or unrelated maintenance."
    )
    return header + body


# ---------------------------------------------------------------------------
# TOOLS.md
# ---------------------------------------------------------------------------


def build_tools_md(bench: str, *, http_endpoints: str = "") -> str:
    if bench == "claw":
        base = (
            "Use the OpenClaw `exec` tool to call task-specific HTTP services with curl.\n"
            "Always POST JSON with `-H 'Content-Type: application/json' -d '{...}'` "
            "unless the endpoint docs say otherwise. Observe service responses "
            "carefully and iterate; re-call list/get endpoints to verify state."
        )
        if http_endpoints.strip():
            return base + "\n\n" + http_endpoints.strip()
        return base
    if bench in ("tb2", "sb_ns", "seta_synth"):
        return (
            "Use `exec` with shell commands (ls, find, grep, cat) to inspect the workspace.\n"
            "Use `read` for file contents before modifying. Use `edit` or `write` for changes. "
            "Install missing dependencies with apt-get or pip only when needed for the task."
        )
    if bench == "swe_lite":
        return (
            "Use `exec` with grep/find to locate code, plus `read` for file contents.\n"
            "Use `edit` for in-place fixes or `apply_patch` for multi-file diffs. "
            "Use `exec` to run relevant tests when practical."
        )
    return (
        "Use the available OpenClaw tools (read, write, edit, exec, process, "
        "web_fetch) as appropriate to the task."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_workspace_files_for_bench(
    bench: str,
    *,
    repo_path: str = "",
    repo_listing: str = "",
    git_log: str = "",
    http_endpoints: str = "",
) -> dict[str, str | None]:
    """Return dict of context-file basenames → inline content for this bench.

    Mirrors the probe T086 convention: AGENTS.md and TOOLS.md carry real
    content, while SOUL.md / IDENTITY.md / USER.md are listed with
    ``content=None`` so OpenClaw renders a ``[MISSING] Expected at: ...``
    placeholder. Order matches OpenClaw's
    ``CONTEXT_FILE_ORDER`` (agents, soul, identity, user, tools).
    """
    return {
        "AGENTS.md": build_agents_md(
            bench,
            repo_path=repo_path,
            repo_listing=repo_listing,
            git_log=git_log,
        ),
        "SOUL.md": None,
        "IDENTITY.md": None,
        "USER.md": None,
        "TOOLS.md": build_tools_md(bench, http_endpoints=http_endpoints),
    }


# ---------------------------------------------------------------------------
# User-message tail handling
# ---------------------------------------------------------------------------


def strip_runtime_context_from_user_msg(user_msg: str) -> str:
    """Remove the legacy ``## Benchmark Runtime Context`` / ``## Repository
    Runtime Context`` tail so the user message is just the task prompt."""
    cuts = []
    for marker in (
        "\n## Benchmark Runtime Context",
        "\n## Repository Runtime Context",
    ):
        idx = user_msg.find(marker)
        if idx >= 0:
            cuts.append(idx)
    if not cuts:
        return user_msg.rstrip()
    return user_msg[: min(cuts)].rstrip()


def extract_user_runtime_context_tail(user_msg: str) -> str:
    """Return everything starting at the ``## Benchmark/Repository Runtime
    Context`` heading (used by the converter to reuse historical content)."""
    cuts = []
    for marker in (
        "\n## Benchmark Runtime Context",
        "\n## Repository Runtime Context",
    ):
        idx = user_msg.find(marker)
        if idx >= 0:
            cuts.append(idx)
    if not cuts:
        return ""
    return user_msg[min(cuts):].lstrip()


def extract_swe_repo_state(runtime_context_tail: str) -> tuple[str, str, str]:
    """Parse ``Repository location`` / ``Repository structure`` / ``Recent
    git history`` blocks from a SWE-Bench runtime-context tail.

    The legacy phase1 collector wrote the section without strict double-newline
    spacing; we use the next section heading as the right anchor instead.
    """
    repo_path = ""
    repo_listing = ""
    git_log = ""
    m = re.search(r"Repository location:\s*([^\n]+)", runtime_context_tail)
    if m:
        repo_path = m.group(1).strip()
    m = re.search(
        r"Repository structure:\s*\n(.*?)(?=\n*Recent git history:|\n*Workflow:|\n*Your workflow|\n*Important:|\n*You have access|\Z)",
        runtime_context_tail,
        re.S,
    )
    if m:
        repo_listing = m.group(1).strip()
    m = re.search(
        r"Recent git history:\s*\n(.*?)(?=\n*Workflow:|\n*Your workflow|\n*Important:|\n*You have access|\Z)",
        runtime_context_tail,
        re.S,
    )
    if m:
        git_log = m.group(1).strip()
    return repo_path, repo_listing, git_log


def extract_claw_http_endpoints(runtime_context_tail: str) -> str:
    """Return the ``**HTTP Tools available...**`` block (verbatim) from a
    Claw-Eval runtime-context tail. Empty string if the task has no HTTP
    tools (legacy renderer wrote a placeholder line in that case)."""
    m = re.search(
        r"(\*\*HTTP Tools available[^*]*\*\*\s*\n+.*?)(?:\Z|\n\n##)",
        runtime_context_tail,
        re.S,
    )
    if m:
        block = m.group(1).strip()
        # Drop the legacy "no HTTP tools" placeholder.
        if "no HTTP tools" in block.lower():
            return ""
        return block
    return ""

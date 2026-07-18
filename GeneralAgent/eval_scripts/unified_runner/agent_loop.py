"""Unified agent loop: LLM + OpenClaw tool calling.

Drives a function-calling loop where the LLM decides which tools to use,
the ToolLayer executes them, and the results are fed back to the model.
The full conversation is captured as a trajectory suitable for SFT training.

Usage:
    from unified_runner.agent_loop import UnifiedAgentLoop
    from unified_runner.tool_layer import ToolLayer
    from unified_runner.base import RunConfig

    config = RunConfig(model="qwen3-14b", api_base="http://localhost:30000/v1")
    layer = ToolLayer(mode="host", workdir="/tmp/test")
    agent = UnifiedAgentLoop(config, layer)
    result = agent.run("Read /etc/hostname and write it to /tmp/test_output.txt")
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .base import RunConfig, TaskResult
from .openclaw_compat import build_openclaw_system_prompt
from .tool_layer import ToolLayer
from .tool_schemas import get_tools

# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = build_openclaw_system_prompt(
    workspace_dir="{workspace_dir}",
    tool_names=None,
    runtime_label="unified_runner.default",
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentTrajectory:
    """Complete record of an agent run, including the SFT-ready messages list."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    time_sec: float = 0.0
    finish_reason: str = ""  # "completed", "max_turns", "timeout", "error"
    error: str = ""
    final_response: str = ""

    def to_sft_messages(self) -> list[dict[str, Any]]:
        """Return the messages list in standard SFT format.

        Filters out internal metadata, keeping only role/content/tool_calls/
        tool_call_id fields that training frameworks expect.
        """
        sft = []
        for msg in self.messages:
            entry: dict[str, Any] = {"role": msg["role"]}
            if msg.get("content"):
                entry["content"] = msg["content"]
            if msg.get("tool_calls"):
                entry["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                entry["tool_call_id"] = msg["tool_call_id"]
            if msg.get("name"):
                entry["name"] = msg["name"]
            sft.append(entry)
        return sft


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class UnifiedAgentLoop:
    """Function-calling agent loop using OpenAI-compatible API + unified ToolLayer."""

    def __init__(
        self,
        config: RunConfig,
        tool_layer: ToolLayer,
        tools: list[str] | None = None,
        max_tool_calls_per_turn: int = 5,
        llm_max_retries: int = 3,
        llm_retry_backoff_sec: float = 2.0,
        confirm_completion: bool = True,
    ) -> None:
        self.config = config
        self.tool_layer = tool_layer
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.llm_max_retries = self._env_int("UNIFIED_LLM_MAX_RETRIES", llm_max_retries)
        self.llm_retry_backoff_sec = self._env_float("UNIFIED_LLM_RETRY_BACKOFF_SEC", llm_retry_backoff_sec)
        self.llm_retry_max_backoff_sec = self._env_float("UNIFIED_LLM_RETRY_MAX_BACKOFF_SEC", 60.0)
        self.llm_request_timeout_sec = self._env_float("UNIFIED_LLM_REQUEST_TIMEOUT_SEC", 300.0)
        self.llm_retry_http_statuses = self._env_status_set(
            "UNIFIED_LLM_RETRY_HTTP_STATUSES",
            {408, 409, 425, 429, 500, 502, 503, 504, 529},
        )
        self.confirm_completion = confirm_completion

        # Build tool schemas (filtered if requested)
        if tools is not None:
            self._tool_schemas = get_tools(include=tools)
        else:
            self._tool_schemas = get_tools()
        self._tool_names = [t["function"]["name"] for t in self._tool_schemas]

        # 2026-05-05: tri-state tools-schema mode — explicit knob replaces the
        # earlier UNIFIED_DISABLE_TOOLS_SCHEMA=1 binary flag. Three modes:
        #   - "openai_tools"  : send `tools=...` in the request, let SGLang's
        #                       chat template auto-inject the schema block at
        #                       render time. Default; matches baseline/retrieval.
        #   - "none"          : do NOT send `tools=`, do NOT inject anything
        #                       into system. For SFT data without schema (legacy).
        #   - "manual_schema" : do NOT send `tools=`, but render the schema
        #                       block via the same tokenizer/chat template as
        #                       training data and prepend to system_prompt.
        #                       For SFT data augmented via
        #                       augment_hindsight.py --inject-tools-schema.
        # Backward compat: UNIFIED_DISABLE_TOOLS_SCHEMA=1 maps to "none" so the
        # name keeps its literal meaning ("disable"); manual_schema must be
        # opted into explicitly to avoid silent injection surprises.
        self._tools_schema_mode = self._resolve_tools_schema_mode()
        self._inject_system_schema_block: str = ""
        if self._tools_schema_mode == "manual_schema":
            tok_path = os.environ.get(
                "UNIFIED_INJECT_SCHEMA_TOKENIZER_PATH",
                str(Path(os.environ.get("SKILLRL_ROOT", str(Path(__file__).resolve().parents[3]))) / "models/Qwen3.5-9B"),
            )
            try:
                self._inject_system_schema_block = self._render_tools_schema_block(
                    tok_path, self._tool_schemas
                )
                if self._inject_system_schema_block:
                    print(
                        f"[unified_agent] tools_schema_mode=manual_schema: "
                        f"prepending {len(self._inject_system_schema_block)}-char "
                        f"schema block to system (tokenizer={tok_path})",
                        flush=True,
                    )
                else:
                    raise RuntimeError(
                        "manual_schema selected but rendered block is empty — "
                        "refusing to start (set UNIFIED_TOOLS_SCHEMA_MODE=none "
                        "if no schema injection is desired)."
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"manual_schema render failed ({exc!r}); refuse to start. "
                    f"Set UNIFIED_TOOLS_SCHEMA_MODE=openai_tools or none."
                )
        else:
            print(
                f"[unified_agent] tools_schema_mode={self._tools_schema_mode}",
                flush=True,
            )

    @staticmethod
    def _resolve_tools_schema_mode() -> str:
        """Return one of openai_tools|none|manual_schema, with backward-compat
        for UNIFIED_DISABLE_TOOLS_SCHEMA=1 mapping to "none"."""
        explicit = os.environ.get("UNIFIED_TOOLS_SCHEMA_MODE", "").strip().lower()
        legacy_disable = os.environ.get("UNIFIED_DISABLE_TOOLS_SCHEMA", "").strip() == "1"
        if explicit:
            valid = {"openai_tools", "none", "manual_schema"}
            if explicit not in valid:
                raise ValueError(
                    f"UNIFIED_TOOLS_SCHEMA_MODE={explicit!r} invalid; "
                    f"expected one of {sorted(valid)}"
                )
            if legacy_disable and explicit == "openai_tools":
                # User set both flags in conflicting ways; explicit wins, warn.
                print(
                    "[unified_agent] WARN: UNIFIED_DISABLE_TOOLS_SCHEMA=1 set "
                    "but UNIFIED_TOOLS_SCHEMA_MODE=openai_tools also set; "
                    "explicit mode wins (openai_tools).",
                    flush=True,
                )
            return explicit
        if legacy_disable:
            return "none"
        return "openai_tools"

    @staticmethod
    def _render_tools_schema_block(tokenizer_path: str, tools: list[dict]) -> str:
        """Render the SGLang-equivalent tools-schema block by diffing chat
        template output. Mirrors GeneralAgent/sft_data_collection/augment_hindsight.py
        so training data and eval prompts use the same string."""
        from transformers import AutoTokenizer  # local import: keep import-time cheap

        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        placeholder = "__SYSTEM_PLACEHOLDER__"
        msgs = [
            {"role": "system", "content": placeholder},
            {"role": "user", "content": "x"},
        ]
        rendered = tok.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=False
        )
        sys_start = rendered.find("<|im_start|>system\n")
        if sys_start < 0:
            return ""
        sys_start += len("<|im_start|>system\n")
        pl_idx = rendered.find(placeholder, sys_start)
        if pl_idx < 0:
            return ""
        return rendered[sys_start:pl_idx].rstrip()

    def _build_runtime_metadata(self) -> dict[str, str]:
        """Populate `## Runtime` metadata line at eval time.

        Mirrors probe T086's pipe-separated key=value format. Falls back to
        "unknown" for fields we cannot read at runtime; the resulting prompt
        is structurally identical to a real OpenClaw deployment.
        """
        try:
            uname = os.uname()
            host = uname.nodename or "unknown"
            os_str = f"{uname.sysname} {uname.release} ({uname.machine})"
        except Exception:
            host, os_str = "unknown", "unknown"
        repo = getattr(self.tool_layer, "workdir", None) or "unknown"
        model = getattr(self.config, "model", None) or "unknown"
        return {
            "agent": "main",
            "host": host,
            "repo": repo,
            "os": os_str,
            "node": "unknown",  # we do not run a node runtime
            "model": f"sglang/{model}" if "/" not in model else model,
            "default_model": f"sglang/{model}" if "/" not in model else model,
            "shell": "bash",
            "thinking": "off",
        }

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name, "").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.environ.get(name, "").strip()
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _env_status_set(name: str, default: set[int]) -> set[int]:
        value = os.environ.get(name, "").strip()
        if not value:
            return set(default)
        statuses: set[int] = set()
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                statuses.add(int(part))
            except ValueError:
                pass
        return statuses or set(default)

    # --- public API --------------------------------------------------------

    def run(
        self,
        task_prompt: str,
        system_prompt: str | None = None,
    ) -> AgentTrajectory:
        """Run the agent loop until completion, max_turns, or timeout.

        Args:
            task_prompt: The user-facing task description.
            system_prompt: Optional custom system prompt. If None, uses default.

        Returns:
            AgentTrajectory with full messages list and metadata.
        """
        traj = AgentTrajectory()
        start = time.time()

        # Build system prompt
        if system_prompt is None:
            system_prompt = build_openclaw_system_prompt(
                workspace_dir=getattr(self.tool_layer, "workdir", "/workspace"),
                tool_names=self._tool_names,
                runtime_label="unified_runner.default",
                runtime_metadata=self._build_runtime_metadata(),
            )

        # 2026-05-05: when training data was augmented with a schema block
        # in front of system, prepend the same block here so the eval prompt
        # is byte-identical to the one used during SFT.
        if self._inject_system_schema_block and (
            self._inject_system_schema_block.strip() not in system_prompt
        ):
            system_prompt = (
                self._inject_system_schema_block + "\n\n" + system_prompt
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        # Double-confirmation state: model must produce two consecutive
        # tool-call-free responses to actually finish. Prevents premature
        # "I give up" exits and matches terminus-2's task_complete behavior.
        pending_completion = False

        # 2026-04-20 v6: early-stop on N consecutive identical assistant turns.
        # Disabled when config.early_stop_repeat_n <= 0. Tracks (content, tool_calls_sig).
        recent_sigs: list[str] = []
        early_stop_n = max(0, getattr(self.config, "early_stop_repeat_n", 0) or 0)
        # 2026-05-09: SECONDARY prefix-based loop check. The strict identity
        # check above misses confirmation loops where the model says
        # "我已经完成任务。让我再确认一下..." then makes the same tool call but
        # appends slightly different content each turn (length grows). The
        # prefix sig only compares the first N chars of content + the function
        # name + first M chars of arguments — catches the "verify once more"
        # pattern that ate the SFT model's turn budget on claw.
        recent_prefix_sigs: list[str] = []
        prefix_stop_n = int(os.environ.get("UNIFIED_PREFIX_STOP_N",
                                            str(early_stop_n)) or "0")
        prefix_content_len = int(os.environ.get("UNIFIED_PREFIX_CONTENT_LEN", "120"))
        prefix_args_len = int(os.environ.get("UNIFIED_PREFIX_ARGS_LEN", "80"))

        for turn in range(self.config.max_turns):
            elapsed = time.time() - start
            if elapsed > self.config.max_time_sec:
                traj.finish_reason = "timeout"
                traj.error = f"Timeout after {int(elapsed)}s"
                break

            traj.turns = turn + 1

            # Call the model
            response = self._chat_completion(messages)

            if "error" in response:
                traj.finish_reason = "error"
                traj.error = f"LLM API error: {response['error']}"
                break

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "")

            # Track tokens
            usage = response.get("usage", {})
            traj.total_input_tokens += usage.get("prompt_tokens", 0)
            traj.total_output_tokens += usage.get("completion_tokens", 0)

            # Extract tool calls (native or fallback XML parsing)
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                parsed = self._parse_tool_calls_from_content(
                    message.get("content", "")
                )
                if parsed:
                    tool_calls = parsed
                    message["tool_calls"] = tool_calls

            # 2026-04-20 v6: early-stop repetition detection (pre-deduplication).
            # Compute signature of this assistant turn: content + ordered tool_calls.
            # If last N sigs identical, agent is stuck in loop — break out.
            if early_stop_n > 0 or prefix_stop_n > 0:
                content_sig = (message.get("content") or "").strip()
                tc_sig = ""
                if tool_calls:
                    tc_sig = "|".join(
                        f"{tc.get('function',{}).get('name','')}:"
                        f"{tc.get('function',{}).get('arguments','')}"
                        for tc in tool_calls
                    )
                sig = f"{content_sig}##{tc_sig}"
                # Strict identity check (preserved from v6).
                if early_stop_n > 0:
                    recent_sigs.append(sig)
                    if len(recent_sigs) > early_stop_n:
                        recent_sigs.pop(0)
                    if (len(recent_sigs) >= early_stop_n
                        and all(s == recent_sigs[0] for s in recent_sigs)
                        and recent_sigs[0]):
                        traj.finish_reason = "early_stop_repetition"
                        traj.error = (
                            f"Identical assistant turn {early_stop_n}x consecutively; "
                            f"agent stuck in loop. Last content: {content_sig[:120]!r}"
                        )
                        print(f"  [early-stop] {early_stop_n}x consecutive identical turn → halt", flush=True)
                        messages.append({"role": "assistant",
                                         "content": message.get("content", ""),
                                         "tool_calls": tool_calls or []})
                        break
                # Prefix check (2026-05-09): catches "let me verify once more"
                # loops where each turn varies in length but starts the same way
                # and reissues the same tool call.
                if prefix_stop_n > 0:
                    pc = content_sig[:prefix_content_len]
                    ptc = ""
                    if tool_calls:
                        ptc = "|".join(
                            f"{tc.get('function',{}).get('name','')}:"
                            f"{(tc.get('function',{}).get('arguments','') or '')[:prefix_args_len]}"
                            for tc in tool_calls
                        )
                    psig = f"{pc}##{ptc}"
                    recent_prefix_sigs.append(psig)
                    if len(recent_prefix_sigs) > prefix_stop_n:
                        recent_prefix_sigs.pop(0)
                    if (len(recent_prefix_sigs) >= prefix_stop_n
                        and all(s == recent_prefix_sigs[0] for s in recent_prefix_sigs)
                        and recent_prefix_sigs[0]):
                        traj.finish_reason = "early_stop_prefix_repetition"
                        traj.error = (
                            f"Same content+tool-call prefix {prefix_stop_n}x consecutively; "
                            f"agent stuck in confirmation loop. Prefix: {pc[:120]!r}"
                        )
                        print(f"  [early-stop] {prefix_stop_n}x consecutive same prefix → halt", flush=True)
                        messages.append({"role": "assistant",
                                         "content": message.get("content", ""),
                                         "tool_calls": tool_calls or []})
                        break

            # If no tool calls, the model wants to end. Apply double confirmation.
            if not tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": message.get("content", ""),
                })
                if self.confirm_completion and not pending_completion:
                    pending_completion = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "Before finalizing, double-check that the task is fully complete. "
                            "If you need to verify the environment, call tools (read/ls/exec) one more time. "
                            "If you are confident the task is done, reply again without any tool calls "
                            "and your previous response will stand."
                        ),
                    })
                    continue
                traj.final_response = message.get("content", "")
                traj.finish_reason = "completed"
                break

            # Model called tools — reset the pending-completion flag in case it was set
            pending_completion = False

            # Deduplicate identical tool calls (same name + same arguments).
            # Qwen3.5 + SGLang sometimes emits the same tool call 3–5 times in
            # one assistant turn; executing them concurrently causes apt/dpkg
            # lock contention and wastes turns. Keep first occurrence only.
            seen = set()
            deduped = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                key = (fn.get("name", ""), fn.get("arguments", ""))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(tc)
            if len(deduped) != len(tool_calls):
                print(f"  [agent_loop] deduped {len(tool_calls)} → {len(deduped)} tool_calls")
            tool_calls = deduped
            message["tool_calls"] = tool_calls

            # Limit tool calls per turn
            if len(tool_calls) > self.max_tool_calls_per_turn:
                tool_calls = tool_calls[: self.max_tool_calls_per_turn]
                message["tool_calls"] = tool_calls

            # Append assistant message (with tool_calls)
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if message.get("content"):
                assistant_msg["content"] = message["content"]
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    arguments = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                # Dispatch through unified tool layer
                tool_result = self.tool_layer.dispatch(tool_name, arguments)
                result_str = json.dumps(tool_result, ensure_ascii=False, default=str)

                # Truncate overly long results
                if len(result_str) > self.config.max_output_chars:
                    half = self.config.max_output_chars // 2 - 50
                    omitted = len(result_str) - self.config.max_output_chars
                    result_str = (
                        result_str[:half]
                        + f"\n\n... [{omitted} chars truncated] ...\n\n"
                        + result_str[-half:]
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{turn}_{tool_name}"),
                    "name": tool_name,
                    "content": result_str,
                })

            # Context management: trim old tool results if context is getting large.
            # Threshold lowered 200_000 → 100_000 chars (2026-04-19) after 14 HTTP-400
            # ctx-overflow on tb2/sb: 200K chars allowed the conversation to grow past
            # 123K input_tokens (past SGLang 131K ctx - 8K completion reserve).
            # Env var escape hatch for debugging (set UNIFIED_TRIM_CHAR_THRESHOLD=200000
            # to restore pre-fix behaviour).
            total_chars = sum(
                len(json.dumps(m, default=str)) for m in messages
            )
            trim_threshold = int(os.environ.get("UNIFIED_TRIM_CHAR_THRESHOLD", "100000"))
            if total_chars > trim_threshold:
                messages = self._trim_messages(messages)

        else:
            # Loop exhausted without break
            traj.finish_reason = "max_turns"

        traj.messages = messages
        traj.time_sec = time.time() - start
        return traj

    def run_task(
        self,
        task_id: str,
        dataset: str,
        task_prompt: str,
        system_prompt: str | None = None,
    ) -> TaskResult:
        """Convenience wrapper that runs the agent and returns a TaskResult.

        Note: This does NOT evaluate correctness (resolved/score).
        Those must be set by the dataset adapter after checking the environment.
        """
        traj = self.run(task_prompt, system_prompt=system_prompt)
        return TaskResult(
            task_id=task_id,
            dataset=dataset,
            turns=traj.turns,
            time_sec=int(traj.time_sec),
            error=traj.error,
            trajectory=traj.to_sft_messages(),
            extra={
                "finish_reason": traj.finish_reason,
                "final_response": traj.final_response,
                "input_tokens": traj.total_input_tokens,
                "output_tokens": traj.total_output_tokens,
            },
        )

    # --- LLM API -----------------------------------------------------------

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call the OpenAI-compatible chat completion API with retry.

        Retries on transient endpoint failures. Defaults are intentionally broad
        for MaaS teacher collection; override with UNIFIED_LLM_* env vars.
        Does NOT retry on:
          - HTTP 4xx (other than 408/429) — those are bad requests, retry won't help
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        # 2026-04-20 v6: presence_penalty for anti-repetition on 27B no-think.
        pp = getattr(self.config, "presence_penalty", 0.0) or 0.0
        if pp:
            payload["presence_penalty"] = pp
        # 2026-04-20 v6: SGLang extra_body (e.g. chat_template_kwargs for no-think).
        extra = getattr(self.config, "extra_body", None)
        if extra:
            # Merge extra_body keys at top level for OpenAI-compat endpoints that
            # accept extra fields (SGLang does). Alternative: nest under "extra_body".
            for k, v in extra.items():
                if k not in payload:
                    payload[k] = v
        # 2026-05-05: tri-state tools-schema mode (see __init__ docstring).
        # Only "openai_tools" sends tools=, leaving SGLang's chat template to
        # auto-inject the schema block. "none" and "manual_schema" both omit
        # the request-level tools= parameter; "manual_schema" instead injects
        # the same schema block into system_prompt at run() time so the eval
        # prompt is byte-identical to the schema-injected training data. Tool
        # calls still come through because the model emits XML
        # <tool_call>...</tool_call> which _parse_tool_calls_from_content()
        # picks up regardless of how the schema reached the prompt.
        if self._tool_schemas and self._tools_schema_mode == "openai_tools":
            payload["tools"] = self._tool_schemas
            payload["tool_choice"] = "auto"

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = self.config.api_base.rstrip("/") + "/chat/completions"

        last_error: dict[str, Any] = {"error": "no attempt made"}
        for attempt in range(self.llm_max_retries):
            headers = {"Content-Type": "application/json"}
            api_key = getattr(self.config, "api_key", "") or os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.llm_request_timeout_sec) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    parsed = json.loads(body)
                    if (
                        isinstance(parsed, dict)
                        and "error" in parsed
                        and "choices" not in parsed
                        and self._looks_transient_error(json.dumps(parsed.get("error"), ensure_ascii=False))
                    ):
                        last_error = {"error": f"transient response error: {str(parsed.get('error'))[:500]}"}
                        if self._sleep_before_retry(attempt, last_error["error"]):
                            continue
                    return parsed
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                last_error = {"error": f"HTTP {e.code}: {body}"}
                body_lower = body.lower()
                if (
                    e.code == 400
                    and payload.get("max_tokens", 0) > 512
                    and "maximum context length" in body_lower
                ):
                    limit_match = re.search(r"maximum context length of (\d+)", body)
                    input_match = re.search(r"(\d+) tokens from the input messages", body)
                    if limit_match and input_match:
                        limit = int(limit_match.group(1))
                        input_tokens = int(input_match.group(1))
                        new_max_tokens = max(256, min(payload["max_tokens"] // 2, limit - input_tokens - 128))
                        if new_max_tokens < payload["max_tokens"] and new_max_tokens > 0:
                            payload["max_tokens"] = new_max_tokens
                            if new_max_tokens < 4096:
                                payload["messages"] = self._hard_trim_messages(payload.get("messages", []))
                            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                            print(
                                "[llm-context-shrink] HTTP 400 context overflow; "
                                f"retrying with max_tokens={new_max_tokens} "
                                f"(limit={limit}, input={input_tokens}, hard_trim={new_max_tokens < 4096})",
                                flush=True,
                            )
                            continue
                if (
                    e.code == 400
                    and ("context length" in body_lower or "longer than the model" in body_lower)
                ):
                    shrunk_messages = self._hard_trim_messages(payload.get("messages", []))
                    if shrunk_messages != payload.get("messages"):
                        payload["messages"] = shrunk_messages
                        payload["max_tokens"] = min(payload.get("max_tokens", self.config.max_tokens), 1024)
                        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                        print(
                            "[llm-context-hard-trim] HTTP 400 context overflow; "
                            f"retrying with hard-trimmed history and max_tokens={payload['max_tokens']}",
                            flush=True,
                        )
                        continue
                if e.code in self.llm_retry_http_statuses or e.code >= 500 or self._looks_transient_error(body):
                    if self._sleep_before_retry(attempt, last_error["error"]):
                        continue
                # Non-transient: don't retry
                return last_error
            except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as e:
                last_error = {"error": f"{type(e).__name__}: {e}"}
                if self._sleep_before_retry(attempt, last_error["error"]):
                    continue
                return last_error
            except json.JSONDecodeError as e:
                last_error = {"error": f"JSONDecodeError: {e}"}
                if self._sleep_before_retry(attempt, last_error["error"]):
                    continue
                return last_error
            except Exception as e:
                last_error = {"error": f"{type(e).__name__}: {e}"}
                if self._looks_transient_error(str(e)) and self._sleep_before_retry(attempt, last_error["error"]):
                    continue
                return last_error
        return last_error

    def _sleep_before_retry(self, attempt: int, error: str) -> bool:
        if attempt >= self.llm_max_retries - 1:
            return False
        delay = min(
            self.llm_retry_max_backoff_sec,
            self.llm_retry_backoff_sec * (2 ** attempt),
        )
        delay += random.uniform(0, min(1.0, delay * 0.2))
        print(
            f"[llm-retry] attempt {attempt + 1}/{self.llm_max_retries} failed: "
            f"{error[:220]} ; retrying in {delay:.1f}s",
            flush=True,
        )
        time.sleep(delay)
        return True

    @staticmethod
    def _looks_transient_error(text: str) -> bool:
        value = (text or "").lower()
        transient_markers = [
            "timeout", "timed out", "temporar", "try again", "too many requests",
            "rate limit", "throttle", "overload", "busy", "unavailable",
            "upstream", "gateway", "bad gateway", "connection reset",
            "connection aborted", "remote end closed", "internal error",
            "service error", "server error",
        ]
        fatal_markers = [
            "invalid api key", "unauthorized", "permission denied",
            "context length", "maximum context", "invalid_request_error",
            "tool schema", "invalid tool", "model not found",
        ]
        return any(marker in value for marker in transient_markers) and not any(
            marker in value for marker in fatal_markers
        )

    # --- Qwen3 XML fallback ------------------------------------------------

    @staticmethod
    def _parse_tool_calls_from_content(content: str) -> list[dict]:
        """Parse tool calls from Qwen model content when SGLang doesn't parse them.

        Handles two formats:
        1. Qwen3 JSON: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        2. Qwen3.5/2.5 XML: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        """
        if not content:
            return []
        tool_calls = []

        # Format 1: Qwen3 JSON style
        json_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        for i, match in enumerate(json_matches[:5]):
            try:
                parsed = json.loads(match)
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", {})
                tool_calls.append({
                    "id": f"call_{i}_{int(time.time())}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    },
                })
            except json.JSONDecodeError:
                continue

        if tool_calls:
            return tool_calls

        # Format 2: Qwen3.5/2.5 XML style
        # <tool_call><function=read_file><parameter=path>/etc/hostname</parameter></function></tool_call>
        xml_pattern = r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>"
        xml_matches = re.findall(xml_pattern, content, re.DOTALL)
        for i, (fn_name, params_block) in enumerate(xml_matches[:5]):
            param_pattern = r"<parameter=(\w+)>\s*(.*?)\s*</parameter>"
            params = re.findall(param_pattern, params_block, re.DOTALL)
            arguments = {}
            for key, value in params:
                # Try to parse as JSON for complex values
                value = value.strip()
                try:
                    arguments[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = value
            tool_calls.append({
                "id": f"call_{i}_{int(time.time())}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            })

        return tool_calls

    # --- Context trimming --------------------------------------------------

    @staticmethod
    def _trim_messages(messages: list[dict]) -> list[dict]:
        """Trim old tool results to keep context under control.

        Keeps system + first user message + last N messages intact.
        Middle tool messages get their content truncated.

        Tunable via env vars (2026-04-19 fix for ctx-overflow HTTP 400):
          - UNIFIED_TRIM_KEEP_LAST  (default 6, was 12):
              Number of most-recent messages to keep intact.
          - UNIFIED_TRIM_MIDDLE_CHARS (default 1500, was 200):
              Per tool-result char cap in the middle window. 200 was so aggressive
              that models often re-explored by re-reading files; 1500 preserves
              the signal (first file content, error summary) without reinflating
              context.
        """
        keep_last = max(2, int(os.environ.get("UNIFIED_TRIM_KEEP_LAST", "6")))
        middle_cap = max(100, int(os.environ.get("UNIFIED_TRIM_MIDDLE_CHARS", "1500")))
        # Always keep system + first_user (2) + keep_last; need at least keep_last+2+1 to do anything useful
        if len(messages) <= keep_last + 2:
            return messages
        # Keep system + first user
        trimmed = [messages[0], messages[1]]
        # Compress middle
        for msg in messages[2:-keep_last]:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > middle_cap:
                    content = content[:middle_cap] + f"\n[TRIMMED: {len(msg.get('content', '')) - middle_cap} chars omitted]"
                trimmed.append({**msg, "content": content})
            else:
                trimmed.append(msg)
        # Keep last N intact
        trimmed.extend(messages[-keep_last:])
        return trimmed

    @staticmethod
    def _hard_trim_messages(messages: list[dict]) -> list[dict]:
        """Aggressive overflow fallback after the server rejects context length.

        Normal trimming preserves recent messages intact for quality. This path
        runs only after a hard HTTP 400 from the model server, so preserving a
        valid trajectory attempt is more important than keeping full history.
        """
        if len(messages) <= 6:
            return messages

        def cap_message(msg: dict, cap: int) -> dict:
            content = msg.get("content")
            if isinstance(content, str) and len(content) > cap:
                omitted = len(content) - cap
                return {**msg, "content": content[:cap] + f"\n[HARD_TRIMMED: {omitted} chars omitted]"}
            return msg

        trimmed: list[dict] = [messages[0], messages[1]]
        for msg in messages[2:-4]:
            if msg.get("role") == "tool":
                trimmed.append(cap_message(msg, 800))
            elif msg.get("role") == "assistant":
                trimmed.append(cap_message(msg, 1200))
            else:
                trimmed.append(cap_message(msg, 2000))
        for msg in messages[-4:]:
            if msg.get("role") == "tool":
                trimmed.append(cap_message(msg, 2000))
            elif msg.get("role") == "assistant":
                trimmed.append(cap_message(msg, 2500))
            else:
                trimmed.append(cap_message(msg, 4000))
        return trimmed


# ---------------------------------------------------------------------------
# CLI: quick test
# ---------------------------------------------------------------------------

def main():
    """Quick test: read /etc/hostname and write it to /tmp/test_output.txt."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test the unified agent loop")
    parser.add_argument(
        "--task",
        default="Read /etc/hostname and write its content to /tmp/unified_test_output.txt",
        help="Task prompt to execute",
    )
    parser.add_argument("--model", default="qwen3-14b", help="Model name")
    parser.add_argument(
        "--api-base",
        default="http://localhost:30000/v1",
        help="API base URL",
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--workdir", default="/tmp")
    parser.add_argument(
        "--tools",
        nargs="*",
        default=None,
        help="Tool subset to use (default: all 9)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full messages")
    args = parser.parse_args()

    config = RunConfig(
        model=args.model,
        api_base=args.api_base,
        max_turns=args.max_turns,
        max_time_sec=300,
        workdir=args.workdir,
    )
    layer = ToolLayer(mode="host", workdir=args.workdir)
    agent = UnifiedAgentLoop(config, layer, tools=args.tools)

    print(f"Model: {config.model}")
    print(f"API: {config.api_base}")
    print(f"Tools: {agent._tool_names}")
    print(f"Task: {args.task}")
    print("=" * 60)

    traj = agent.run(args.task)

    print(f"\n{'=' * 60}")
    print(f"Finished: {traj.finish_reason}")
    print(f"Turns: {traj.turns}")
    print(f"Time: {traj.time_sec:.1f}s")
    print(f"Tokens: {traj.total_input_tokens} in / {traj.total_output_tokens} out")

    if traj.error:
        print(f"Error: {traj.error}")

    if traj.final_response:
        print(f"\nFinal response:\n{traj.final_response[:500]}")

    # Print tool call summary
    tool_calls_made = []
    for msg in traj.messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                tool_calls_made.append(fn.get("name", "?"))
    print(f"\nTool calls: {tool_calls_made}")

    if args.verbose:
        print(f"\n{'=' * 60}")
        print("Full messages:")
        for i, msg in enumerate(traj.messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            tc = msg.get("tool_calls")
            print(f"\n--- [{i}] {role} ---")
            if content:
                print(content[:300])
            if tc:
                for t in tc:
                    fn = t.get("function", {})
                    print(f"  → {fn.get('name')}({fn.get('arguments', '')[:100]})")

    # Output SFT-format trajectory
    sft_msgs = traj.to_sft_messages()
    print(f"\nSFT trajectory: {len(sft_msgs)} messages")

    return 0 if traj.finish_reason == "completed" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""Augment SFT messages with hindsight skill-decision reasoning.

For each sample in `sft_messages.jsonl`, ask a teacher (default Qwen3.5-27B)
to write a short reasoning *as if it had only seen the task*, justifying the
decision the trajectory actually took (read skill / not read skill). The
reasoning is prepended to the first assistant message inside a
`<skill_reasoning>...</skill_reasoning>` block so the trained model learns to
emit a skill-decision reasoning step before any tool call.

Why hindsight: the trajectory's outcome (success + used_skill) is known, so
the reasoning conclusion is guaranteed-correct. This injects a
condition→decision signal that pure imitation of the action sequence cannot
provide.

Usage (from project root):

  python3 GeneralAgent/sft_data_collection/augment_hindsight.py \
    --input GeneralAgent/sft_training/datasets/20260503_sft_campaign_1015/sft_messages.jsonl \
    --output GeneralAgent/sft_training/datasets/20260503_sft_campaign_1015_hindsight/sft_messages.jsonl \
    --api-base http://127.0.0.1:30002/v1 \
    --model qwen3.5-27b-hindsight \
    --workers 8

The output file is append-only and idempotent across runs: each line is
keyed by `metadata.trial_id`, and existing lines are kept. Re-running the
script picks up where it left off.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """You are an agent that solves tasks under the OpenClaw tool interface. Below is the system context (including the list of retrieved skills) and the user task you received.

[SYSTEM_CONTEXT]
{system}
[/SYSTEM_CONTEXT]

[USER_TASK]
{user_task}
[/USER_TASK]

[GROUND_TRUTH]
The action you actually took next was: **{decision_en}**, and you ultimately solved the task.
[/GROUND_TRUTH]

Pretend you have just received the task and have not started yet. Write a short reasoning passage (no more than 500 tokens) that walks through your decision of whether to read the retrieved skill files before solving the task. Your conclusion **must** match the GROUND_TRUTH (i.e. you decide to {do_or_dont_en} read the skill files).

The reasoning must contain these three things:
1. What the core requirement of the task is (1-2 sentences).
2. A quick glance at the retrieved skills' names / descriptions, judging their relevance to this task.
3. The decision and justification — explain why you {do_or_dont_en} read the skill files, drawing on points 1 and 2.

Language requirement: write the reasoning in the **same primary language as the USER_TASK**. If the user task is mostly Chinese, reason in Chinese. If the user task is mostly English, reason in English. If the user task is bilingual, mirror its dominant language. Do not switch languages mid-reasoning. This is important: train-time reasoning language must match deployment-time user language so the SFT model learns the right surface form.

Output only the reasoning text. No wrapping tags. No "Reasoning:" prefix. No JSON. No quotes.
"""


def repo_path(value: str | os.PathLike) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def first_assistant_index(messages: list[dict]) -> int | None:
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return i
    return None


def render_tools_schema_block(tokenizer_path: str, tools: list[dict]) -> str:
    """Compute the exact tools-schema block that the Qwen3.5 chat template
    injects when an OpenAI-style `tools=[...]` parameter is sent.

    Qwen3.5 chat_template.jinja places the schema **before** the user's system
    content inside the system turn, like:

        <|im_start|>system
        # Tools
        ... <tools>...</tools> ...
        <IMPORTANT>...</IMPORTANT>

        {user_system_content}<|im_end|>

    So with-tools rendering = `<schema_block>\n\n{user_system_content}` while
    without-tools rendering = `{user_system_content}`. We extract the schema
    block by taking everything in the with-tools system turn that appears
    *before* a placeholder we put as user_system_content.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    placeholder = "__SYSTEM_PLACEHOLDER__"
    msgs = [
        {"role": "system", "content": placeholder},
        {"role": "user", "content": "x"},
    ]
    rendered_with = tok.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=False
    )
    sys_start = rendered_with.find("<|im_start|>system\n")
    if sys_start < 0:
        return ""
    sys_start += len("<|im_start|>system\n")
    pl_idx = rendered_with.find(placeholder, sys_start)
    if pl_idx < 0:
        return ""
    block = rendered_with[sys_start:pl_idx]
    # The template appends "\n\n" before the user content; strip trailing
    # whitespace so callers can join with their own separator.
    return block.rstrip()


def build_prompt(sample: dict) -> str | None:
    msgs = sample.get("messages") or []
    if len(msgs) < 2:
        return None
    system = next((m for m in msgs if m.get("role") == "system"), None)
    user = next((m for m in msgs if m.get("role") == "user"), None)
    if not system or not user:
        return None
    used_skill = bool(sample.get("metadata", {}).get("used_skill"))
    decision_en = (
        "decided to read the retrieved skill files before starting the task"
        if used_skill
        else "decided to NOT read the retrieved skill files and rely on your own ability to start solving directly"
    )
    do_or_dont_en = "DO" if used_skill else "do NOT"
    return PROMPT_TEMPLATE.format(
        system=(system.get("content") or "")[:8000],
        user_task=(user.get("content") or "")[:4000],
        decision_en=decision_en,
        do_or_dont_en=do_or_dont_en,
    )


def call_teacher(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_sec: int,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read()
    obj = json.loads(body.decode("utf-8"))
    return obj["choices"][0]["message"]["content"] or ""


def truncate_token_budget(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def augment_one(
    sample: dict,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout_sec: int,
    retries: int,
    char_budget: int,
    tools_schema_block: str = "",
) -> tuple[dict | None, str | None]:
    prompt = build_prompt(sample)
    if prompt is None:
        return None, "no_prompt"
    last_err = None
    for attempt in range(retries):
        try:
            reasoning = call_teacher(
                api_base, api_key, model, prompt, max_tokens, timeout_sec
            ).strip()
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = repr(exc)
            time.sleep(min(2 ** attempt, 30))
    else:
        return None, f"call_failed: {last_err}"

    if not reasoning:
        return None, "empty_reasoning"

    reasoning = truncate_token_budget(reasoning, char_budget)
    msgs = sample.get("messages") or []
    idx = first_assistant_index(msgs)
    if idx is None:
        return None, "no_assistant"

    # Optionally prepend the SGLang-equivalent tools-schema block to the
    # system message so training data == inference prompt. Pair with
    # UNIFIED_TOOLS_SCHEMA_MODE=manual_schema at eval time so agent_loop
    # injects the same block and skips request-level tools=. (Sending
    # tools= AND having the block in system → SGLang double-injects.)
    if tools_schema_block:
        sys_idx = next(
            (i for i, m in enumerate(msgs) if m.get("role") == "system"), None
        )
        if sys_idx is not None:
            sys_msg = dict(msgs[sys_idx])
            sys_content = (sys_msg.get("content") or "").lstrip()
            if tools_schema_block.strip() not in sys_content:
                # Qwen3.5 chat template places schema BEFORE user system
                # content. Mirror that order so the rendered prompt is
                # identical when inference runs with no tools= parameter.
                sys_msg["content"] = tools_schema_block + "\n\n" + sys_content
                msgs = list(msgs)
                msgs[sys_idx] = sys_msg

    original = msgs[idx].get("content") or ""
    msgs[idx] = dict(msgs[idx])
    # Byte-align with the qwen3.5 chat-template generation prompt:
    #   - enable_thinking=False jinja prepends `<think>\n\n</think>\n\n`
    #   - default (enable_thinking=True) jinja prepends `<think>\n` and the
    #     model continues with `\n</think>\n\n` then the body below
    # Both prefixes share the same byte sequence at the cut points used by
    # SGLang, so this format trains a single model that works under either
    # deployment mode. The empty think block also discourages the base 9B's
    # natural tendency to emit a long CoT inside <think>.
    msgs[idx]["content"] = (
        f"<think>\n\n</think>\n\n"
        f"<skill_reasoning>\n{reasoning}\n</skill_reasoning>\n\n"
        f"{original}"
    )
    sample = dict(sample)
    sample["messages"] = msgs
    sample.setdefault("metadata", {})["hindsight_reasoning"] = reasoning
    sample["metadata"]["hindsight_model"] = model
    if tools_schema_block:
        sample["metadata"]["tools_schema_injected"] = True
    return sample, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="Path to source sft_messages.jsonl")
    parser.add_argument("--output", required=True, help="Path to write augmented jsonl")
    parser.add_argument("--api-base", default="http://127.0.0.1:30002/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "sk-local-anything"))
    parser.add_argument("--model", default="qwen3.5-27b-hindsight")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=550)
    parser.add_argument("--char-budget", type=int, default=2400, help="Hard cap on reasoning chars after generation")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N (0 = all)")
    parser.add_argument(
        "--min-turns",
        type=int,
        default=0,
        help="Drop samples whose metadata.turns is below this (P0 filter against premature-stop pattern). 0 disables.",
    )
    parser.add_argument(
        "--inject-tools-schema",
        action="store_true",
        help="Prepend the SGLang-equivalent OpenAI-tools schema block to each sample's "
             "system content so training prompt matches the inference prompt that SGLang "
             "automatically renders when tools=[...] is sent. Pair with "
             "UNIFIED_TOOLS_SCHEMA_MODE=manual_schema at inference (which makes "
             "agent_loop inject the same block and stop sending tools=) to avoid double-injection.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.environ.get("SKILLRL_ROOT", "/path/to/skillRL") + "/models/Qwen3.5-9B",
        help="Tokenizer/chat-template source for rendering the tools schema block.",
    )
    args = parser.parse_args()

    in_path = repo_path(args.input)
    out_path = repo_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(in_path)
    raw_n = len(samples)
    if args.min_turns > 0:
        samples = [
            s
            for s in samples
            if (s.get("metadata") or {}).get("turns") is not None
            and int(s["metadata"]["turns"]) >= args.min_turns
        ]
        print(f"min_turns={args.min_turns}: filtered {raw_n} -> {len(samples)}", flush=True)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print(f"no samples loaded from {in_path}", file=sys.stderr)
        return 2

    existing = read_jsonl(out_path)
    done_ids = {
        str(r.get("metadata", {}).get("trial_id", ""))
        for r in existing
        if r.get("metadata", {}).get("trial_id")
    }
    pending: list[dict] = []
    for s in samples:
        tid = str(s.get("metadata", {}).get("trial_id", ""))
        if tid and tid not in done_ids:
            pending.append(s)

    print(
        f"input={in_path} output={out_path} total={len(samples)} "
        f"resume_done={len(done_ids)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    tools_schema_block = ""
    if args.inject_tools_schema:
        # Lazy-import unified_runner.tool_schemas as a package member so its
        # relative imports (`from .openclaw_probe_full_tools import ...`) work.
        sys.path.insert(0, str(repo_path("GeneralAgent/eval_scripts")))
        from unified_runner.tool_schemas import get_tools  # type: ignore  # noqa: E402

        tools = get_tools()
        tools_schema_block = render_tools_schema_block(args.tokenizer_path, tools)
        print(
            f"inject_tools_schema=ON tools={len(tools)} block_chars={len(tools_schema_block)}",
            flush=True,
        )
        print(f"schema block preview: {tools_schema_block[:300]!r}", flush=True)

    write_lock = threading.Lock()
    out_fh = out_path.open("a", encoding="utf-8")
    processed = 0
    failed = 0
    started = time.time()

    def task(s: dict) -> tuple[str, dict | None, str | None]:
        tid = str(s.get("metadata", {}).get("trial_id", ""))
        out_sample, err = augment_one(
            s,
            args.api_base,
            args.api_key,
            args.model,
            args.max_tokens,
            args.timeout_sec,
            args.retries,
            args.char_budget,
            tools_schema_block,
        )
        return tid, out_sample, err

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(task, s) for s in pending]
            for fut in as_completed(futures):
                tid, out_sample, err = fut.result()
                if err:
                    failed += 1
                    print(f"[FAIL] trial_id={tid} err={err}", flush=True)
                    continue
                if out_sample is None:
                    failed += 1
                    continue
                line = json.dumps(out_sample, ensure_ascii=False)
                with write_lock:
                    out_fh.write(line + "\n")
                    out_fh.flush()
                processed += 1
                if processed % 25 == 0:
                    rate = processed / max(time.time() - started, 1)
                    print(
                        f"[OK] processed={processed}/{len(pending)} failed={failed} rate={rate:.2f}/s",
                        flush=True,
                    )
    finally:
        out_fh.close()

    print(f"done: processed={processed} failed={failed} elapsed={time.time()-started:.1f}s", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

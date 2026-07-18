#!/usr/bin/env python3
"""Compare SFT training prompt prefixes with SGLang-rendered inference prompts."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_DIR = PROJECT_ROOT / (
    "GeneralAgent/sft_training/merged_models/"
    "qwen35_9b_sft_campaign_20260507_2042_all_phase12_openclaw_full_4gpu_82k_5epoch_r32_liger"
)
DEFAULT_TRAIN_JSON = PROJECT_ROOT / (
    "GeneralAgent/sft_training/llamafactory_data/"
    "20260507_sft_campaign_2042_all_phase12_openclaw_full/"
    "agent_sft_campaign_20260507_2042_all_phase12_openclaw_full.json"
)
DEFAULT_EVAL_TRAJECTORY = PROJECT_ROOT / (
    "experiments/20260507/20260507_quick30_sft2042_openclaw_full_retrieval/"
    "results/claw/"
    "20260507_quick30_sft2042_openclaw_full_retrieval_sft_9b_eval_retrieval_claw_T016_kb_search_t00_retrieval/"
    "trajectories/T016_kb_search.json"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / (
    "experiments/20260507/20260507_quick30_sft2042_openclaw_full_retrieval/reports/"
    "prompt_prefix_compare_T016"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_generation_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            break
        prefix.append({"role": msg["role"], "content": msg.get("content", "")})
    return prefix


def render_llamafactory_qwen35_nothink(messages: list[dict[str, Any]]) -> str:
    """Render the first assistant prefix used by LLaMA-Factory qwen3_5_nothink."""
    chunks: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            chunks.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            chunks.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
        elif role == "tool":
            chunks.append(
                f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            raise ValueError(f"unexpected role before first generation: {role}")
    return "".join(chunks)


def load_tokenizer(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        template_path = model_dir / "chat_template.jinja"
        tok.chat_template = template_path.read_text(encoding="utf-8")
    return tok


def render_sglang_chat_template(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool | None,
) -> tuple[list[int], str]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": False,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    ids = tokenizer.apply_chat_template(messages, **kwargs)
    return ids, tokenizer.decode(ids, skip_special_tokens=False)


def longest_common_prefix(a: str, b: str) -> int:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def line_col(text: str, idx: int) -> tuple[int, int]:
    line = text.count("\n", 0, idx) + 1
    last_newline = text.rfind("\n", 0, idx)
    col = idx + 1 if last_newline < 0 else idx - last_newline
    return line, col


def char_diff_summary(a: str, b: str, name_a: str, name_b: str, context: int = 220) -> dict[str, Any]:
    lcp = longest_common_prefix(a, b)
    start = max(0, lcp - context)
    end_a = min(len(a), lcp + context)
    end_b = min(len(b), lcp + context)
    line_a, col_a = line_col(a, lcp)
    line_b, col_b = line_col(b, lcp)
    diff = "\n".join(
        difflib.unified_diff(
            a[start:end_a].splitlines(),
            b[start:end_b].splitlines(),
            fromfile=name_a,
            tofile=name_b,
            lineterm="",
        )
    )
    return {
        "equal": a == b,
        "len_a": len(a),
        "len_b": len(b),
        "lcp_chars": lcp,
        "first_mismatch": {
            name_a: {"line": line_a, "col": col_a, "char": repr(a[lcp : lcp + 1])},
            name_b: {"line": line_b, "col": col_b, "char": repr(b[lcp : lcp + 1])},
        },
        "diff": diff,
    }


def system_tool_schema_block(system: str) -> str:
    marker = "</IMPORTANT>"
    end = system.find(marker)
    return system[: end + len(marker)] if end >= 0 else system


def server_prefill_check(
    server: str,
    input_ids: list[int],
    *,
    local_prompt: str,
    tail_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    start_len = max(0, len(input_ids) - tail_tokens)
    body = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": start_len,
        "return_text_in_logprobs": True,
    }
    resp = requests.post(server.rstrip("/") + "/generate", json=body, timeout=timeout)
    out: dict[str, Any] = {
        "url": server.rstrip("/") + "/generate",
        "status_code": resp.status_code,
        "logprob_start_len": start_len,
    }
    resp.raise_for_status()
    obj = resp.json()
    meta = obj.get("meta_info", {})
    pieces = [x[2] for x in meta.get("input_token_logprobs", []) if len(x) >= 3 and x[2] is not None]
    server_tail = "".join(pieces)
    out.update(
        {
            "prompt_tokens_reported": meta.get("prompt_tokens"),
            "local_prompt_tokens": len(input_ids),
            "prompt_token_count_matches": meta.get("prompt_tokens") == len(input_ids),
            "server_tail_text": server_tail,
            "server_tail_matches_local_suffix": local_prompt.endswith(server_tail),
            "output_text": obj.get("text", ""),
            "output_ids": obj.get("output_ids", []),
            "output_token_logprobs": meta.get("output_token_logprobs", []),
            "finish_reason": meta.get("finish_reason"),
            "e2e_latency": meta.get("e2e_latency"),
            "prefill_launch_latency": meta.get("prefill_launch_latency"),
        }
    )
    return out


def server_generate_text(
    server: str,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    body = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
    }
    resp = requests.post(server.rstrip("/") + "/generate", json=body, timeout=timeout)
    out: dict[str, Any] = {
        "url": server.rstrip("/") + "/generate",
        "status_code": resp.status_code,
        "max_new_tokens": max_new_tokens,
    }
    resp.raise_for_status()
    obj = resp.json()
    meta = obj.get("meta_info", {})
    out.update(
        {
            "text": obj.get("text", ""),
            "output_ids": obj.get("output_ids", []),
            "finish_reason": meta.get("finish_reason"),
            "prompt_tokens_reported": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "e2e_latency": meta.get("e2e_latency"),
        }
    )
    return out


def find_training_record(records: list[dict[str, Any]], bench: str, mode: str) -> tuple[int, dict[str, Any]]:
    for idx, rec in enumerate(records):
        md = rec.get("metadata", {})
        if md.get("bench") == bench and md.get("mode") == mode:
            return idx, rec
    raise ValueError(f"no training record for bench={bench!r} mode={mode!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    ap.add_argument("--eval-trajectory", type=Path, default=DEFAULT_EVAL_TRAJECTORY)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--server", default="http://127.0.0.1:30001")
    ap.add_argument("--tail-tokens", type=int, default=96)
    ap.add_argument("--ab-max-new-tokens", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_dir)

    eval_traj = load_json(args.eval_trajectory)
    eval_messages = first_generation_messages(eval_traj["messages"])
    train_records = load_json(args.train_json)
    train_idx, train_record = find_training_record(train_records, bench="claw", mode="student_use_skill")
    train_messages = first_generation_messages(train_record["messages"])

    train_template_on_eval = render_llamafactory_qwen35_nothink(eval_messages)
    train_prompt_actual = render_llamafactory_qwen35_nothink(train_messages)
    infer_ids_false, infer_prompt_false = render_sglang_chat_template(
        tokenizer, eval_messages, enable_thinking=False
    )
    infer_ids_default, infer_prompt_default = render_sglang_chat_template(
        tokenizer, eval_messages, enable_thinking=None
    )
    train_ids_on_eval = tokenizer.encode(train_template_on_eval, add_special_tokens=False)

    server_check = server_prefill_check(
        args.server,
        infer_ids_false,
        local_prompt=infer_prompt_false,
        tail_tokens=args.tail_tokens,
        timeout=args.timeout,
    )
    ab_generation = {
        "llamafactory_train_template_on_eval": server_generate_text(
            args.server,
            train_ids_on_eval,
            max_new_tokens=args.ab_max_new_tokens,
            timeout=args.timeout,
        ),
        "sglang_enable_thinking_false_on_eval": server_generate_text(
            args.server,
            infer_ids_false,
            max_new_tokens=args.ab_max_new_tokens,
            timeout=args.timeout,
        ),
    }

    eval_system = eval_messages[0]["content"]
    train_system = train_messages[0]["content"]
    schema_eval = system_tool_schema_block(eval_system)
    schema_train = system_tool_schema_block(train_system)

    comparisons = {
        "same_eval_messages_train_template_vs_sglang_enable_false": char_diff_summary(
            train_template_on_eval,
            infer_prompt_false,
            "llamafactory_qwen3_5_nothink_on_eval",
            "sglang_chat_template_enable_thinking_false_on_eval",
        ),
        "same_eval_messages_sglang_enable_false_vs_default": char_diff_summary(
            infer_prompt_false,
            infer_prompt_default,
            "sglang_enable_thinking_false",
            "sglang_default",
        ),
        "tool_schema_block_train_vs_eval": char_diff_summary(
            schema_train,
            schema_eval,
            "train_tool_schema_block",
            "eval_tool_schema_block",
        ),
    }

    prompts_dir = args.report_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    prompt_files = {
        "eval_prompt_llamafactory_train_template.txt": train_template_on_eval,
        "eval_prompt_sglang_enable_thinking_false.txt": infer_prompt_false,
        "eval_prompt_sglang_default.txt": infer_prompt_default,
        "actual_train_prompt_llamafactory.txt": train_prompt_actual,
    }
    for name, text in prompt_files.items():
        (prompts_dir / name).write_text(text, encoding="utf-8")

    summary = {
        "eval": {
            "trajectory": str(args.eval_trajectory.relative_to(PROJECT_ROOT)),
            "task_id": eval_traj.get("task_id"),
            "dataset": eval_traj.get("dataset"),
            "score": eval_traj.get("score"),
            "resolved": eval_traj.get("resolved"),
            "first_assistant_prefix": eval_traj["messages"][2]["content"][:80],
        },
        "train": {
            "json": str(args.train_json.relative_to(PROJECT_ROOT)),
            "record_index": train_idx,
            "metadata": train_record.get("metadata", {}),
            "first_assistant_prefix": train_record["messages"][2]["content"][:80],
        },
        "token_counts": {
            "eval_prompt_llamafactory_train_template": len(train_ids_on_eval),
            "eval_prompt_sglang_enable_thinking_false": len(infer_ids_false),
            "eval_prompt_sglang_default": len(infer_ids_default),
        },
        "prompt_suffixes": {
            "llamafactory_train_template": train_template_on_eval[-220:],
            "sglang_enable_thinking_false": infer_prompt_false[-220:],
            "sglang_default": infer_prompt_default[-220:],
        },
        "comparisons": comparisons,
        "server_prefill_check": server_check,
        "ab_generation": ab_generation,
        "prompt_files": {k: str((prompts_dir / k).relative_to(PROJECT_ROOT)) for k in prompt_files},
    }

    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Prompt Prefix Comparison: SFT Training vs SGLang",
        "",
        f"- Eval trajectory: `{summary['eval']['trajectory']}`",
        f"- Training record: index `{train_idx}`, trial `{train_record.get('metadata', {}).get('trial_id')}`",
        f"- Server prefill: `{server_check['url']}`, status `{server_check['status_code']}`",
        "",
        "## Key Result",
        "",
        "For the same eval messages, the LLaMA-Factory training template ends the prompt as:",
        "",
        "```text",
        summary["prompt_suffixes"]["llamafactory_train_template"],
        "```",
        "",
        "SGLang with `chat_template_kwargs.enable_thinking=false` ends the prompt as:",
        "",
        "```text",
        summary["prompt_suffixes"]["sglang_enable_thinking_false"],
        "```",
        "",
        "SGLang default ends the prompt as:",
        "",
        "```text",
        summary["prompt_suffixes"]["sglang_default"],
        "```",
        "",
        "## Token Counts",
        "",
        f"- Training template on eval messages: `{len(train_ids_on_eval)}`",
        f"- SGLang enable_thinking=false: `{len(infer_ids_false)}`",
        f"- SGLang default: `{len(infer_ids_default)}`",
        "",
        "## Character Diff",
        "",
        "### Training Template vs SGLang enable_thinking=false",
        "",
        f"- equal: `{comparisons['same_eval_messages_train_template_vs_sglang_enable_false']['equal']}`",
        f"- common prefix chars: `{comparisons['same_eval_messages_train_template_vs_sglang_enable_false']['lcp_chars']}`",
        "",
        "```diff",
        comparisons["same_eval_messages_train_template_vs_sglang_enable_false"]["diff"],
        "```",
        "",
        "### Tool Schema Block: Actual Training Sample vs Eval Sample",
        "",
        f"- equal: `{comparisons['tool_schema_block_train_vs_eval']['equal']}`",
        f"- common prefix chars: `{comparisons['tool_schema_block_train_vs_eval']['lcp_chars']}`",
        "",
        "```diff",
        comparisons["tool_schema_block_train_vs_eval"]["diff"],
        "```",
        "",
        "## Server Prefill Check",
        "",
        f"- prompt tokens reported by SGLang: `{server_check['prompt_tokens_reported']}`",
        f"- local rendered prompt tokens: `{server_check['local_prompt_tokens']}`",
        f"- token count matches: `{server_check['prompt_token_count_matches']}`",
        f"- captured server tail matches local suffix: `{server_check['server_tail_matches_local_suffix']}`",
        f"- one-token output: `{server_check['output_text']}` ids `{server_check['output_ids']}`",
        "",
        "Captured server prompt tail:",
        "",
        "```text",
        server_check["server_tail_text"],
        "```",
        "",
        "## A/B Greedy Generation",
        "",
        "Same eval messages, only the prompt prefix differs.",
        "",
        "Training-template prompt output:",
        "",
        "```text",
        ab_generation["llamafactory_train_template_on_eval"]["text"],
        "```",
        "",
        "SGLang enable_thinking=false prompt output:",
        "",
        "```text",
        ab_generation["sglang_enable_thinking_false_on_eval"]["text"],
        "```",
    ]
    (args.report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "report": str((args.report_dir / "summary.md").relative_to(PROJECT_ROOT)),
        "summary_json": str((args.report_dir / "summary.json").relative_to(PROJECT_ROOT)),
        "server_prompt_token_count_matches": server_check["prompt_token_count_matches"],
        "server_tail_matches_local_suffix": server_check["server_tail_matches_local_suffix"],
        "train_vs_sglang_equal": comparisons["same_eval_messages_train_template_vs_sglang_enable_false"]["equal"],
        "ab_train_template_output": ab_generation["llamafactory_train_template_on_eval"]["text"],
        "ab_sglang_enable_false_output": ab_generation["sglang_enable_thinking_false_on_eval"]["text"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sample oracle-BC trajectories and materialize hard-span selected tokens.

The script is intentionally read-only with respect to RL runs: it scans saved
rollout JSONL files, calls the same hard-span selector used by training, and
writes a standalone JSON audit artifact.  It is meant for before/after mask
reviews, for example re-running the same manifest after a future v4 selector.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RELAX_ROOT = ROOT / "Relax"
DEFAULT_TOKENIZER_PATH = (
    ROOT
    / "GeneralAgent/sft_training/merged_models"
    / "qwen35_9b_sft_campaign_20260512_clean_plus_claw_thinkwrap_4gpu_49k_5epoch_r32_liger"
)
DEFAULT_INPUT_GLOB = (
    "experiments/rl/runs/*/segments/"
    "*oraclepromptbc*/rollout_result/train/*.jsonl"
)
MASK_GAP = "\n<<<MASK_GAP>>>\n"
SKILL_LEAK_RE = re.compile(
    r"(skill_reasoning|retrieved skill|preloaded(?: oracle| top1| skill)?|oracle skill|"
    r"available_skills|SKILL\.md|\.claude/skills|\bskills?\b)",
    re.IGNORECASE,
)
STRONG_SKILL_LEAK_RE = re.compile(
    r"(skill_reasoning|retrieved skill|preloaded(?: oracle| top1| skill)?|oracle skill|"
    r"available_skills|SKILL\.md|\.claude/skills|"
    r"read(?:ing)? (?:the |this |a )?skill|"
    r"(?:the|this|provided|pre-existing) skill (?:says|contains|provides|mentions|indicates)|"
    r"/root/\.cache/retrieval/context|/root/retrieve_skill\b|/preread_files\b|"
    r"/root/solutions\b|/root/seta_claude_skip\b|"
    r"oracle_top1_skills|skill_libraries|"
    r"\boracle\b|\bretriever\b|retrieval (?:context|ground truth)|retrieved files?|"
    r"previous context from (?:the )?retriever|from (?:the )?retrieval|"
    r"golden ground truth|ground truth from (?:the )?retrieval|"
    r"exact answer|exact solution|reference solution|solution reference|benchmark reference|"
    r"exact values specified in (?:the )?benchmark|known working|historical passing|"
    r"solution from (?:the )?context|exact (?:answer|solution) from (?:the )?context|"
    r"provided script|ready-to-run|one-pass workflow)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    source_key: str
    source_path: str
    line_no: int
    run_name: str
    split: str
    step: int | None
    bench: str
    task_id: str
    task_key: str
    score: float
    raw_score: float | None
    response_sha256: str
    row: dict[str, Any]


class CharTokenizer:
    name_or_path = "char-offset-fallback"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del add_special_tokens
        result: dict[str, list[int] | list[tuple[int, int]]] = {"input_ids": [ord(ch) for ch in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return result

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(i) if 0 <= int(i) <= 0x10FFFF else "" for i in ids)


def main() -> None:
    args = parse_args()
    if not args.input_glob:
        args.input_glob = [DEFAULT_INPUT_GLOB]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(RELAX_ROOT))

    from examples.agent_bench.hard_span_mask import build_hard_span_token_mask

    input_paths = resolve_input_paths(args.input_glob)
    exclude_source_keys = load_manifest_source_keys(args.exclude_manifest)
    manifest_source_keys = load_manifest_source_keys([args.manifest]) if args.manifest else []

    candidates = collect_candidates(
        input_paths,
        min_score=args.min_score,
        score_field=args.score_field,
        update_kinds=set(args.update_kind),
        benches=set(args.bench or []),
    )
    candidate_by_key = {candidate.source_key: candidate for candidate in candidates}

    if manifest_source_keys:
        selected = []
        missing = []
        for source_key in manifest_source_keys:
            candidate = candidate_by_key.get(source_key)
            if candidate is None:
                missing.append(source_key)
            elif candidate.source_key not in exclude_source_keys:
                selected.append(candidate)
        if missing and not args.allow_missing_manifest:
            raise SystemExit(f"manifest references {len(missing)} samples not found; use --allow-missing-manifest to skip")
        if args.count > 0:
            selected = selected[: args.count]
    else:
        available = [candidate for candidate in candidates if candidate.source_key not in exclude_source_keys]
        selected = stratified_select(available, count=args.count, seed=args.seed)

    if len(selected) < args.count and not args.allow_fewer:
        raise SystemExit(f"selected only {len(selected)} samples, requested {args.count}; use --allow-fewer to continue")

    tokenizer, tokenizer_meta = load_tokenizer(args.tokenizer_path, args.tokenizer_mode)
    hard_span_params = {
        "version": args.version,
        "mode": args.action_mask_mode,
        "reasoning_max_chars": args.reasoning_max_chars,
        "final_max_chars": args.final_max_chars,
        "max_response_tokens": args.max_response_tokens,
        "keep_final": not args.no_keep_final,
        "require_useful_reasoning": not args.no_require_useful_reasoning,
        "base_loss_mask": "all_response_tokens_true",
    }

    sample_records: list[dict[str, Any]] = []
    selected_stats: list[dict[str, float | str]] = []
    for ordinal, candidate in enumerate(selected):
        row = candidate.row
        response = row.get("response") or ""
        encoded = encode_with_offsets(tokenizer, response)
        response_token_count = response_token_count_for(row, response, encoded, tokenizer_meta["mode"])
        mask, stats = build_hard_span_token_mask(
            response,
            response_token_count=response_token_count,
            tokenizer=tokenizer,
            mode=args.action_mask_mode,
            base_loss_mask=None,
            sample_status=row.get("status"),
            reasoning_max_chars=args.reasoning_max_chars,
            final_max_chars=args.final_max_chars,
            max_response_tokens=args.max_response_tokens,
            keep_final=not args.no_keep_final,
            require_useful_reasoning=not args.no_require_useful_reasoning,
            version=args.version,
        )
        if str(stats.get("version")) != args.version and not args.allow_version_fallback:
            raise SystemExit(
                f"hard-span implementation returned version={stats.get('version')!r} for requested "
                f"{args.version!r}; use --allow-version-fallback only for intentional fallback audits"
            )
        filtered = materialize_selected_tokens(
            response,
            mask,
            tokenizer,
            encoded,
            include_token_texts=not args.no_token_texts,
        )
        broad_leak_hits = find_leak_hits(filtered["selected_text"], SKILL_LEAK_RE)
        strong_leak_hits = find_leak_hits(filtered["selected_text"], STRONG_SKILL_LEAK_RE)
        selected_stats.append(stats)
        sample_records.append(
            {
                "ordinal": ordinal,
                "sample_id": candidate.sample_id,
                "source": {
                    "path": candidate.source_path,
                    "line_no": candidate.line_no,
                    "run_name": candidate.run_name,
                    "split": candidate.split,
                    "step": candidate.step,
                    "source_key": candidate.source_key,
                    "response_sha256": candidate.response_sha256,
                },
                "task": {
                    "bench": candidate.bench,
                    "task_id": candidate.task_id,
                    "task_key": candidate.task_key,
                    "score": candidate.score,
                    "raw_score": candidate.raw_score,
                    "status": row.get("status"),
                    "update_kind": row.get("update_kind"),
                    "relax_pair_decision": row.get("relax_pair_decision"),
                    "relax_pair_role": row.get("relax_pair_role"),
                },
                "original_trajectory": row,
                "hard_span_audit": {
                    "requested_version": args.version,
                    "returned_version": stats.get("version"),
                    "params": hard_span_params,
                    "stats": stats,
                    "tokenizer": {
                        "mode": tokenizer_meta["mode"],
                        "path": tokenizer_meta["path"],
                        "response_token_count_used": response_token_count,
                        "encoded_token_count": len(encoded["input_ids"]),
                        "dump_response_length": row.get("response_length"),
                    },
                    "existing_train_dump_stats": extract_existing_hard_span_stats(row),
                    "selected_token_ranges": filtered["selected_token_ranges"],
                    "selected_chunks": filtered["selected_chunks"],
                    "selected_text": filtered["selected_text"],
                    "selected_token_texts": filtered["selected_token_texts"],
                    "leak_check": {
                        "broad_skill_pattern_hit_count": len(broad_leak_hits),
                        "broad_skill_pattern_hits": broad_leak_hits[:20],
                        "strong_skill_pattern_hit_count": len(strong_leak_hits),
                        "strong_skill_pattern_hits": strong_leak_hits[:20],
                    },
                },
            }
        )

    out_dir = args.out_dir or default_out_dir(args.version, len(selected), args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"oracle_bc_{args.version}_filtered_{len(selected)}_seed{args.seed}.json"
    output_path = out_dir / output_name

    result = {
        "metadata": {
            "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "script": str(Path(__file__).relative_to(ROOT)),
            "root": str(ROOT),
            "git": git_metadata(),
            "input_globs": args.input_glob,
            "input_file_count": len(input_paths),
            "candidate_count": len(candidates),
            "candidate_counts_by_bench": dict(sorted(Counter(c.bench for c in candidates).items())),
            "selected_count": len(selected),
            "selected_counts_by_bench": dict(sorted(Counter(c.bench for c in selected).items())),
            "selected_counts_by_run": dict(sorted(Counter(c.run_name for c in selected).items())),
            "seed": args.seed,
            "min_score": args.min_score,
            "score_field": args.score_field,
            "update_kinds": args.update_kind,
            "excluded_manifest_count": len(exclude_source_keys),
            "reused_manifest": str(args.manifest) if args.manifest else None,
            "tokenizer": tokenizer_meta,
            "hard_span_params": hard_span_params,
            "mask_gap_marker": MASK_GAP.strip(),
        },
        "summary": build_summary(selected, selected_stats, sample_records),
        "sample_manifest": [
            {
                "sample_id": c.sample_id,
                "source_key": c.source_key,
                "source_path": c.source_path,
                "line_no": c.line_no,
                "run_name": c.run_name,
                "split": c.split,
                "step": c.step,
                "bench": c.bench,
                "task_id": c.task_id,
                "task_key": c.task_key,
                "score": c.score,
                "raw_score": c.raw_score,
                "response_sha256": c.response_sha256,
            }
            for c in selected
        ],
        "samples": sample_records,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "summary": result["summary"]}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--version", default="v3")
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--score-field", choices=["score", "raw_score", "max"], default="score")
    parser.add_argument("--update-kind", action="append", default=["oracle_prompt_bc"])
    parser.add_argument("--bench", action="append", default=[])
    parser.add_argument("--manifest", type=Path, help="Previous result JSON whose sample_manifest should be reused.")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--allow-missing-manifest", action="store_true")
    parser.add_argument("--allow-fewer", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--output-name")
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--tokenizer-mode", choices=["auto", "char"], default="auto")
    parser.add_argument("--no-token-texts", action="store_true", help="Store chunks only, not every selected token string.")
    parser.add_argument("--action-mask-mode", default="tool_call")
    parser.add_argument("--reasoning-max-chars", type=int, default=4096)
    parser.add_argument("--final-max-chars", type=int, default=4096)
    parser.add_argument("--max-response-tokens", type=int, default=0)
    parser.add_argument("--no-keep-final", action="store_true")
    parser.add_argument("--no-require-useful-reasoning", action="store_true")
    parser.add_argument("--allow-version-fallback", action="store_true")
    return parser.parse_args()


def resolve_input_paths(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        raw_matches = glob.glob(str(ROOT / pattern) if not pattern.startswith("/") else pattern)
        for match in raw_matches:
            path = Path(match)
            if path.is_file():
                paths.add(path.resolve())
    return sorted(paths)


def collect_candidates(
    paths: list[Path],
    *,
    min_score: float,
    score_field: str,
    update_kinds: set[str],
    benches: set[str],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_response: set[str] = set()
    for path in paths:
        source_path = relpath(path)
        run_name, split, step = parse_rollout_path(path)
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row_matches_update_kind(row, update_kinds):
                    continue
                score, raw_score = get_scores(row, score_field)
                if score < min_score:
                    continue
                response = row.get("response") or ""
                if not response:
                    continue
                bench, task_id, task_key = get_task_identity(row)
                if benches and bench not in benches:
                    continue
                response_sha = hashlib.sha256(response.encode("utf-8", errors="replace")).hexdigest()
                if response_sha in seen_response:
                    continue
                seen_response.add(response_sha)
                source_key = f"{source_path}:{line_no}:{response_sha}"
                sample_id = hashlib.sha256(source_key.encode()).hexdigest()[:20]
                candidates.append(
                    Candidate(
                        sample_id=sample_id,
                        source_key=source_key,
                        source_path=source_path,
                        line_no=line_no,
                        run_name=run_name,
                        split=split,
                        step=step,
                        bench=bench,
                        task_id=task_id,
                        task_key=task_key,
                        score=score,
                        raw_score=raw_score,
                        response_sha256=response_sha,
                        row=row,
                    )
                )
    return candidates


def row_matches_update_kind(row: dict[str, Any], update_kinds: set[str]) -> bool:
    return row.get("update_kind") in update_kinds or row.get("hybrid_update_kind") in update_kinds


def get_scores(row: dict[str, Any], score_field: str) -> tuple[float, float | None]:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    score_value = reward.get("score", row.get("score", 0.0))
    raw_value = reward.get("raw_score")
    score = safe_float(score_value, 0.0)
    raw_score = None if raw_value is None else safe_float(raw_value, 0.0)
    if score_field == "raw_score" and raw_score is not None:
        return raw_score, raw_score
    if score_field == "max" and raw_score is not None:
        return max(score, raw_score), raw_score
    return score, raw_score


def get_task_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    label = row.get("label") if isinstance(row.get("label"), dict) else {}
    task_key = str(row.get("relax_pair_task_key") or "")
    bench = str(reward.get("bench") or label.get("bench") or "")
    task_id = str(reward.get("task_id") or label.get("task_id") or "")
    if (not bench or not task_id) and "/" in task_key:
        inferred_bench, inferred_task = task_key.split("/", 1)
        bench = bench or inferred_bench
        task_id = task_id or inferred_task
    return bench or "unknown", task_id or task_key or "unknown", task_key or f"{bench}/{task_id}"


def stratified_select(candidates: list[Candidate], *, count: int, seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    by_bench: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_bench[candidate.bench].append(candidate)

    benches = sorted(by_bench)
    if not benches or count <= 0:
        return []

    base_quota = count // len(benches)
    remainder = count % len(benches)
    quotas = {bench: base_quota + (1 if idx < remainder else 0) for idx, bench in enumerate(benches)}
    selected: list[Candidate] = []
    selected_keys: set[str] = set()
    for bench in benches:
        picked = select_task_diverse(by_bench[bench], quotas[bench], rng)
        selected.extend(picked)
        selected_keys.update(candidate.source_key for candidate in picked)

    if len(selected) < count:
        leftovers_by_bench: dict[str, list[Candidate]] = {}
        for bench in benches:
            leftovers = [candidate for candidate in by_bench[bench] if candidate.source_key not in selected_keys]
            rng.shuffle(leftovers)
            leftovers_by_bench[bench] = leftovers
        bench_cursor = 0
        while len(selected) < count and any(leftovers_by_bench.values()):
            bench = benches[bench_cursor % len(benches)]
            bench_cursor += 1
            if not leftovers_by_bench[bench]:
                continue
            candidate = leftovers_by_bench[bench].pop()
            selected.append(candidate)
            selected_keys.add(candidate.source_key)

    return selected[:count]


def select_task_diverse(candidates: list[Candidate], quota: int, rng: random.Random) -> list[Candidate]:
    by_task: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_task[candidate.task_key].append(candidate)
    for rows in by_task.values():
        rng.shuffle(rows)
    tasks = list(by_task)
    rng.shuffle(tasks)

    selected: list[Candidate] = []
    while len(selected) < quota and tasks:
        next_tasks = []
        for task in tasks:
            rows = by_task[task]
            if rows and len(selected) < quota:
                selected.append(rows.pop())
            if rows:
                next_tasks.append(task)
        tasks = next_tasks
    return selected


def load_tokenizer(tokenizer_path: Path, mode: str) -> tuple[Any, dict[str, Any]]:
    if mode == "char":
        return CharTokenizer(), {"mode": "char", "path": None, "loader": "char"}
    errors: list[str] = []
    if tokenizer_path.exists():
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                trust_remote_code=True,
                use_fast=True,
                local_files_only=True,
            )
            encode_with_offsets(tokenizer, "tokenizer smoke")
            return tokenizer, {"mode": "hf", "path": str(tokenizer_path), "loader": "transformers.AutoTokenizer"}
        except Exception as exc:  # pragma: no cover - depends on local env
            errors.append(f"transformers.AutoTokenizer: {exc}")
        try:
            from relax.utils.data.processing_utils import load_tokenizer as relax_load_tokenizer

            tokenizer = relax_load_tokenizer(str(tokenizer_path), trust_remote_code=True)
            encode_with_offsets(tokenizer, "tokenizer smoke")
            return tokenizer, {"mode": "hf", "path": str(tokenizer_path), "loader": "relax.load_tokenizer"}
        except Exception as exc:  # pragma: no cover - depends on local env
            errors.append(f"relax.load_tokenizer: {exc}")
    else:
        errors.append(f"missing tokenizer path: {tokenizer_path}")
    return CharTokenizer(), {"mode": "char_fallback", "path": str(tokenizer_path), "loader": "char", "errors": errors}


def encode_with_offsets(tokenizer: Any, text: str) -> dict[str, Any]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    offsets = encoded["offset_mapping"] if isinstance(encoded, dict) else encoded.offset_mapping
    return {
        "input_ids": list(input_ids),
        "offsets": [(int(start), int(end)) for start, end in offsets],
    }


def response_token_count_for(row: dict[str, Any], response: str, encoded: dict[str, Any], tokenizer_mode: str) -> int:
    if tokenizer_mode.startswith("char"):
        return len(response)
    dump_length = safe_int(row.get("response_length"), 0)
    return dump_length if dump_length > 0 else len(encoded["input_ids"])


def materialize_selected_tokens(
    response: str,
    mask: list[bool],
    tokenizer: Any,
    encoded: dict[str, Any],
    *,
    include_token_texts: bool,
) -> dict[str, Any]:
    token_ranges = merge_true_ranges(mask)
    offsets = encoded["offsets"]
    input_ids = encoded["input_ids"]
    chunks: list[dict[str, Any]] = []
    selected_texts: list[str] = []
    for start, end in token_ranges:
        span_offsets = [(s, e) for s, e in offsets[start : min(end, len(offsets))] if e >= s]
        if span_offsets:
            char_start = min(s for s, _e in span_offsets)
            char_end = max(e for _s, e in span_offsets)
            text = response[char_start:char_end]
        else:
            char_start = None
            char_end = None
            token_ids = input_ids[start : min(end, len(input_ids))]
            text = decode_token_ids(tokenizer, token_ids)
        selected_texts.append(text)
        chunks.append(
            {
                "token_start": start,
                "token_end": end,
                "token_count": end - start,
                "char_start": char_start,
                "char_end": char_end,
                "text": text,
            }
        )

    token_texts: list[str] | None = None
    if include_token_texts:
        token_texts = []
        for idx, keep in enumerate(mask):
            if not keep or idx >= len(input_ids):
                continue
            token_texts.append(decode_token_ids(tokenizer, [input_ids[idx]]))

    return {
        "selected_token_ranges": [{"token_start": s, "token_end": e, "token_count": e - s} for s, e in token_ranges],
        "selected_chunks": chunks,
        "selected_text": MASK_GAP.join(selected_texts),
        "selected_token_texts": token_texts,
    }


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def merge_true_ranges(mask: list[bool]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, keep in enumerate(mask):
        if keep and start is None:
            start = idx
        elif not keep and start is not None:
            ranges.append((start, idx))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))
    return ranges


def find_leak_hits(text: str, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    hits = []
    for match in pattern.finditer(text):
        hits.append(
            {
                "match": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "context": text[max(0, match.start() - 120) : min(len(text), match.end() + 120)],
            }
        )
    return hits


def extract_existing_hard_span_stats(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key.startswith("hard_span_")}


def build_summary(
    selected: list[Candidate],
    stats_rows: list[dict[str, float | str]],
    sample_records: list[dict[str, Any]],
) -> dict[str, Any]:
    numeric_fields = [
        "token_count",
        "token_frac",
        "action_token_count",
        "reasoning_token_count",
        "final_token_count",
        "excluded_skill_token_count",
        "excluded_tool_response_token_count",
        "span_count",
    ]
    hard_summary = {field: summarize_numbers([safe_float(row.get(field), 0.0) for row in stats_rows]) for field in numeric_fields}
    broad_leak_count = sum(
        1 for row in sample_records if row["hard_span_audit"]["leak_check"]["broad_skill_pattern_hit_count"] > 0
    )
    strong_leak_count = sum(
        1 for row in sample_records if row["hard_span_audit"]["leak_check"]["strong_skill_pattern_hit_count"] > 0
    )
    zero_count = sum(1 for row in stats_rows if safe_float(row.get("token_count"), 0.0) <= 0.0)
    return {
        "selected_count": len(selected),
        "selected_counts_by_bench": dict(sorted(Counter(c.bench for c in selected).items())),
        "selected_unique_tasks_by_bench": {
            bench: len({c.task_key for c in selected if c.bench == bench}) for bench in sorted({c.bench for c in selected})
        },
        "hard_span": hard_summary,
        "zero_selected_count": zero_count,
        "selected_broad_skill_pattern_hit_sample_count": broad_leak_count,
        "selected_strong_skill_pattern_hit_sample_count": strong_leak_count,
        "drop_reasons": dict(sorted(Counter(str(row.get("drop_reason", "")) for row in stats_rows).items())),
    }


def summarize_numbers(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "mean": float(statistics.fmean(values)),
        "p10": quantile(ordered, 0.10),
        "p50": quantile(ordered, 0.50),
        "p90": quantile(ordered, 0.90),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    idx = min(max(round((len(ordered) - 1) * q), 0), len(ordered) - 1)
    return float(ordered[idx])


def load_manifest_source_keys(paths: list[Path]) -> list[str]:
    source_keys: list[str] = []
    for path in paths:
        if not path:
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            manifest = payload.get("sample_manifest") or payload.get("samples") or []
        else:
            manifest = payload
        for item in manifest:
            if not isinstance(item, dict):
                continue
            source_key = item.get("source_key") or (item.get("source") or {}).get("source_key")
            if source_key:
                source_keys.append(str(source_key))
    return source_keys


def parse_rollout_path(path: Path) -> tuple[str, str, int | None]:
    parts = path.parts
    if "rollout_result" in parts:
        idx = parts.index("rollout_result")
        run_name = parts[idx - 1] if idx > 0 else "unknown"
        split = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
    else:
        run_name = path.parent.parent.name
        split = path.parent.name
    try:
        step = int(path.stem)
    except ValueError:
        step = None
    return run_name, split, step


def default_out_dir(version: str, count: int, seed: int) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "docs/plot_analysis" / f"oracle_bc_hard_span_{version}_{count}_seed{seed}_{stamp}"


def git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(run_git(["status", "--porcelain"])),
    }


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()

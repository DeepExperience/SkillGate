#!/usr/bin/env python3
"""Offline validation of the M1 "clean transform" algorithm on existing oracle
RL rollout dumps.

M1 = run oracle-skill RL rollouts, then CLEAN each trajectory (strip the
`## Skills (mandatory)` prompt section + delete <skill_reasoning> + the skill-read
tool call + the skill-file tool output), and feed the cleaned no-skill trajectory
through the existing GRPO path.

This script does NOT touch the training code. It loads the flattened-ChatML
rollout dumps (rollout_result/train/<step>.jsonl), simulates the clean on the
`prompt` + `response` strings, and reports:
  - how many trajectories actually read a skill,
  - whether the skill-read is the first action / appears multiple times / shares
    a turn with a real action (mixed turn),
  - whether a non-empty SOLUTION SUFFIX remains after removal,
  - whether ANY residual skill exposure survives the clean (correctness gate),
  - the static "conversion rate" = clean & non-empty-suffix & no-residual.

It also dumps a few before/after examples for eyeballing.

Usage:
  python ops/workflows/rl_data_prep/validate_m1_clean_offline.py \
      --max-files-per-run 3 --examples 3
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "experiments/rl/runs"

ORACLE_RUNS = [
    "4bench_cp2_70k_lr1e6_active128_nolen_skillgate30_oracle1_from_sft_20260614_060952",
    "4bench_cp2_70k_lr1e6_active128_oracle1_baseline_noskillrw_nogate_from_sft_20260615_161851",
]

# ---- detectors (mirror Relax/examples/agent_bench/skill_group_reward.py) ----
SKILL_FILE_RE = re.compile(r"/(?:root/)?\.claude/skills/[^\s'\"<>|;&]+/(?:SKILL|README)\.md")
AVAILABLE_SKILLS_RE = re.compile(r"<available_skills>.*?</available_skills>", re.S)
SKILL_REASONING_RE = re.compile(r"<skill_reasoning>.*?</skill_reasoning>", re.S)
# prompt section strip (mirror make_4bench_factual_noskill_parquet.py)
SKILLS_SECTION_RE = re.compile(r"\n## Skills \(mandatory\)\n.*?(?=\n## Memory Recall\n)", re.S)
EXEC_READ_RE = re.compile(r"\b(cat|sed|awk|grep|head|tail|less|more|python3?|perl|ruby|node)\b")
# leak markers (mirror export_llamafactory.py HIDDEN_INSTRUCTION_LEAK_MARKERS, partial)
LEAK_MARKERS = [
    "read at least one retrieved skill",
    "best solved using your own general knowledge",
    "Mandatory:",
    "retrieved skill file",
]

# ChatML turn split: response is a flattened transcript. Roles appear as
# <|im_start|>{role}\n ... <|im_end|>. The first assistant turn has no leading
# <|im_start|> (it continues from the prompt's trailing assistant header).
IM_START_RE = re.compile(r"<\|im_start\|>(\w+)\n")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)


def split_turns(response: str) -> list[tuple[str, str]]:
    """Return list of (role, content). First chunk (before any <|im_start|>) is
    the continuation of the assistant turn the prompt opened."""
    turns: list[tuple[str, str]] = []
    pos = 0
    first = IM_START_RE.search(response)
    if first is None:
        return [("assistant", response)]
    if first.start() > 0:
        turns.append(("assistant", response[: first.start()]))
    for m in IM_START_RE.finditer(response):
        role = m.group(1)
        start = m.end()
        nxt = IM_START_RE.search(response, start)
        end = nxt.start() if nxt else len(response)
        content = response[start:end]
        content = content.replace("<|im_end|>", "")
        turns.append((role, content))
        pos = end
    return turns


def turn_skill_reads(content: str) -> tuple[int, int]:
    """Return (n_skill_read_calls, n_non_skill_action_calls) for an assistant turn."""
    skill = 0
    other = 0
    for fn, args in TOOL_CALL_RE.findall(content):
        fn = fn.strip()
        blob = fn + " " + args
        is_skill = bool(SKILL_FILE_RE.search(blob)) and (
            fn == "read" or (fn == "exec" and EXEC_READ_RE.search(args)) or fn in ("read",)
        )
        # also treat any tool call whose args contain a SKILL path read-ish as skill
        if SKILL_FILE_RE.search(blob) and fn in ("read", "exec", "process"):
            is_skill = True
        if is_skill:
            skill += 1
        else:
            other += 1
    return skill, other


def content_is_skill_output(content: str) -> bool:
    """A tool/user observation carrying SKILL.md content."""
    if SKILL_FILE_RE.search(content):
        return True
    # line-numbered markdown frontmatter typical of a SKILL.md read result
    if "name:" in content and "description:" in content and "tool_response" in content:
        # heuristic: skill files have YAML frontmatter; only flag if also skill-ish
        return False
    return False


def clean_response(turns: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], dict]:
    """Simulate M1 clean on the turn list. Returns (cleaned_turns, stats)."""
    stats = {
        "n_skill_read_turns_dropped": 0,
        "n_skill_outputs_dropped": 0,
        "n_mixed_turns": 0,
        "n_skill_reasoning_blocks": 0,
        "skill_read_first": False,
        "n_skill_read_calls_total": 0,
    }
    cleaned: list[tuple[str, str]] = []
    drop_next_tool_output = 0  # how many subsequent observation turns to drop
    first_action_seen = False
    for idx, (role, content) in enumerate(turns):
        if role == "assistant":
            n_skill, n_other = turn_skill_reads(content)
            stats["n_skill_read_calls_total"] += n_skill
            # strip skill_reasoning text regardless
            sr = SKILL_REASONING_RE.findall(content)
            if sr:
                stats["n_skill_reasoning_blocks"] += len(sr)
                content = SKILL_REASONING_RE.sub("", content)
            if n_skill > 0 and n_other == 0:
                # pure skill-read turn -> drop entirely + drop its paired output(s)
                stats["n_skill_read_turns_dropped"] += 1
                if not first_action_seen:
                    stats["skill_read_first"] = True
                drop_next_tool_output += n_skill
                continue
            if n_skill > 0 and n_other > 0:
                # mixed turn: remove only skill-read tool_call blocks
                stats["n_mixed_turns"] += 1
                def _drop_skill_block(m):
                    fn = m.group(1).strip()
                    blob = fn + " " + m.group(2)
                    if SKILL_FILE_RE.search(blob):
                        return ""
                    return m.group(0)
                content = TOOL_CALL_RE.sub(_drop_skill_block, content)
                drop_next_tool_output += n_skill
            if n_other > 0 or content.strip():
                first_action_seen = True
            cleaned.append((role, content))
        else:
            # observation / tool / user turn
            if drop_next_tool_output > 0 and (
                content_is_skill_output(content) or "<tool_response" in content
            ):
                stats["n_skill_outputs_dropped"] += 1
                drop_next_tool_output -= 1
                continue
            cleaned.append((role, content))
    return cleaned, stats


SOFT_LEAK_RE = re.compile(
    r"(?i)(SKILL\.md|README\.md|retrieved skill|the skill\b|read the skill|skill file|"
    r"skill says|according to the skill|available[_ ]skills|\.claude/skills|provided skill|"
    r"\bskills?\b(?=[^a-z]*(file|library|directory|entry|describ|provid|retriev)))"
)


_TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")


def scrub_skill_prose(content: str) -> tuple[str, int]:
    """Remove skill-referencing sentences from assistant prose/think while
    preserving <tool_call> action blocks verbatim. Returns (scrubbed, chars_removed)."""
    # protect tool_call blocks
    blocks = []
    def _stash(m):
        blocks.append(m.group(0))
        return f"\x00TC{len(blocks)-1}\x00"
    protected = _TOOLCALL_BLOCK_RE.sub(_stash, content)
    removed = 0
    out_sentences = []
    for sent in _SENT_SPLIT_RE.split(protected):
        if "\x00TC" in sent:  # keep any sentence carrying an action block
            out_sentences.append(sent)
            continue
        if SOFT_LEAK_RE.search(sent):
            removed += len(sent)
            continue
        out_sentences.append(sent)
    scrubbed = " ".join(s for s in out_sentences if s.strip() or "\x00TC" in s)
    for i, b in enumerate(blocks):
        scrubbed = scrubbed.replace(f"\x00TC{i}\x00", b)
    return scrubbed, removed


def scrub_turns(cleaned_turns: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int, int]:
    out = []
    total_removed = 0
    total_assistant_chars = 0
    for role, content in cleaned_turns:
        if role == "assistant":
            total_assistant_chars += len(content)
            sc, rem = scrub_skill_prose(content)
            total_removed += rem
            out.append((role, sc))
        else:
            out.append((role, content))
    return out, total_removed, total_assistant_chars


def soft_leak_in_assistant(cleaned_turns: list[tuple[str, str]]) -> list[str]:
    """Skill-referencing phrases in KEPT assistant (loss=1) text — these get trained."""
    hits = []
    for role, content in cleaned_turns:
        if role != "assistant":
            continue
        for m in SOFT_LEAK_RE.finditer(content):
            hits.append(m.group(0).strip().lower())
    return hits


def has_solution_suffix(cleaned_turns: list[tuple[str, str]]) -> bool:
    """Non-empty solution suffix = >=1 assistant turn with a real action or a
    substantive final answer after cleaning."""
    for role, content in cleaned_turns:
        if role != "assistant":
            continue
        _, n_other = turn_skill_reads(content)
        if n_other > 0:
            return True
        txt = SKILL_REASONING_RE.sub("", content).strip()
        if len(txt) > 40 and "<tool_call>" not in txt:
            return True  # substantive final answer
    return False


def residual_skill(prompt_clean: str, cleaned_turns: list[tuple[str, str]]) -> list[str]:
    """Return list of residual skill-exposure markers surviving the clean."""
    res = []
    full = prompt_clean + "\n" + "\n".join(c for _, c in cleaned_turns)
    if SKILL_FILE_RE.search(full):
        res.append("skill_path")
    if AVAILABLE_SKILLS_RE.search(full):
        res.append("available_skills")
    if "<skill_reasoning>" in full:
        res.append("skill_reasoning")
    if "## Skills (mandatory)" in full:
        res.append("skills_section")
    for mk in LEAK_MARKERS:
        if mk in full:
            res.append(f"leak:{mk[:20]}")
    return res


def strip_prompt_skills(prompt: str) -> tuple[str, bool]:
    stripped, n = SKILLS_SECTION_RE.subn("\n", prompt, count=1)
    return stripped, bool(n)


def process_line(obj: dict) -> dict | None:
    reward = obj.get("reward") or {}
    raw = reward.get("raw_score")
    bench = reward.get("bench") or (obj.get("label") or {}).get("bench")
    task_id = reward.get("task_id") or (obj.get("label") or {}).get("task_id")
    prompt = obj.get("prompt") or ""
    response = obj.get("response") or ""
    prompt_clean, prompt_stripped = strip_prompt_skills(prompt)
    turns = split_turns(response)
    cleaned, st = clean_response(turns)
    read_skill = st["n_skill_read_calls_total"] > 0 or bool(SKILL_FILE_RE.search(response))
    suffix_ok = has_solution_suffix(cleaned)
    residual = residual_skill(prompt_clean, cleaned)
    soft = soft_leak_in_assistant(cleaned)
    scrubbed_turns, removed_chars, asst_chars = scrub_turns(cleaned)
    soft_after = soft_leak_in_assistant(scrubbed_turns)
    residual_after = residual_skill(prompt_clean, scrubbed_turns)
    suffix_after = has_solution_suffix(scrubbed_turns)
    return {
        "soft_leak": soft,
        "soft_leak_after_scrub": soft_after,
        "residual_after_scrub": residual_after,
        "suffix_ok_after_scrub": suffix_after,
        "scrub_removed_chars": removed_chars,
        "assistant_chars": asst_chars,
        "bench": bench,
        "task_id": task_id,
        "raw_score": raw,
        "success": (raw is not None and raw >= 1.0),
        "prompt_had_skills_section": prompt_stripped,
        "read_skill": read_skill,
        "suffix_ok": suffix_ok,
        "residual": residual,
        "stats": st,
        "_prompt_clean": prompt_clean,
        "_cleaned_turns": cleaned,
        "_orig_response": response,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files-per-run", type=int, default=3)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "z_cc_terminal_imgs/m1_offline_validation"))
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    segments: dict[str, Path] = {}
    for run_json in RUNS_ROOT.glob("*/segments/*/run.json"):
        try:
            manifest = json.loads(run_json.read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        for key in (run_json.parent.name, manifest.get("segment_id"), manifest.get("legacy_run_id")):
            if key:
                segments[str(key)] = run_json.parent

    files = []
    for run in ORACLE_RUNS:
        segment = segments.get(run)
        d = (segment / "rollout_result" / "train") if segment else Path("/__missing_segment__")
        if not d.is_dir():
            print(f"[warn] missing canonical segment for legacy run {run!r}")
            continue
        step_files = sorted(d.glob("*.jsonl"), key=lambda p: int(p.stem))
        # sample evenly across the run
        if len(step_files) > args.max_files_per_run:
            idxs = [int(i * (len(step_files) - 1) / (args.max_files_per_run - 1)) for i in range(args.max_files_per_run)]
            step_files = [step_files[i] for i in sorted(set(idxs))]
        for f in step_files:
            files.append((run, f))

    print(f"[info] scanning {len(files)} step files")
    rows = []
    for run, f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                r = process_line(obj)
                if r:
                    r["_run"] = run.split("_oracle1")[0][-12:]
                    r["_file"] = f.name
                    rows.append(r)

    total = len(rows)
    succ = [r for r in rows if r["success"]]
    succ_read = [r for r in succ if r["read_skill"]]
    # conversion: among successful trajectories that read a skill, how many clean
    # to a non-empty suffix with NO residual skill exposure.
    converted = [r for r in succ_read if r["suffix_ok"] and not r["residual"]]
    # correctness gate: any residual exposure after clean (over ALL rows = bad)
    residual_any = [r for r in rows if r["residual"]]

    bench_succ = Counter(r["bench"] for r in succ)
    bench_conv = Counter(r["bench"] for r in converted)
    residual_kinds = Counter(k for r in rows for k in r["residual"])
    soft_traj = [r for r in succ_read if r["soft_leak"]]
    soft_phrase_counts = Counter(p for r in succ_read for p in r["soft_leak"])

    summary = {
        "files_scanned": len(files),
        "total_trajectories": total,
        "successful": len(succ),
        "successful_read_skill": len(succ_read),
        "successful_read_skill_pct": round(100 * len(succ_read) / max(len(succ), 1), 1),
        "converted_clean": len(converted),
        "conversion_rate_pct": round(100 * len(converted) / max(len(succ_read), 1), 1),
        "rows_with_residual_after_clean": len(residual_any),
        "residual_pct_of_all": round(100 * len(residual_any) / max(total, 1), 1),
        "residual_kinds": dict(residual_kinds),
        "skill_read_first_pct": round(
            100 * sum(1 for r in succ_read if r["stats"]["skill_read_first"]) / max(len(succ_read), 1), 1
        ),
        "mixed_turn_trajectories": sum(1 for r in succ_read if r["stats"]["n_mixed_turns"] > 0),
        "multi_skill_read_trajectories": sum(1 for r in succ_read if r["stats"]["n_skill_read_calls_total"] > 1),
        "by_bench_successful": dict(bench_succ),
        "by_bench_converted": dict(bench_conv),
        "soft_verbal_leak_trajectories": len(soft_traj),
        "soft_verbal_leak_pct_of_read": round(100 * len(soft_traj) / max(len(succ_read), 1), 1),
        "soft_leak_top_phrases": dict(soft_phrase_counts.most_common(15)),
        "AFTER_SCRUB": {
            "soft_leak_trajectories": sum(1 for r in succ_read if r["soft_leak_after_scrub"]),
            "soft_leak_pct_of_read": round(
                100 * sum(1 for r in succ_read if r["soft_leak_after_scrub"]) / max(len(succ_read), 1), 1
            ),
            "residual_trajectories": sum(1 for r in succ_read if r["residual_after_scrub"]),
            "suffix_ok_pct": round(
                100 * sum(1 for r in succ_read if r["suffix_ok_after_scrub"]) / max(len(succ_read), 1), 1
            ),
            "fully_clean_pct": round(
                100 * sum(1 for r in succ_read if not r["soft_leak_after_scrub"] and not r["residual_after_scrub"] and r["suffix_ok_after_scrub"]) / max(len(succ_read), 1), 1
            ),
            "median_assistant_chars_removed_pct": round(
                100 * (sum(r["scrub_removed_chars"] for r in succ_read) / max(sum(r["assistant_chars"] for r in succ_read), 1)), 1
            ),
            "remaining_soft_phrases": dict(Counter(p for r in succ_read for p in r["soft_leak_after_scrub"]).most_common(10)),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # dump examples (prefer converted ones, then ones with residual to inspect failures)
    ex_lines = ["# M1 clean — before/after examples\n"]
    picks = converted[: args.examples] + [r for r in residual_any if r["success"]][:2]
    for i, r in enumerate(picks):
        ex_lines.append(f"\n## Example {i} — bench={r['bench']} task={r['task_id']} raw={r['raw_score']} residual={r['residual']}\n")
        ex_lines.append(f"stats={json.dumps(r['stats'], ensure_ascii=False)}\n")
        ex_lines.append("\n### ORIG response (first 1500 chars):\n```\n" + r["_orig_response"][:1500] + "\n```\n")
        cleaned_txt = "\n".join(f"<{role}> {content[:300]}" for role, content in r["_cleaned_turns"])
        ex_lines.append("\n### CLEANED turns (roles + first 300 chars each, first 12):\n```\n" + "\n".join(cleaned_txt.splitlines()[:40]) + "\n```\n")
    (outdir / "examples.md").write_text("".join(ex_lines))
    print(f"\n[done] wrote {outdir}/summary.json + examples.md")


if __name__ == "__main__":
    main()

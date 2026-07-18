# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Custom multi-turn rollout for the agent_bench Relax campaign.

Adapted from ``examples/deepeyes/rollout.py`` with all multimodal handling
removed (we run text-only OpenClaw tool-calling agents) and the env module
swapped from ``env_deepeyes`` to :mod:`examples.agent_bench.env_agent_bench`.

Selected via ``--custom-generate-function-path examples.agent_bench.rollout.generate``.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any

from examples.agent_bench.env_agent_bench import build_env  # noqa: F401  (used via attribute)
from examples.agent_bench.skill_group_reward import (
    SKILL_FILE_RE,
    _tool_call_reads_skill,
)
from relax.engine.rollout.sglang_rollout import GenerateState
from relax.utils.data.processing_utils import _ENCODE_EXECUTOR
from relax.utils.http_utils import post
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)


DEFAULT_ENV_MODULE = "examples.agent_bench.env_agent_bench"
# Per-sample wallclock cap. A slow agent task (e.g. PDF OCR loop) without
# this cap can run for hours, starving actor TP ranks waiting on
# transfer_queue.get_meta, which then trips the NCCL 1800s TP-broadcast
# watchdog and SIGABRT-kills the whole actor service. Default 900s = 15 min
# is well under the 1800s NCCL limit even with stragglers stacking.
_DEFAULT_WALLCLOCK_CAP_SEC = 900.0


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


# The Relax rollout batch is algorithmic (rollout_batch_size × n_samples_per_prompt)
# and should stay large enough for GRPO statistics. The *environment* fan-out is
# operationally different: every active Docker-mode sample owns a long-lived
# `docker exec -i ... bash`. Letting all samples stay active at once can trip
# Docker setup/control-plane limits and turn infra failures into noisy rewards.
#
# Cap the number of concurrently active environments without changing the
# GRPO group size or training global batch. Samples beyond the cap wait here
# before they create containers / persistent shells.
_ACTIVE_ENV_CONCURRENCY = _positive_int_env("AGENT_BENCH_ACTIVE_ENV_CONCURRENCY", 12)
_ACTIVE_ENV_SEMAPHORE = asyncio.Semaphore(_ACTIVE_ENV_CONCURRENCY)


@asynccontextmanager
async def _active_env_slot(task_id: str):
    logger.debug(
        "[%s] waiting for active env slot (cap=%s)",
        task_id,
        _ACTIVE_ENV_CONCURRENCY,
    )
    await _ACTIVE_ENV_SEMAPHORE.acquire()
    try:
        yield
    finally:
        _ACTIVE_ENV_SEMAPHORE.release()


# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------
def _load_env_module(env_path: str | None):
    target = env_path or DEFAULT_ENV_MODULE
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    fn = getattr(env_module, "build_env", None)
    if not callable(fn):
        raise ValueError(
            f"Environment module {env_module.__name__} must expose `build_env(sample, args)`."
        )
    return fn(sample, args)


# ---------------------------------------------------------------------------
# Encoding helpers (text-only — no image/video path)
# ---------------------------------------------------------------------------
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _encode_observation_for_generation(
    tokenizer,
    message: dict,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """Render a single ``{role, content}`` observation into the token sequence
    Relax appends to the running sample.

    The chat-template-trim trick mirrors deepeyes: render a known-prefix
    (DUMMY_MESSAGES) twice, subtract the prefix length, keep only the
    observation-relevant suffix. Critical for **not** double-injecting tool
    schema (see P0.3 contract in ``rl_training/deploy_check``):

    - ``apply_chat_template`` is passed through, but
    - ``tools`` is **never** set (we don't supply a `tools=` kwarg here),
      so ``chat_template.jinja:45-53`` stays on its ``{%- if tools -%}`` false
      branch and does not append a second ``# Tools`` block.
    """
    apply_kwargs = apply_chat_template_kwargs or {}
    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)[trim_length:]
    else:
        prompt_ids = tokenizer.encode(message.get("content", ""), add_special_tokens=False)
    return prompt_ids


def _prepare_initial_inputs(sample: Sample, tokenizer):
    if isinstance(sample.prompt, list):
        # Apply chat template on the full messages list so the rendered
        # initial prompt mirrors SFT-time text byte-for-byte.
        formatted = tokenizer.apply_chat_template(
            sample.prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    return prompt_ids


def _opsd_mode_enabled() -> bool:
    return os.environ.get("RELAX_OPSD_MODE", "0").lower() in {"1", "true", "yes", "on"}


def _opsd_attach_teacher_prompt_ids(sample: Sample, tokenizer) -> None:
    """OPSD: render the paired oracle-arm prompt (donor) into token ids.

    The donor prompt was stashed on the sample by the pair-atomic candidate
    prep (sglang_rollout._prepare_pair_atomic_candidates). Rendering uses the
    exact generation-path renderer (_prepare_initial_inputs) so the teacher
    prompt ids are byte-identical to what a real oracle-arm rollout would see.
    The ids are consumed (and popped) by sglang_rollout._opsd_score_group.
    """
    if not _opsd_mode_enabled():
        return
    metadata = sample.metadata if isinstance(sample.metadata, dict) else None
    if not metadata or "opsd_teacher_prompt" not in metadata:
        return
    if int(sample.response_length or 0) <= 0:
        return
    donor_prompt = metadata.get("opsd_teacher_prompt")
    if not donor_prompt:
        return
    try:
        donor_shim = Sample(prompt=donor_prompt)
        teacher_prompt_ids = _prepare_initial_inputs(donor_shim, tokenizer)
    except Exception as exc:
        logger.warning(
            "[%s] OPSD teacher prompt render failed (%s); sample will use inert fallback teacher log-probs.",
            _sample_extra_info(sample).get("task_id", "?"),
            exc,
        )
        return
    if teacher_prompt_ids:
        metadata["opsd_teacher_prompt_ids"] = list(teacher_prompt_ids)
        metadata["opsd_teacher_prompt_len"] = len(teacher_prompt_ids)


# ---------------------------------------------------------------------------
# M1 "skill-free shadow" clean transform
#
# Idea: docs/idea/skill_free_shadow_update.md (M1 variant). Run oracle-skill
# rollouts for exploration, then strip ALL skill exposure from each trajectory
# and feed the cleaned no-skill trajectory through the EXISTING GRPO path. The
# model is updated as if it had solved the task without ever reading a skill.
#
# Gated by RELAX_M1_CLEAN=1. Requires --use-tis OFF (cleaned tokens are
# synthetic; per-token rollout_log_probs no longer correspond) and
# RELAX_SKILL_GROUP_REWARD=0 (skill subgroup reward is meaningless post-clean).
# Validated offline: ops/workflows/rl_data_prep/validate_m1_clean_offline.py
# (99.5% hard conversion; soft verbal leak 88.9%->5.5% after prose scrub;
# residual gate zeros loss_mask on the ~few% that still leak).
# ---------------------------------------------------------------------------
# Strip the whole "## Skills (mandatory)" prompt section (mirror
# ops/workflows/rl_data_prep/make_4bench_factual_noskill_parquet.py).
_M1_SKILLS_SECTION_RE = re.compile(r"\n## Skills \(mandatory\)\n.*?(?=\n## Memory Recall\n)", re.S)
_M1_SKILL_REASONING_RE = re.compile(r"<skill_reasoning>.*?</skill_reasoning>", re.S)
_M1_AVAILABLE_SKILLS_RE = re.compile(r"<available_skills>.*?</available_skills>", re.S)
_PROMPT_ONLY_PRELOADED_SECTION_RE = re.compile(
    r"\n## Preloaded (?:Oracle )?Skill(?: Content)?\n.*?(?=\n## Memory Recall\n|$)",
    re.S,
)
_PROMPT_ONLY_PRELOADED_XML_RE = re.compile(
    r"<preloaded_(?:oracle|top1)_skill>.*?</preloaded_(?:oracle|top1)_skill>",
    re.S,
)
_M1_FUNC_BLOCK_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
_M1_TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>\s*<function=[^>\n]+>.*?</function>\s*</tool_call>", re.S)
_M1_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")
# Skill-referencing prose phrases that, if left in trained assistant text, teach
# the model to narrate reading skills it never read.
_M1_SOFT_LEAK_RE = re.compile(
    r"(?i)(SKILL\.md|retrieved skill|the skill\b|read the skill|skill file|"
    r"skill says|according to the skill|available[_ ]skills|\.claude/skills|provided skill|"
    r"\bskills?\b(?=[^a-z]*(file|library|directory|entry|describ|provid|retriev)))"
)
_M1_SHADOW_UPDATE_KINDS = {"oracle_shadow", "shadow", "m1_shadow", "hybrid_shadow"}
_PROMPT_ONLY_SHADOW_UPDATE_KINDS = {"oracle_prompt_bc", "oracle_direct_bc", "prompt_shadow"}


def _m1_enabled() -> bool:
    return os.environ.get("RELAX_M1_CLEAN", "0").lower() in {"1", "true", "yes", "on"}


def _prompt_only_clean_enabled() -> bool:
    return os.environ.get("RELAX_PROMPT_ONLY_SHADOW_CLEAN", "0").lower() in {"1", "true", "yes", "on"}


def _sample_extra_info(sample: Sample) -> dict:
    metadata = sample.metadata or {}
    if not isinstance(metadata, dict):
        return {}
    extra = metadata.get("extra_info")
    if isinstance(extra, dict):
        return extra
    return metadata


def _sample_update_kind(sample: Sample) -> str:
    extra = _sample_extra_info(sample)
    return str(extra.get("update_kind") or extra.get("hybrid_update_kind") or "").strip().lower()


def _m1_enabled_for_sample(sample: Sample) -> bool:
    if not _m1_enabled():
        return False
    update_kind = _sample_update_kind(sample)
    if update_kind:
        return update_kind in _M1_SHADOW_UPDATE_KINDS
    return True


def _prompt_only_enabled_for_sample(sample: Sample) -> bool:
    if not _prompt_only_clean_enabled():
        return False
    update_kind = _sample_update_kind(sample)
    return update_kind in _PROMPT_ONLY_SHADOW_UPDATE_KINDS


def _m1_turn_skill_counts(text: str) -> tuple[int, int]:
    """(n_skill_read_calls, n_other_calls) in an assistant turn's raw text."""
    n_skill = n_other = 0
    for m in _M1_FUNC_BLOCK_RE.finditer(text):
        name = m.group(1).strip()
        args = m.group(2)
        is_skill = _tool_call_reads_skill(name, args) or (
            bool(SKILL_FILE_RE.search(name + " " + args)) and name in ("read", "exec", "process")
        )
        if is_skill:
            n_skill += 1
        else:
            n_other += 1
    return n_skill, n_other


def _m1_drop_skill_func_blocks(text: str) -> str:
    """Remove only the skill-read <tool_call> blocks from a mixed assistant turn."""
    def _repl(m: re.Match) -> str:
        if SKILL_FILE_RE.search(m.group(0)):
            return ""
        return m.group(0)
    return _M1_TOOLCALL_BLOCK_RE.sub(_repl, text)


def _m1_scrub_prose(text: str) -> str:
    """Remove skill-referencing sentences from assistant prose/think while
    preserving <tool_call> action blocks verbatim."""
    text = _M1_SKILL_REASONING_RE.sub("", text)
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"\x00TC{len(blocks) - 1}\x00"

    protected = _M1_TOOLCALL_BLOCK_RE.sub(_stash, text)
    kept = []
    for sent in _M1_SENT_SPLIT_RE.split(protected):
        if "\x00TC" in sent:
            kept.append(sent)
            continue
        if _M1_SOFT_LEAK_RE.search(sent):
            continue
        kept.append(sent)
    out = " ".join(s for s in kept if s.strip() or "\x00TC" in s)
    for i, b in enumerate(blocks):
        out = out.replace(f"\x00TC{i}\x00", b)
    return out


def _m1_strip_prompt_skills(prompt):
    """Return a no-skill copy of sample.prompt (messages list or string)."""
    if isinstance(prompt, list):
        new_msgs = []
        for msg in prompt:
            content = msg.get("content")
            if isinstance(content, str):
                content, _ = _M1_SKILLS_SECTION_RE.subn("\n", content, count=1)
            new_msgs.append({**msg, "content": content})
        return new_msgs
    if isinstance(prompt, str):
        out, _ = _M1_SKILLS_SECTION_RE.subn("\n", prompt, count=1)
        return out
    return prompt


def _prompt_only_strip_privileged_prompt(prompt):
    """Remove rollout-only skill context from sample.prompt without touching the response."""
    removed = 0

    def _clean_content(content: str) -> str:
        nonlocal removed
        content, count = _M1_SKILLS_SECTION_RE.subn("\n", content, count=1)
        removed += count
        content, count = _PROMPT_ONLY_PRELOADED_SECTION_RE.subn("\n", content, count=1)
        removed += count
        content, count = _PROMPT_ONLY_PRELOADED_XML_RE.subn("", content)
        removed += count
        return content

    if isinstance(prompt, list):
        new_msgs = []
        for msg in prompt:
            content = msg.get("content")
            if isinstance(content, str):
                content = _clean_content(content)
            new_msgs.append({**msg, "content": content})
        return new_msgs, removed
    if isinstance(prompt, str):
        return _clean_content(prompt), removed
    return prompt, removed


def _prompt_only_prompt_has_privileged_skill(text: str) -> bool:
    return bool(
        SKILL_FILE_RE.search(text)
        or _M1_AVAILABLE_SKILLS_RE.search(text)
        or "## Skills (mandatory)" in text
        or "## Preloaded Oracle Skill" in text
        or "## Preloaded Skill Content" in text
        or "<preloaded_oracle_skill>" in text
        or "<preloaded_top1_skill>" in text
    )


def _m1_text_has_skill(text: str) -> bool:
    return bool(
        SKILL_FILE_RE.search(text)
        or _M1_AVAILABLE_SKILLS_RE.search(text)
        or "<skill_reasoning>" in text
        or "## Skills (mandatory)" in text
        or _M1_SOFT_LEAK_RE.search(text)
    )


def _m1_messages_text(prompt) -> str:
    if isinstance(prompt, list):
        return "\n".join(m.get("content") or "" for m in prompt if isinstance(m.get("content"), str))
    return prompt or ""


def _m1_clip(text: str | None, limit: int = 16000) -> str | None:
    if text is None or len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n...<clip {len(text) - limit} chars>...\n" + text[-half:]


def _m1_write_audit(args, sample, *, mode, residual, stats, n_trained, resp_len,
                    orig_response, cleaned_response, loss_mask_len) -> None:
    """Append a per-sample M1-clean audit record so the no-skill setting is
    verifiable from disk: ORIGINAL traj + CLEANED traj + loss-mask summary +
    drop reason, for EVERY sample (kept AND aborted). Aborted samples are
    refilled out of the main rollout dump, so this is the only place their
    audit survives. Best-effort; never raises into the rollout."""
    try:
        result_dir = getattr(args, "rollout_result_dir", None)
        if not result_dir:
            return
        extra = _sample_extra_info(sample)
        d = os.path.join(result_dir, "m1_audit")
        os.makedirs(d, exist_ok=True)
        rec = {
            "task_id": extra.get("task_id", "?"),
            "bench": extra.get("bench"),
            "update_kind": extra.get("update_kind") or extra.get("hybrid_update_kind"),
            "mode": mode,                       # cleaned | aborted_residual | aborted_degenerate
            "residual_skill": bool(residual),
            "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
            "cleaned_response_len_tokens": int(resp_len),
            "n_trained_tokens": int(n_trained),  # ==sum(loss_mask); 0 => zero-gradient/dropped
            "loss_mask_len": int(loss_mask_len),
            "stats": stats,
            "orig_response": _m1_clip(orig_response),
            "cleaned_response": _m1_clip(cleaned_response),
        }
        with open(os.path.join(d, "samples.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _prompt_only_write_audit(
    args,
    sample,
    *,
    mode: str,
    residual: bool,
    stats: dict,
    n_trained: int,
    resp_len: int,
    orig_prompt,
    cleaned_prompt,
) -> None:
    try:
        result_dir = getattr(args, "rollout_result_dir", None)
        if not result_dir:
            return
        extra = _sample_extra_info(sample)
        d = os.path.join(result_dir, "prompt_only_shadow_audit")
        os.makedirs(d, exist_ok=True)
        rec = {
            "task_id": extra.get("task_id", "?"),
            "bench": extra.get("bench"),
            "update_kind": extra.get("update_kind") or extra.get("hybrid_update_kind"),
            "mode": mode,
            "residual_privileged_prompt": bool(residual),
            "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
            "response_len_tokens": int(resp_len),
            "n_trained_tokens": int(n_trained),
            "stats": stats,
            "orig_prompt": _m1_clip(_m1_messages_text(orig_prompt)),
            "cleaned_prompt": _m1_clip(_m1_messages_text(cleaned_prompt)),
        }
        with open(os.path.join(d, "samples.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _clean_prompt_only_shadow(sample: Sample, response_tokens: list[int], tokenizer, args) -> None:
    """Retain the oracle-conditioned response, but train it under a no-skill prompt."""
    orig_prompt = sample.prompt
    orig_prompt_ids = _prepare_initial_inputs(sample, tokenizer)
    response_tokens = list(response_tokens)
    if len(sample.tokens) >= len(orig_prompt_ids):
        response_tokens = list(sample.tokens[len(orig_prompt_ids):])

    loss_mask = list(sample.loss_mask or [])
    logprobs = list(sample.rollout_log_probs or [])
    stats = {
        "old_prompt_len": int(len(orig_prompt_ids)),
        "response_len": int(len(response_tokens)),
        "removed_prompt_blocks": 0,
    }

    cleaned_prompt, removed = _prompt_only_strip_privileged_prompt(orig_prompt)
    stats["removed_prompt_blocks"] = int(removed)
    sample.prompt = cleaned_prompt
    new_prompt_ids = _prepare_initial_inputs(sample, tokenizer)
    stats["new_prompt_len"] = int(len(new_prompt_ids))

    prompt_text = _m1_messages_text(sample.prompt)
    residual = _prompt_only_prompt_has_privileged_skill(prompt_text)
    n_trained = sum(loss_mask)
    length_mismatch = len(loss_mask) != len(response_tokens) or len(logprobs) != len(response_tokens)
    degenerate = (len(response_tokens) == 0) or (n_trained == 0)
    if removed <= 0 or residual or degenerate or length_mismatch:
        sample.tokens = list(new_prompt_ids)
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.response_length = 0
        sample.response = ""
        reason = "prompt_only_residual_privileged_prompt"
        if removed <= 0:
            reason = "prompt_only_no_privileged_prompt_removed"
        elif degenerate:
            reason = "prompt_only_degenerate_empty"
        elif length_mismatch:
            reason = "prompt_only_length_mismatch"
        _mark_infra_aborted(
            sample,
            reason=reason,
            category="prompt_only_shadow_clean_drop",
            extra={
                "prompt_only_stats": stats,
                "prompt_only_residual": bool(residual),
                "prompt_only_length_mismatch": bool(length_mismatch),
                "prompt_only_trained_tokens": int(n_trained),
            },
        )
        _prompt_only_write_audit(
            args,
            sample,
            mode="aborted",
            residual=residual,
            stats=stats,
            n_trained=n_trained,
            resp_len=len(response_tokens),
            orig_prompt=orig_prompt,
            cleaned_prompt=sample.prompt,
        )
        return

    sample.tokens = list(new_prompt_ids) + response_tokens
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = logprobs
    sample.response_length = len(response_tokens)
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    sample.metadata["prompt_only_shadow_cleaned"] = True
    sample.metadata["prompt_only_shadow_stats"] = stats
    _prompt_only_write_audit(
        args,
        sample,
        mode="cleaned",
        residual=False,
        stats=stats,
        n_trained=n_trained,
        resp_len=len(response_tokens),
        orig_prompt=orig_prompt,
        cleaned_prompt=sample.prompt,
    )


def _clean_trajectory_m1(sample: Sample, turns_record: list[dict], tokenizer, args) -> None:
    """Rebuild sample.{tokens,loss_mask,rollout_log_probs,response_length,response}
    so the trained sequence is the no-skill "shadow" of an oracle rollout."""
    orig_prompt = sample.prompt
    prompt_text = _m1_messages_text(orig_prompt)
    has_skill_prompt = bool(_M1_SKILLS_SECTION_RE.search(prompt_text)) or bool(SKILL_FILE_RE.search(prompt_text))
    has_skill_turn = any(
        t["kind"] == "assistant" and _m1_turn_skill_counts(t["text"])[0] > 0 for t in turns_record
    )
    if not has_skill_prompt and not has_skill_turn:
        # no-op (e.g. no-skill eval rollouts): finalize with original tokens.
        prompt_ids = _prepare_initial_inputs(sample, tokenizer)
        resp = sample.tokens[len(prompt_ids):] if len(sample.tokens) >= len(prompt_ids) else []
        sample.response = tokenizer.decode(resp, skip_special_tokens=False)
        sample.response_length = len(resp)
        if sample.status is None:
            sample.status = Sample.Status.COMPLETED
        return

    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = None

    # capture the ORIGINAL (with-skill) trajectory for audit/diagnosis before clean
    orig_response = "\n".join(f"[{t['kind']}] {t['text']}" for t in turns_record if t.get("text"))

    # 1) no-skill prompt + re-tokenize via the same path as the live rollout
    sample.prompt = _m1_strip_prompt_skills(orig_prompt)
    new_prompt_ids = _prepare_initial_inputs(sample, tokenizer)

    # 2) walk turns: drop skill-read turns + paired obs, scrub prose elsewhere
    resp_tokens: list[int] = []
    loss_mask: list[int] = []
    logprobs: list[float] = []
    kept_texts: list[str] = []
    # The rollout log stores one merged observation turn for an assistant turn,
    # even if the assistant made multiple skill calls in that turn.
    drop_obs = 0
    stats = {"dropped_skill_turns": 0, "dropped_obs": 0, "mixed_turns": 0, "reencoded_turns": 0}
    for t in turns_record:
        if t["kind"] == "assistant":
            n_skill, n_other = _m1_turn_skill_counts(t["text"])
            if n_skill > 0 and n_other == 0:
                stats["dropped_skill_turns"] += 1
                drop_obs += 1
                continue
            text = t["text"]
            if n_skill > 0:
                text = _m1_drop_skill_func_blocks(text)
                stats["mixed_turns"] += 1
                drop_obs += 1
            text = _m1_scrub_prose(text)
            if text != t["text"]:
                toks = tokenizer.encode(text, add_special_tokens=False)
                if im_end_id is not None and t["tokens"] and t["tokens"][-1] == im_end_id:
                    if not toks or toks[-1] != im_end_id:
                        toks = toks + [im_end_id]
                lps = [0.0] * len(toks)
                stats["reencoded_turns"] += 1
            else:
                toks = list(t["tokens"])
                lps = list(t["logprobs"])
            if not toks:
                continue
            resp_tokens.extend(toks)
            loss_mask.extend([1] * len(toks))
            logprobs.extend(lps)
            kept_texts.append(text)
        else:  # obs
            if drop_obs > 0:
                drop_obs -= 1
                stats["dropped_obs"] += 1
                continue
            resp_tokens.extend(t["tokens"])
            loss_mask.extend([0] * len(t["tokens"]))
            logprobs.extend([0.0] * len(t["tokens"]))
            kept_texts.append(t["text"])

    # 3) degeneracy / residual gate. A cleaned sample that is empty
    #    (response_length==0) or has zero trained tokens (sum(loss_mask)==0)
    #    deadlocks the CP/TP data broadcast (broadcast_object_list in
    #    compute_ref_log_prob) at train time — one rank diverges on the empty
    #    sequence and the collective hangs forever. Likewise a sample where
    #    skill exposure survived the clean would silently train no-skill-eval
    #    leakage. ANY such sample is marked ABORTED so sglang_rollout drops +
    #    refills it and it NEVER reaches training. This guard is load-bearing.
    full_text = _m1_messages_text(sample.prompt) + "\n" + "\n".join(kept_texts)
    residual = _m1_text_has_skill(full_text)
    n_trained = sum(loss_mask)
    degenerate = (len(resp_tokens) == 0) or (n_trained == 0)
    if residual or degenerate:
        sample.tokens = list(new_prompt_ids)
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.response_length = 0
        sample.response = ""
        _mark_infra_aborted(
            sample,
            reason="m1_residual_skill" if residual else "m1_degenerate_empty",
            category="m1_clean_drop",
            extra={
                "m1_stats": stats,
                "m1_residual": bool(residual),
                "m1_resp_len": len(resp_tokens),
                "m1_trained_tokens": int(n_trained),
            },
        )
        _m1_write_audit(
            args, sample,
            mode="aborted_residual" if residual else "aborted_degenerate",
            residual=residual, stats=stats, n_trained=n_trained, resp_len=len(resp_tokens),
            orig_response=orig_response, cleaned_response="\n".join(kept_texts),
            loss_mask_len=len(loss_mask),
        )
        return

    # 4) rebuild + invariants (valid cleaned no-skill trajectory)
    sample.tokens = list(new_prompt_ids) + resp_tokens
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = logprobs
    sample.response_length = len(resp_tokens)
    sample.response = tokenizer.decode(resp_tokens, skip_special_tokens=False)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    sample.metadata["m1_cleaned"] = True
    sample.metadata["m1_stats"] = stats
    _m1_write_audit(
        args, sample, mode="cleaned", residual=False, stats=stats,
        n_trained=n_trained, resp_len=len(resp_tokens),
        orig_response=orig_response, cleaned_response="\n".join(kept_texts),
        loss_mask_len=len(loss_mask),
    )


# ---------------------------------------------------------------------------
# SGLang HTTP step
# ---------------------------------------------------------------------------
async def _run_inference_step(url: str, tokens: list[int], sampling_params: dict):
    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    output = await post(url, payload)
    response_text = output["text"]
    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []
    finish_type = output["meta_info"]["finish_reason"]["type"]
    meta_info = output["meta_info"]
    return response_text, new_tokens, new_log_probs, finish_type, meta_info


async def _process_env_step(env, response_text: str, tokenizer, args, remaining_sec: float | None = None):
    # Run env.step in a thread executor so synchronous docker exec calls
    # inside ToolLayer.dispatch don't block the asyncio loop (was the
    # cause of "fake concurrency" — 70 rollouts but serialized through
    # subprocess.Popen).
    # remaining_sec: wallclock budget left for this rollout. If the inner
    # docker exec hangs, asyncio.wait_for cancels the awaitable and we raise
    # TimeoutError up to generate() which marks the sample TRUNCATED.
    step_coro = asyncio.to_thread(env.step, response_text)
    if remaining_sec is not None and remaining_sec > 0:
        result = await asyncio.wait_for(step_coro, timeout=remaining_sec)
    else:
        result = await step_coro
    if inspect.isawaitable(result):
        result = await result
    observation, done, info = result
    if done:
        return None, True, info, ""
    next_user_message = {
        "role": observation.get("role", "user"),
        "content": observation.get("obs_str", ""),
    }
    loop = asyncio.get_running_loop()
    obs_prompt_ids = await loop.run_in_executor(
        _ENCODE_EXECUTOR,
        _encode_observation_for_generation,
        tokenizer,
        next_user_message,
        args.apply_chat_template,
        args.apply_chat_template_kwargs,
    )
    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]
    return obs_prompt_ids, False, info, next_user_message["content"]


# ---------------------------------------------------------------------------
# Sample accumulation
# ---------------------------------------------------------------------------
def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _should_stop_on_finish(sample: Sample, finish_type: str) -> str | None:
    match finish_type:
        case "length":
            sample.status = Sample.Status.TRUNCATED
            return "finish_length"
        case "abort":
            sample.status = Sample.Status.ABORTED
            return "finish_abort"
    return None


def _update_budget(budget, consumed: int):
    return None if budget is None else budget - consumed


def _finalize_sample(sample: Sample, tokenizer, response_tokens):
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    return sample


def _mark_infra_aborted(
    sample: Sample,
    *,
    reason: str,
    category: str,
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    """Mark infrastructure failures as ABORTED with structured metadata.

    ABORTED samples are dropped/refilled by sglang_rollout.py and therefore do
    not enter GRPO training. The metadata is still persisted in debug artifacts
    and logs so setup/verifier/tool infra failures can be counted separately
    from model failures and length/context truncation.
    """
    sample.status = Sample.Status.ABORTED
    metadata = sample.metadata.setdefault("abort_info", {})
    sample.metadata["rollout_stop_reason"] = reason
    sample.metadata["rollout_abort_category"] = category
    sample.metadata["rollout_infra_failure"] = True
    metadata["reason"] = reason
    metadata["category"] = category
    if error:
        metadata["error"] = error
    if extra:
        metadata.update(extra)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    extra = _sample_extra_info(sample)
    task_id_for_slot = (extra or sample.metadata or {}).get("task_id", "?")
    async with _active_env_slot(task_id_for_slot):
        return await _generate_with_env_slot(args, sample, sampling_params)


async def _generate_with_env_slot(args: Any, sample: Sample, sampling_params) -> Sample:
    env_module = _load_env_module(getattr(args, "rollout_interaction_env_path", None))
    max_turns = int(getattr(args, "max_turns", 30) or 30)
    wallclock_cap = float(os.environ.get("UNIFIED_ROLLOUT_WALLCLOCK_CAP_SEC", _DEFAULT_WALLCLOCK_CAP_SEC))
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)
    sampling_params = sampling_params.copy()

    # M1 skill-free shadow: record per-turn (tokens, text, obs) so we can rebuild
    # a no-skill cleaned trajectory before handing the sample to training.
    m1_clean = _m1_enabled_for_sample(sample)
    prompt_only_clean = (not m1_clean) and _prompt_only_enabled_for_sample(sample)
    m1_turns: list[dict] | None = [] if m1_clean else None

    loop = asyncio.get_running_loop()
    prompt_ids = await loop.run_in_executor(
        _ENCODE_EXECUTOR, _prepare_initial_inputs, sample, state.tokenizer
    )

    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    response_tokens: list[int] = (
        sample.tokens[len(prompt_ids):] if len(sample.tokens) >= len(prompt_ids) else []
    )
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    budget = None
    if getattr(args, "rollout_max_context_len", None) is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sample.response_length > 0 and sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - sample.response_length

    rollout_traces = sample.metadata.setdefault("rollout_traces", [])
    stop_reason = None
    turns_executed = 0

    def _budget_exhausted() -> bool:
        return budget is not None and budget <= 0

    try:
        # Run env.reset in thread executor so sync docker run / docker exec
        # calls don't block the asyncio loop. This is what lets 70 concurrent
        # rollouts truly parallelize docker subprocess work.
        reset_result = await asyncio.to_thread(env.reset)
        # If env.reset returned a tuple (obs, info) with skipped=True flag,
        # the launcher failed and this sample should NOT participate in the
        # GRPO advantage (otherwise infra failures pollute the reward signal).
        if isinstance(reset_result, tuple) and len(reset_result) == 2 and isinstance(reset_result[1], dict) and reset_result[1].get("skipped"):
            reset_info = reset_result[1]
            _mark_infra_aborted(
                sample,
                reason=reset_info.get("error") or "env_setup_failed",
                category=reset_info.get("abort_category") or "setup_infra",
                error=reset_info.get("error_detail"),
                extra={
                    "setup_attempts": reset_info.get("setup_attempts"),
                    "setup_timeout_sec": reset_info.get("setup_timeout_sec"),
                },
            )
            return _finalize_sample(sample, state.tokenizer, response_tokens)
        if _budget_exhausted():
            sample.status = Sample.Status.TRUNCATED
            sample.metadata["rollout_stop_reason"] = "budget_exhausted"
            return _finalize_sample(sample, state.tokenizer, response_tokens)

        # Wallclock deadline starts after env.reset so we don't penalise slow
        # docker boots (which retry internally) against the agent's turn budget.
        deadline = time.time() + wallclock_cap

        cur_sampling_params = sampling_params
        for turn_idx in range(max_turns):
            turns_executed = turn_idx + 1
            if budget is not None:
                cur_sampling_params["max_new_tokens"] = budget

            remaining = deadline - time.time()
            if remaining <= 0:
                sample.status = Sample.Status.TRUNCATED
                stop_reason = stop_reason or "wallclock_cap"
                break

            t0 = time.time()
            response_token_start = len(response_tokens)
            response_text, new_tokens, new_logprobs, finish_type, _meta = await _run_inference_step(
                url, sample.tokens, cur_sampling_params
            )
            if budget is not None and len(new_tokens) > budget:
                # SGLang should respect max_new_tokens, but keep a hard guard
                # here because Megatron actor training OOMs on samples that
                # exceed rollout_max_context_len. Preserve prefix tokens and
                # truncate only the newly generated assistant span.
                keep = max(0, int(budget))
                new_tokens = new_tokens[:keep]
                new_logprobs = new_logprobs[:keep]
                sample.status = Sample.Status.TRUNCATED
                stop_reason = stop_reason or "budget_exhausted_generation"
            rollout_traces.append({
                "turn_index": turn_idx,
                "inference": {
                    "response_text": response_text,
                    "finish_type": finish_type,
                    "elapsed": round(time.time() - t0, 3),
                    "response_token_start": response_token_start,
                },
            })
            _append_to_sample(sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1)
            if m1_turns is not None:
                m1_turns.append({
                    "kind": "assistant",
                    "tokens": list(new_tokens),
                    "logprobs": list(new_logprobs),
                    "text": response_text,
                })
            budget = _update_budget(budget, len(new_tokens))

            if sample.status == Sample.Status.TRUNCATED and stop_reason == "budget_exhausted_generation":
                break

            stop = _should_stop_on_finish(sample, finish_type)
            if stop:
                stop_reason = stop
                break
            if _budget_exhausted():
                sample.status = Sample.Status.TRUNCATED
                stop_reason = stop_reason or "budget_exhausted"
                break

            t1 = time.time()
            step_remaining = deadline - time.time()
            if step_remaining <= 0:
                sample.status = Sample.Status.TRUNCATED
                stop_reason = stop_reason or "wallclock_cap"
                break
            try:
                obs_prompt_ids, done, info, obs_str = await _process_env_step(
                    env, response_text, state.tokenizer, args, remaining_sec=step_remaining,
                )
            except asyncio.TimeoutError:
                sample.status = Sample.Status.TRUNCATED
                stop_reason = stop_reason or "wallclock_step_timeout"
                rollout_traces[-1]["env_step"] = {
                    "done": False,
                    "info": {"error": "wallclock_step_timeout"},
                    "elapsed": round(time.time() - t1, 3),
                }
                logger.warning(
                    f"[{_sample_extra_info(sample).get('task_id', '?')}] "
                    f"env.step exceeded wallclock budget ({step_remaining:.1f}s); marking TRUNCATED"
                )
                break
            rollout_traces[-1]["env_step"] = {
                "done": done,
                "info": info,
                "elapsed": round(time.time() - t1, 3),
            }
            if os.environ.get("RELAX_SELECTOR_ACTION_CREDIT", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                from examples.agent_bench.selector_action_credit import record_assistant_turn

                record_assistant_turn(
                    sample,
                    response_text=response_text,
                    new_tokens=new_tokens,
                    response_token_start=response_token_start,
                    tokenizer=state.tokenizer,
                    turn_index=turn_idx,
                    dispatched_tool_call_count=(
                        int(info.get("tool_calls", 0)) if isinstance(info, dict) else 0
                    ),
                )
            if done:
                if isinstance(info, dict) and info.get("skipped"):
                    stop_reason = stop_reason or info.get("error", "env_step_skipped")
                    _mark_infra_aborted(
                        sample,
                        reason=stop_reason,
                        category=info.get("abort_category") or "verifier_infra",
                        error=info.get("error_detail") or info.get("error"),
                    )
                    break
                sample.status = Sample.Status.COMPLETED
                stop_reason = stop_reason or "env_done"
                # final_score has already been stashed on sample.metadata by
                # AgentBenchEnv.step(); nothing more to do here.
                break

            if budget is not None and len(obs_prompt_ids) > budget:
                # Tool observations can be very large (logs, pytest output,
                # repeated tracebacks). The old code appended the whole
                # observation then marked the sample truncated, which still
                # sent >40k-token samples to actor training and caused OOM.
                # Keep as much observation as fits, then terminate the sample.
                keep = max(0, int(budget))
                if keep > 0:
                    obs_prompt_ids = obs_prompt_ids[:keep]
                    obs_log_probs = [0.0] * len(obs_prompt_ids)
                    _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
                    if m1_turns is not None:
                        m1_turns.append({"kind": "obs", "tokens": list(obs_prompt_ids), "text": obs_str})
                    budget = _update_budget(budget, len(obs_prompt_ids))
                sample.status = Sample.Status.TRUNCATED
                stop_reason = stop_reason or "budget_exhausted_observation"
                break

            obs_log_probs = [0.0] * len(obs_prompt_ids)
            _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
            if m1_turns is not None:
                m1_turns.append({"kind": "obs", "tokens": list(obs_prompt_ids), "text": obs_str})
            budget = _update_budget(budget, len(obs_prompt_ids))

            if _budget_exhausted():
                sample.status = Sample.Status.TRUNCATED
                stop_reason = stop_reason or "budget_exhausted"
                break
            if turn_idx + 1 >= max_turns:
                sample.status = Sample.Status.COMPLETED
                stop_reason = stop_reason or "max_turns"
                break

        sample.metadata["rollout_turns"] = turns_executed
        sample.metadata["rollout_stop_reason"] = stop_reason or "completed"
        if m1_turns is not None:
            try:
                _clean_trajectory_m1(sample, m1_turns, state.tokenizer, args)
            except Exception as exc:  # never let cleaning crash a rollout
                logger.warning(
                    "[%s] M1 clean failed (%s); falling back to uncleaned sample",
                    _sample_extra_info(sample).get("task_id", "?"),
                    exc,
                )
                _finalize_sample(sample, state.tokenizer, response_tokens)
            return sample
        if prompt_only_clean:
            try:
                _clean_prompt_only_shadow(sample, response_tokens, state.tokenizer, args)
            except Exception as exc:
                logger.warning(
                    "[%s] prompt-only shadow clean failed (%s); aborting sample",
                    _sample_extra_info(sample).get("task_id", "?"),
                    exc,
                )
                _mark_infra_aborted(
                    sample,
                    reason="prompt_only_clean_exception",
                    category="prompt_only_shadow_clean_drop",
                    error=str(exc),
                )
                _finalize_sample(sample, state.tokenizer, response_tokens)
            return sample
        _finalize_sample(sample, state.tokenizer, response_tokens)
        if _opsd_mode_enabled() and isinstance(sample.metadata, dict) and "opsd_teacher_prompt" in sample.metadata:
            # Rendering a ~70k-char oracle prompt through the chat template is
            # CPU-heavy; keep it off the rollout event loop like the main
            # prompt render above.
            await loop.run_in_executor(_ENCODE_EXECUTOR, _opsd_attach_teacher_prompt_ids, sample, state.tokenizer)
        return sample
    finally:
        try:
            # Async-wrap close too: launcher.teardown does docker rm -f which
            # is sync subprocess and would block the loop in cleanup phase.
            await asyncio.to_thread(env.close)
        except Exception:
            pass

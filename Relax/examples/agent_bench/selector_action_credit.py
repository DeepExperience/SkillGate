"""Token-local selector credit for normal mixed-skill GRPO rollouts."""

from __future__ import annotations

import json
import math
import os
import re
from statistics import mean, stdev
from typing import Any, Sequence

from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.types import Sample


SCHEMA = "selector_action_credit_v1"
EXPECTED_GROUP_SIZE = 8
EXPECTED_SLATE_SIZE = 16
EXPECTED_UPDATE_KIND = "mixed_bonus_compare_grpo"
EXPECTED_EVAL_UPDATE_KIND = "mixed_bonus_compare_eval"

_JSON_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
_XML_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_SKILL_PATH_RE = re.compile(
    r"(?:/root|~)?/\.claude/skills/([A-Za-z0-9_.-]+)/(?:SKILL|README)\.md",
    re.IGNORECASE,
)
_EXEC_READ_RE = re.compile(
    r"\b(cat|sed|awk|grep|head|tail|less|more|python3?|perl|ruby|node)\b",
    re.IGNORECASE,
)


def enabled() -> bool:
    return os.environ.get("RELAX_SELECTOR_ACTION_CREDIT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _plain_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set, frozenset)):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _extra_info(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    extra = metadata.get("extra_info")
    return extra if isinstance(extra, dict) else metadata


def _categories(sample: Sample) -> dict[str, Any]:
    extra = _extra_info(sample)
    kind = str(extra.get("update_kind") or extra.get("hybrid_update_kind") or "").strip().lower()
    if kind not in {EXPECTED_UPDATE_KIND, EXPECTED_EVAL_UPDATE_KIND}:
        raise ValueError(
            "selector credit expected mixed-slate update_kind in "
            f"{{{EXPECTED_UPDATE_KIND!r}, {EXPECTED_EVAL_UPDATE_KIND!r}}}, got {kind!r}"
        )
    retrieval = _plain_list(extra.get("retrieval_skills_top_n"))
    misleading = _plain_list(extra.get("slate_misleading_names"))
    relevant = _plain_list(extra.get("slate_relevant_names"))
    irrelevant = _plain_list(extra.get("slate_irrelevant_names"))
    oracle = str(extra.get("slate_gold_name") or "").strip()
    try:
        contains_gold = float(extra.get("slate_contains_gold") or 0.0) == 1.0
    except (TypeError, ValueError):
        contains_gold = False
    if (
        not contains_gold
        or not oracle
        or len(retrieval) != EXPECTED_SLATE_SIZE
        or len(set(retrieval)) != EXPECTED_SLATE_SIZE
        or oracle not in retrieval
        or len(misleading) != 5
        or len(relevant) != 5
        or len(irrelevant) != 5
    ):
        raise ValueError(
            "selector credit requires one-gold 16-skill slate with 5/5/5 controls: "
            f"gold={oracle!r} retrieval={len(retrieval)}/{len(set(retrieval))} "
            f"misleading={len(misleading)} relevant={len(relevant)} irrelevant={len(irrelevant)}"
        )
    return {
        "oracle": oracle,
        "retrieval": set(retrieval),
        "misleading": set(misleading),
        "relevant": set(relevant),
        "irrelevant": set(irrelevant),
    }


def _normalize_call(name: Any, arguments: Any) -> tuple[str, Any]:
    name = str(name or "").strip()
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            pass
    return name, arguments


def _located_tool_calls(text: str) -> list[dict[str, Any]]:
    """Mirror the canonical JSON-first OpenClaw parser while retaining spans."""

    json_calls: list[dict[str, Any]] = []
    for match in list(_JSON_TOOL_CALL_RE.finditer(text))[:5]:
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        json_calls.append(
            {
                "name": parsed.get("name", ""),
                "arguments": parsed.get("arguments", {}),
                "call_char_span": list(match.span()),
                "raw": match.group(0),
            }
        )
    if json_calls:
        return json_calls

    xml_calls: list[dict[str, Any]] = []
    for match in list(_XML_TOOL_CALL_RE.finditer(text))[:5]:
        arguments: dict[str, Any] = {}
        for key, value in _XML_PARAM_RE.findall(match.group(2)):
            value = value.strip()
            try:
                arguments[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                arguments[key] = value
        xml_calls.append(
            {
                "name": match.group(1),
                "arguments": arguments,
                "call_char_span": list(match.span()),
                "raw": match.group(0),
            }
        )
    return xml_calls


def _canonical_tool_calls(text: str) -> list[tuple[str, Any]]:
    from unified_runner.agent_loop import UnifiedAgentLoop  # type: ignore

    normalized: list[tuple[str, Any]] = []
    for call in UnifiedAgentLoop._parse_tool_calls_from_content(text):
        function = call.get("function") or {}
        normalized.append(_normalize_call(function.get("name"), function.get("arguments")))
    return normalized


def _call_signature(calls: Sequence[dict[str, Any]]) -> list[tuple[str, Any]]:
    return [_normalize_call(call.get("name"), call.get("arguments")) for call in calls]


def _category(skill_name: str, categories: dict[str, Any]) -> str:
    if skill_name == categories["oracle"]:
        return "oracle"
    if skill_name in categories["misleading"]:
        return "misleading"
    if skill_name in categories["relevant"]:
        return "relevant"
    if skill_name in categories["irrelevant"]:
        return "irrelevant"
    return "unadvertised"


def _overlap_token_indices(offsets: Sequence[Sequence[int]], start: int, end: int) -> list[int]:
    return [index for index, (left, right) in enumerate(offsets) if left < end and right > start]


def record_assistant_turn(
    sample: Sample,
    *,
    response_text: str,
    new_tokens: Sequence[int],
    response_token_start: int,
    tokenizer: Any,
    turn_index: int,
    dispatched_tool_call_count: int,
) -> None:
    """Record ordered, actually-dispatched skill reads and exact response-token spans."""

    if not enabled():
        return
    sample.metadata = sample.metadata or {}
    state = sample.metadata.setdefault(
        "selector_action_credit",
        {
            "schema": SCHEMA,
            "actions": [],
            "turns_checked": 0,
            "alignment_mismatch": 0,
            "parse_dispatch_mismatch": 0,
            "span_mismatch": 0,
        },
    )
    state["turns_checked"] = int(state.get("turns_checked", 0)) + 1

    located_calls = _located_tool_calls(response_text)
    canonical_calls = _canonical_tool_calls(response_text)
    if _call_signature(located_calls) != canonical_calls or len(located_calls) != int(dispatched_tool_call_count):
        state["parse_dispatch_mismatch"] = int(state.get("parse_dispatch_mismatch", 0)) + 1
        return

    categories = _categories(sample)
    candidate_actions: list[dict[str, Any]] = []
    for call_index, call in enumerate(located_calls):
        tool_name = str(call["name"] or "").strip().lower().rsplit(".", 1)[-1]
        arguments_text = json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True)
        if tool_name == "read":
            pass
        elif tool_name == "exec":
            if not _EXEC_READ_RE.search(arguments_text):
                continue
        else:
            continue

        call_start = int(call["call_char_span"][0])
        path_matches = list(_SKILL_PATH_RE.finditer(call["raw"]))
        if "/.claude/skills/" in arguments_text and not path_matches:
            state["span_mismatch"] = int(state.get("span_mismatch", 0)) + 1
            continue
        for path_index, path_match in enumerate(path_matches):
            identity_start, identity_end = path_match.span(1)
            skill_name = path_match.group(1)
            candidate_actions.append(
                {
                    "turn_index": int(turn_index),
                    "call_index": int(call_index),
                    "path_index": int(path_index),
                    "tool_name": tool_name,
                    "skill_name": skill_name,
                    "category": _category(skill_name, categories),
                    "advertised": 1.0 if skill_name in categories["retrieval"] else 0.0,
                    "call_char_span": list(call["call_char_span"]),
                    "identity_char_span": [call_start + identity_start, call_start + identity_end],
                }
            )

    if not candidate_actions:
        return

    try:
        encoded = tokenizer(
            response_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
    except Exception:
        state["alignment_mismatch"] = int(state.get("alignment_mismatch", 0)) + 1
        return
    generated_ids = list(new_tokens)
    trailing_ids = generated_ids[len(token_ids) :]
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if (
        token_ids != generated_ids[: len(token_ids)]
        or len(offsets) != len(token_ids)
        or any(token_id not in special_ids for token_id in trailing_ids)
    ):
        state["alignment_mismatch"] = int(state.get("alignment_mismatch", 0)) + 1
        return

    for action in candidate_actions:
        call_indices = _overlap_token_indices(offsets, *action["call_char_span"])
        if (
            trailing_ids
            and int(action["call_char_span"][1]) == len(response_text.rstrip())
        ):
            call_indices.extend(range(len(token_ids), len(generated_ids)))
        identity_indices = _overlap_token_indices(offsets, *action["identity_char_span"])
        if not call_indices or not identity_indices or any(index not in call_indices for index in identity_indices):
            state["span_mismatch"] = int(state.get("span_mismatch", 0)) + 1
            continue
        action["call_token_indices"] = [int(response_token_start + index) for index in call_indices]
        action["identity_token_indices"] = [int(response_token_start + index) for index in identity_indices]
        state["actions"].append(action)


def _selector_state(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    state = metadata.get("selector_action_credit")
    if not isinstance(state, dict):
        state = {
            "schema": SCHEMA,
            "actions": [],
            "turns_checked": 0,
            "alignment_mismatch": 0,
            "parse_dispatch_mismatch": 0,
            "span_mismatch": 0,
        }
        metadata["selector_action_credit"] = state
        sample.metadata = metadata
    return state


def _raw_score(sample: Sample, args: Any) -> float:
    if isinstance(sample.reward, dict) and "raw_score" in sample.reward:
        value = sample.reward["raw_score"]
    else:
        value = sample.get_reward_value(args)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"selector credit requires finite raw task score, got {value!r}")
    return result


def _annotate_reward(sample: Sample, values: dict[str, float]) -> None:
    if isinstance(sample.reward, dict):
        sample.reward.update(values)


def sample_behavior_metrics(sample: Sample) -> dict[str, float]:
    """Return trajectory-local selector diagnostics for both train and eval."""

    state = _selector_state(sample)
    actions = list(state.get("actions") or [])
    categories = [str(action.get("category") or "unadvertised") for action in actions]
    first_category = categories[0] if categories else "none"
    oracle_only = bool(categories and all(category == "oracle" for category in categories))
    misleading_only = bool(categories and all(category == "misleading" for category in categories))
    return {
        "selector_attributed_action_count": float(len(actions)),
        "selector_trajectory_action_count": float(len(actions)),
        "selector_first_read_oracle": 1.0 if first_category == "oracle" else 0.0,
        "selector_oracle_exposure": 1.0 if "oracle" in categories else 0.0,
        "selector_oracle_only": 1.0 if oracle_only else 0.0,
        "selector_misleading_exposure": 1.0 if "misleading" in categories else 0.0,
        "selector_misleading_only": 1.0 if misleading_only else 0.0,
        "selector_no_read": 1.0 if not actions else 0.0,
        "selector_multi_read": 1.0 if len(actions) > 1 else 0.0,
        "selector_alignment_mismatch": float(state.get("alignment_mismatch", 0) or 0),
        "selector_parse_dispatch_mismatch": float(state.get("parse_dispatch_mismatch", 0) or 0),
        "selector_span_mismatch": float(state.get("span_mismatch", 0) or 0),
    }


def annotate_group_selector_advantages(group: Sequence[Sample]) -> dict[str, float]:
    if len(group) != EXPECTED_GROUP_SIZE:
        raise ValueError(f"selector credit requires group size {EXPECTED_GROUP_SIZE}, got {len(group)}")

    oracle = _categories(group[0])["oracle"]
    actions: list[dict[str, Any]] = []
    per_sample_actions: list[list[dict[str, Any]]] = []
    for sample in group:
        categories = _categories(sample)
        if categories["oracle"] != oracle:
            raise ValueError("selector credit group has inconsistent oracle labels")
        state = _selector_state(sample)
        if any(int(state.get(key, 0)) for key in ("alignment_mismatch", "parse_dispatch_mismatch", "span_mismatch")):
            raise ValueError("selector action attribution mismatch")
        sample_actions = list(state.get("actions") or [])
        seen_oracle = False
        for action in sample_actions:
            is_first_oracle = action.get("skill_name") == oracle and not seen_oracle
            if action.get("skill_name") == oracle:
                seen_oracle = True
            action["utility"] = 1.0 if is_first_oracle else 0.0
        per_sample_actions.append(sample_actions)
        actions.extend(sample_actions)

    utilities = [float(action["utility"]) for action in actions]
    baseline = float(mean(utilities)) if utilities else 0.0
    active = bool(actions and any(value > 0 for value in utilities) and any(value <= 0 for value in utilities))
    for action in actions:
        action["selector_advantage"] = float(action["utility"] - baseline) if active else 0.0
    zero_mean_error = abs(sum(float(action["selector_advantage"]) for action in actions))
    selector_advantages = [float(action["selector_advantage"]) for action in actions]
    advantage_mean = float(mean(selector_advantages)) if selector_advantages else 0.0
    advantage_min = min(selector_advantages, default=0.0)
    advantage_max = max(selector_advantages, default=0.0)

    oracle_actions = sum(1 for action in actions if float(action["utility"]) > 0)
    nonoracle_actions = len(actions) - oracle_actions
    for sample, sample_actions in zip(group, per_sample_actions, strict=True):
        state = _selector_state(sample)
        state["group_action_baseline"] = baseline
        state["group_selector_active"] = 1.0 if active else 0.0
        state["group_action_count"] = len(actions)
        state["group_oracle_action_count"] = oracle_actions
        state["group_nonoracle_action_count"] = nonoracle_actions
        state["group_weighted_zero_mean_error"] = zero_mean_error
        _annotate_reward(
            sample,
            {
                **sample_behavior_metrics(sample),
                "selector_active_group": 1.0 if active else 0.0,
                "selector_group_action_count": float(len(actions)),
                "selector_group_oracle_action_count": float(oracle_actions),
                "selector_group_nonoracle_action_count": float(nonoracle_actions),
                "selector_group_action_baseline": baseline,
                "selector_group_zero_mean_error": zero_mean_error,
                "selector_group_advantage_mean": advantage_mean,
                "selector_group_advantage_min": advantage_min,
                "selector_group_advantage_max": advantage_max,
                "selector_group_no_oracle_action": 1.0 if oracle_actions == 0 else 0.0,
            },
        )
    return {
        "active": 1.0 if active else 0.0,
        "actions": float(len(actions)),
        "oracle_actions": float(oracle_actions),
        "nonoracle_actions": float(nonoracle_actions),
        "baseline": baseline,
        "zero_mean_error": zero_mean_error,
    }


def _flat_samples(samples: list[Sample] | list[list[Sample]]) -> list[Sample]:
    if samples and isinstance(samples[0], list):
        return [sample for group in samples for sample in group]
    return list(samples)  # type: ignore[arg-type]


def _groups(samples: Sequence[Sample], group_size: int) -> list[list[Sample]]:
    if len(samples) % group_size:
        raise ValueError(f"selector sample count {len(samples)} is not divisible by {group_size}")
    return [list(samples[start : start + group_size]) for start in range(0, len(samples), group_size)]


def keep_raw_task_reward_nonzero_std(args: Any, samples: list[Sample], **_: Any) -> DynamicFilterOutput:
    if not enabled():
        raise RuntimeError("selector filter selected while RELAX_SELECTOR_ACTION_CREDIT is disabled")
    if any(sample.reward is None for sample in samples):
        return DynamicFilterOutput(keep=False, reason="selector_missing_reward")
    if any(
        str(_extra_info(sample).get("update_kind") or _extra_info(sample).get("hybrid_update_kind") or "")
        .strip()
        .lower()
        != EXPECTED_UPDATE_KIND
        for sample in samples
    ):
        return DynamicFilterOutput(keep=False, reason="selector_non_train_update_kind")
    try:
        stats = annotate_group_selector_advantages(samples)
        raw_scores = [_raw_score(sample, args) for sample in samples]
    except (TypeError, ValueError) as error:
        return DynamicFilterOutput(keep=False, reason=f"selector_attribution_error_{type(error).__name__}")
    if max(raw_scores) - min(raw_scores) <= 1e-12:
        suffix = "active" if stats["active"] else "inactive"
        return DynamicFilterOutput(keep=False, reason=f"zero_std_raw_task_selector_{suffix}")
    return DynamicFilterOutput(keep=True, reason=None)


def post_process_rewards(
    args: Any, samples: list[Sample] | list[list[Sample]]
) -> tuple[list[float], list[float]]:
    """Preserve ordinary factual GRPO rewards and annotate action-local credit."""

    if not enabled():
        raise RuntimeError("selector reward postprocess selected while feature is disabled")
    flat = _flat_samples(samples)
    if any(
        str(_extra_info(sample).get("update_kind") or _extra_info(sample).get("hybrid_update_kind") or "")
        .strip()
        .lower()
        != EXPECTED_UPDATE_KIND
        for sample in flat
    ):
        raise ValueError(f"selector reward postprocess requires update_kind={EXPECTED_UPDATE_KIND!r}")
    group_size = int(getattr(args, "n_samples_per_prompt", EXPECTED_GROUP_SIZE) or EXPECTED_GROUP_SIZE)
    raw_rewards: list[float] = []
    processed: list[float] = []
    for group in _groups(flat, group_size):
        annotate_group_selector_advantages(group)
        scores = [_raw_score(sample, args) for sample in group]
        raw_rewards.extend(scores)
        if getattr(args, "rewards_normalization", True):
            centered = [float(score - mean(scores)) for score in scores]
            if bool(getattr(args, "grpo_std_normalization", True)) and len(scores) > 1:
                scale = float(stdev(scores)) + 1e-6
                centered = [value / scale for value in centered]
            processed.extend(centered)
        else:
            processed.extend(scores)
    return raw_rewards, processed


def build_train_fields(samples: Sequence[Sample], base_loss_masks: Sequence[Sequence[float]]) -> dict[str, list]:
    """Build globally normalized task and selector token weights."""

    if len(samples) != len(base_loss_masks):
        raise ValueError("selector train-field sample/mask count mismatch")

    task_masks: list[list[float]] = []
    selector_masks: list[list[float]] = []
    selector_advantages: list[list[float]] = []
    active_actions = 0
    base_total = 0.0
    task_total = 0.0

    for sample, base_mask_value in zip(samples, base_loss_masks, strict=True):
        base_mask = [float(value) for value in base_mask_value]
        response_length = int(sample.response_length or 0)
        if len(base_mask) != response_length:
            raise ValueError("selector base mask length mismatch")
        task_mask = list(base_mask)
        state = _selector_state(sample)
        for action in state.get("actions") or []:
            for index in action.get("call_token_indices") or []:
                if index < 0 or index >= response_length or base_mask[index] <= 0:
                    raise ValueError(f"selector call token outside assistant loss mask: {index}/{response_length}")
                task_mask[index] = 0.0
            if abs(float(action.get("selector_advantage", 0.0))) > 0:
                active_actions += 1
        base_total += sum(base_mask)
        task_total += sum(task_mask)
        task_masks.append(task_mask)

    if base_total <= 0 or task_total <= 0:
        raise ValueError(f"selector requires nonzero base/task tokens: base={base_total} task={task_total}")
    task_scale = base_total / task_total
    task_weights = [[value * task_scale for value in mask] for mask in task_masks]
    selector_scale = base_total / active_actions if active_actions else 0.0

    for sample, base_mask, task_mask in zip(samples, base_loss_masks, task_masks, strict=True):
        response_length = int(sample.response_length or 0)
        selector_mask = [0.0] * response_length
        selector_advantage = [0.0] * response_length
        for action in _selector_state(sample).get("actions") or []:
            advantage = float(action.get("selector_advantage", 0.0))
            if advantage == 0.0:
                continue
            indices = [int(index) for index in action.get("identity_token_indices") or []]
            if not indices:
                raise ValueError("active selector action has no identity tokens")
            per_token_weight = selector_scale / len(indices)
            for index in indices:
                if index < 0 or index >= response_length or float(base_mask[index]) <= 0:
                    raise ValueError(f"selector identity token outside assistant loss mask: {index}/{response_length}")
                if float(task_mask[index]) != 0.0 or selector_mask[index] != 0.0:
                    raise ValueError("selector/task overlap or duplicate identity token")
                selector_mask[index] = per_token_weight
                selector_advantage[index] = advantage
        selector_masks.append(selector_mask)
        selector_advantages.append(selector_advantage)

    if not math.isclose(sum(map(sum, task_weights)), base_total, rel_tol=1e-9, abs_tol=1e-3):
        raise AssertionError("selector task-weight normalization drift")
    expected_selector_total = base_total if active_actions else 0.0
    if not math.isclose(
        sum(map(sum, selector_masks)), expected_selector_total, rel_tol=1e-9, abs_tol=1e-3
    ):
        raise AssertionError("selector action-weight normalization drift")
    if any(
        task > 0 and selector > 0
        for task_mask, selector_mask in zip(task_masks, selector_masks, strict=True)
        for task, selector in zip(task_mask, selector_mask, strict=True)
    ):
        raise AssertionError("selector/task token support overlap")

    return {
        "selector_task_loss_weights": task_weights,
        "selector_action_loss_weights": selector_masks,
        "selector_action_advantages": selector_advantages,
    }


__all__ = [
    "SCHEMA",
    "annotate_group_selector_advantages",
    "build_train_fields",
    "enabled",
    "keep_raw_task_reward_nonzero_std",
    "post_process_rewards",
    "record_assistant_turn",
    "sample_behavior_metrics",
]

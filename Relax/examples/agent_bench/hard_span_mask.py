"""Hard-span oracle BC token masks.

This variant does not use teacher/student log-prob gaps.  It supervises only
tokens that are directly useful on the deployment distribution:

- executable tool-call/action spans,
- action-grounded reasoning immediately before an action,
- the final assistant answer after tool use.

It then subtracts privileged skill disclosure, private think blocks, tool
observations, malformed traces, and any action payload that itself contains
skill prose.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from examples.agent_bench.action_span_mask import find_action_spans


IM_END = "<|im_end|>"
ASSISTANT_START_RE = re.compile(r"<\|im_start\|>assistant\n?", re.IGNORECASE)
TOOL_RESPONSE_RE = re.compile(
    r"(?:<\|im_start\|>user\s*)?<tool_response\b[^>]*>.*?</tool_response>(?:<\|im_end\|>)?",
    re.IGNORECASE | re.DOTALL,
)
SKILL_BLOCK_RE = re.compile(r"<skill_reasoning\b[^>]*>.*?</skill_reasoning>", re.IGNORECASE | re.DOTALL)
SKILL_REASONING_TAG_RE = re.compile(r"</?skill_reasoning\b[^>]*>", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
PRELOADED_SKILL_XML_RE = re.compile(
    r"<preloaded_(?:oracle|top1)_skill\b[^>]*>.*?</preloaded_(?:oracle|top1)_skill>",
    re.IGNORECASE | re.DOTALL,
)
SKILL_LINE_PAT = re.compile(
    r"(skill_reasoning|retrieved skill|preloaded oracle|preloaded skill|oracle skill|skill id|"
    r"<preloaded_(?:oracle|top1)_skill|</preloaded_(?:oracle|top1)_skill|available_skills|"
    r"skill data|SKILL\.md|\.claude/skills|read (?:the )?skill|skill file|skill library|"
    r"provided skill|according to (?:the|this) skill|the skill (?:says|contains|provides|mentions|indicates)|"
    r"\bskills?\b(?=[^a-z]*(?:file|library|directory|entry|describ|provid|retriev)))",
    re.IGNORECASE,
)
USEFUL_REASONING_PAT = re.compile(
    r"(bug|error|exception|traceback|fail|failure|root cause|fix|patch|modify|change|implement|"
    r"test|pytest|unittest|assert|verify|validate|inspect|locat|open|read|write|edit|search|"
    r"file|path|line|function|class|method|repository|code|command|grep|rg|sed|cat|ls|python|"
    r"bash|diff|json|schema|parse|output|result|run|execute|check)",
    re.IGNORECASE,
)
V2_USEFUL_REASONING_PAT = re.compile(
    r"(bug|error|exception|traceback|fail|failure|root cause|cause|because|therefore|so that|"
    r"fix|patch|modify|change|implement|test|pytest|unittest|assert|verify|validate|confirm|"
    r"inspect|diagnos|investigat|locat|open|read|write|edit|search|compare|calculate|"
    r"file|path|line|function|class|method|repository|code|command|grep|rg|sed|cat|ls|python|"
    r"bash|diff|json|schema|parse|output|result|run|execute|check|plan|approach|hypothes|"
    r"need to|should|must|missing|required|constraint|issue|problem|likely)",
    re.IGNORECASE,
)
V3_USEFUL_REASONING_PAT = re.compile(
    r"(bug|error|exception|traceback|fail|failure|root cause|cause|because|therefore|so that|"
    r"fix|patch|modify|change|implement|test|pytest|unittest|assert|verify|validate|confirm|"
    r"inspect|diagnos|investigat|locat|open|read|write|edit|search|compare|calculate|"
    r"file|path|line|function|class|method|repository|code|command|grep|rg|sed|cat|ls|python|"
    r"bash|diff|json|schema|parse|output|result|run|execute|check|plan|approach|hypothes|"
    r"need to|should|must|missing|required|constraint|issue|problem|likely|instead|before|after|"
    r"dependency|config|argument|option|parameter|input|case|edge|branch|condition|state|"
    r"observ|found|shows|indicates|suggests|means|expected|actual|pass|success|correct)",
    re.IGNORECASE,
)
V3_SKILL_DERIVED_PAT = re.compile(
    r"(skill_reasoning|retrieved(?:_|\s)+skill|preloaded(?:_|\s)+(?:oracle|top1|skill)|"
    r"oracle(?:_|\s)+skill|skill(?:_|\s)*id|preloaded_oracle_skill|preloaded_top1_skill|"
    r"available_skills|skill data|SKILL\.md|\.claude/skills|\bskills?\b|"
    r"read(?:ing)? (?:the |this |a )?skill|review(?:ing)? (?:the |this |a )?skill|"
    r"rely(?:ing)? on (?:the |this |a )?skill|use (?:it|the skill|this skill) directly|"
    r"using (?:this |the |a |pre-existing |provided )?skill|adapt (?:it|the skill|this skill)|"
    r"(?:the|this|provided|pre-existing) skill (?:says|contains|provides|mentions|indicates|addresses|details)|"
    r"skill(?:'s)? provided script|provided script and then adapt|verified syntax)",
    re.IGNORECASE,
)
V4_PRIVILEGED_PATH_PAT = re.compile(
    r"(/root/\.cache/retrieval/context|/root/retrieve_skill\b|/preread_files\b|"
    r"/root/solutions\b|/root/seta_claude_skip\b|"
    r"oracle_top1_skills|skill_libraries|\.claude/skills|SKILL\.md)",
    re.IGNORECASE,
)
V4_PRIVILEGED_SOURCE_PAT = re.compile(
    r"(?:\boracle\b|preloaded(?:_|\s)+(?:oracle|top1|skill)|retrieved(?:_|\s)+skill|"
    r"\bretriever\b|retrieval (?:context|ground truth)|retrieved files?|"
    r"previous context from (?:the )?retriever|from (?:the )?retrieval|"
    r"golden ground truth|ground truth from (?:the )?retrieval|"
    r"exact answer|exact solution|reference solution|solution reference|benchmark reference|"
    r"exact values specified in (?:the )?benchmark|known working|historical passing|"
    r"solution from (?:the )?context|exact (?:answer|solution) from (?:the )?context|"
    r"provided script|ready-to-run|one-pass workflow)",
    re.IGNORECASE,
)
V4_LINKED_ACTION_COPY_PAT = re.compile(
    r"(?:\boracle(?:'s)? working approach|oracle skill|preloaded(?:_|\s)+(?:oracle|top1|skill)|"
    r"proven (?:python )?script from (?:the )?oracle skill|provided script|"
    r"tested and verified for (?:this )?exact benchmark|exact answer|exact solution|reference solution|"
    r"solution reference|benchmark reference solution|known working payload|historical passing run|"
    r"exact values specified in (?:the )?benchmark|solution from (?:the )?context|"
    r"exact (?:answer|solution) from (?:the )?context|previous context from (?:the )?retriever|"
    r"from (?:the )?retrieval|golden ground truth|ground truth from (?:the )?retrieval)",
    re.IGNORECASE,
)
V4_DIRECT_SKILL_TAIL_PAT = re.compile(
    r"(?:"
    r"use (?:this |the |it )?skill directly|directly (?:use|apply|adopt|write|implement|execute)|"
    r"complete,?\s*self-contained solution|complete implementation|ready-to-(?:use|run)|"
    r"provided script|full solution|proven [^.\n]{0,80}solution|"
    r"known working [^.\n]{0,80}(?:payload|solution|script)|"
    r"exact required outputs?|exact CSV output expected|exact code for|exact commands? needed"
    r")",
    re.IGNORECASE,
)
V4_REASONING_FORBIDDEN_PAT = re.compile(
    rf"(?:{V3_SKILL_DERIVED_PAT.pattern}|{V4_PRIVILEGED_PATH_PAT.pattern}|{V4_PRIVILEGED_SOURCE_PAT.pattern})",
    re.IGNORECASE,
)
V3_GENERIC_FILLER_PAT = re.compile(
    r"^\s*(?:okay|ok|sure|great|now|next|first|second|third|finally|overall|"
    r"let'?s|i(?:'ll| will)|we(?:'ll| will)|i need to|we need to|"
    r"the task is to|i(?:'m| am) going to|we(?:'re| are) going to)"
    r"[\s,.:;!-]*$",
    re.IGNORECASE,
)
V3_MARKUP_OR_CONTROL_PAT = re.compile(
    r"^\s*(?:<\|im_(?:start|end)\|>|</?tool_call\b[^>]*>|</?function\b[^>]*>|"
    r"</?tool_response\b[^>]*>|</?think\b[^>]*>|</?skill_reasoning\b[^>]*>|"
    r"</?reasoning\b[^>]*>)\s*$",
    re.IGNORECASE,
)
MALFORMED_TAGS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"<tool_call\b", re.IGNORECASE), "</tool_call>"),
    (re.compile(r"<function=", re.IGNORECASE), "</function>"),
    (re.compile(r"<skill_reasoning\b", re.IGNORECASE), "</skill_reasoning>"),
    (re.compile(r"<think\b", re.IGNORECASE), "</think>"),
    (re.compile(r"<tool_response\b", re.IGNORECASE), "</tool_response>"),
)


def build_hard_span_token_mask(
    response: str,
    *,
    response_token_count: int,
    tokenizer: Any,
    mode: str = "tool_call",
    base_loss_mask: Sequence[int | bool] | None = None,
    sample_status: str | None = None,
    reasoning_max_chars: int = 4096,
    final_max_chars: int = 4096,
    max_response_tokens: int = 0,
    keep_final: bool = True,
    require_useful_reasoning: bool = True,
    version: str = "v1",
) -> tuple[list[bool], dict[str, float | str]]:
    """Return hard eligible-token mask and numeric audit stats.

    ``base_loss_mask`` is applied before counting stats so the reported token
    counts match what the loss can actually supervise.
    """

    n_tokens = max(int(response_token_count or 0), 0)
    hard_span_version = _normalize_version(version)
    use_v2 = hard_span_version == "v2"
    use_v3 = hard_span_version == "v3"
    use_v4 = hard_span_version == "v4"
    use_v3_or_later = use_v3 or use_v4
    use_v2_or_later = use_v2 or use_v3_or_later
    base = _base_mask(base_loss_mask, n_tokens)
    empty = [False] * n_tokens

    status = (sample_status or "completed").strip().lower()
    drop_reason = ""
    if n_tokens <= 0:
        drop_reason = "empty"
    elif status not in {"", "completed", "truncated"}:
        drop_reason = status

    if drop_reason:
        stats = _stats(
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            n_tokens,
            drop_reason=drop_reason,
            action_span_count=0,
            reasoning_span_count=0,
            final_span_count=0,
            contaminated_action_span_count=0,
            skill_reasoning_span_count=0,
            version=hard_span_version,
        )
        return empty, stats

    overlong = max_response_tokens > 0 and n_tokens > max_response_tokens
    malformed = _has_malformed_tags(response)
    tail_unsafe = status == "truncated" or overlong or malformed
    if status == "truncated":
        drop_reason = "tail_truncated"
    elif overlong:
        drop_reason = "tail_overlong"
    elif malformed:
        drop_reason = "tail_malformed"

    tool_response_spans = _regex_spans(TOOL_RESPONSE_RE, response)
    malformed_tail_spans = _tail_open_spans(response)
    skill_block_spans = _regex_spans(SKILL_BLOCK_RE, response)
    skill_disclosure_spans = _merge_spans(
        _regex_spans(PRELOADED_SKILL_XML_RE, response)
        + _regex_spans(SKILL_REASONING_TAG_RE, response)
        + _line_spans_matching(response, SKILL_LINE_PAT)
        + (
            _line_spans_matching(response, V4_PRIVILEGED_PATH_PAT)
            + _line_spans_matching(response, V4_PRIVILEGED_SOURCE_PAT)
            if use_v4
            else []
        )
    )
    if use_v3_or_later:
        skill_spans = _merge_spans(skill_block_spans + skill_disclosure_spans)
    elif use_v2:
        skill_spans = skill_disclosure_spans
    else:
        skill_spans = _merge_spans(skill_block_spans + skill_disclosure_spans)
    think_spans = _regex_spans(THINK_BLOCK_RE, response)
    excluded_spans = _merge_spans(tool_response_spans + skill_spans + think_spans + malformed_tail_spans)
    action_context_excluded_spans = _merge_spans(tool_response_spans + skill_block_spans + think_spans + malformed_tail_spans)
    useful_pattern = V3_USEFUL_REASONING_PAT if use_v3_or_later else V2_USEFUL_REASONING_PAT if use_v2 else USEFUL_REASONING_PAT

    raw_action_spans = find_action_spans(response, mode=mode)
    action_spans: list[tuple[int, int]] = []
    v4_contaminated_action_spans: list[tuple[int, int]] = []
    v4_drop_final = False
    v4_reasoning_exclusion_spans: list[tuple[int, int]] = []
    if use_v4:
        direct_skill_tail_starts = [
            block_end
            for block_start, block_end in skill_block_spans
            if V4_DIRECT_SKILL_TAIL_PAT.search(response[block_start:block_end])
        ]
        if direct_skill_tail_starts:
            v4_reasoning_exclusion_spans.append((min(direct_skill_tail_starts), len(response)))
            v4_drop_final = True
    contaminated_action_span_count = 0
    action_forbidden_pattern = V3_SKILL_DERIVED_PAT if use_v3_or_later else SKILL_LINE_PAT
    for start, end in raw_action_spans:
        action_text = response[start:end]
        has_privileged_path = bool(use_v4 and V4_PRIVILEGED_PATH_PAT.search(action_text))
        has_privileged_link = bool(
            use_v4
            and _v4_has_privileged_linked_action_context(
                response,
                start,
                end,
                raw_action_spans,
                action_context_excluded_spans,
            )
        )
        drop_action = bool(
            action_forbidden_pattern.search(action_text)
            or has_privileged_path
            or _overlaps(start, end, tool_response_spans)
        )
        if use_v4 and has_privileged_link:
            v4_drop_final = True
        if drop_action:
            if use_v4:
                v4_contaminated_action_spans.append((start, end))
                if has_privileged_path:
                    v4_drop_final = True
            contaminated_action_span_count += 1
            continue
        action_spans.append((start, end))
    if use_v4 and v4_contaminated_action_spans:
        excluded_spans = _merge_spans(excluded_spans + v4_contaminated_action_spans)
    action_spans = _merge_spans(action_spans)

    reasoning_excluded_spans = (
        _merge_spans(excluded_spans + v4_reasoning_exclusion_spans)
        if use_v4 and v4_reasoning_exclusion_spans
        else excluded_spans
    )
    reasoning_spans = (
        _v3_action_context_reasoning_spans(
            response,
            action_spans,
            reasoning_excluded_spans,
            max_chars=reasoning_max_chars,
            require_useful=require_useful_reasoning,
            useful_pattern=useful_pattern,
            forbidden_pattern=V4_REASONING_FORBIDDEN_PAT if use_v4 else V3_SKILL_DERIVED_PAT,
        )
        if use_v3_or_later
        else _action_linked_reasoning_spans(
            response,
            action_spans,
            excluded_spans,
            max_chars=reasoning_max_chars,
            require_useful=require_useful_reasoning,
            useful_pattern=useful_pattern,
        )
    )
    skill_reasoning_spans = (
        _scrubbed_skill_reasoning_spans(
            response,
            skill_block_spans,
            excluded_spans,
            max_chars=reasoning_max_chars,
            require_useful=require_useful_reasoning,
            useful_pattern=useful_pattern,
            forbidden_pattern=V3_SKILL_DERIVED_PAT if use_v3 else SKILL_LINE_PAT,
        )
        if use_v2
        else []
    )
    reasoning_spans = _merge_spans(reasoning_spans + skill_reasoning_spans)
    final_spans = (
        _final_answer_spans(
            response,
            action_spans,
            tool_response_spans,
            excluded_spans,
            max_chars=final_max_chars,
        )
        if keep_final and not tail_unsafe
        and not (use_v4 and v4_drop_final)
        else []
    )

    action_mask = _spans_to_token_mask(response, n_tokens, action_spans, tokenizer, base)
    reasoning_mask = _spans_to_token_mask(response, n_tokens, reasoning_spans, tokenizer, base)
    final_mask = _spans_to_token_mask(response, n_tokens, final_spans, tokenizer, base)
    skill_mask = _spans_to_token_mask(response, n_tokens, skill_spans, tokenizer, base)
    think_mask = _spans_to_token_mask(response, n_tokens, think_spans, tokenizer, base)
    tool_response_mask = _spans_to_token_mask(response, n_tokens, tool_response_spans, tokenizer, base)
    malformed_tail_mask = _spans_to_token_mask(response, n_tokens, malformed_tail_spans, tokenizer, base)
    excluded_mask = [
        s or t or o or m
        for s, t, o, m in zip(skill_mask, think_mask, tool_response_mask, malformed_tail_mask, strict=False)
    ]

    hard_mask: list[bool] = []
    clean_action_mask: list[bool] = []
    clean_reasoning_mask: list[bool] = []
    clean_final_mask: list[bool] = []
    for action, reasoning, final, excluded, base_value in zip(
        action_mask,
        reasoning_mask,
        final_mask,
        excluded_mask,
        base,
        strict=False,
    ):
        action_value = bool(action and not excluded and base_value)
        reasoning_value = bool(reasoning and not excluded and not action_value and base_value)
        final_value = bool(final and not excluded and not action_value and not reasoning_value and base_value)
        clean_action_mask.append(action_value)
        clean_reasoning_mask.append(reasoning_value)
        clean_final_mask.append(final_value)
        hard_mask.append(action_value or reasoning_value or final_value)

    stats = _stats(
        hard_mask,
        clean_action_mask,
        clean_reasoning_mask,
        clean_final_mask,
        skill_mask,
        think_mask,
        tool_response_mask,
        n_tokens,
        drop_reason=drop_reason,
        action_span_count=len(action_spans),
        reasoning_span_count=len(reasoning_spans),
        final_span_count=len(final_spans),
        contaminated_action_span_count=contaminated_action_span_count,
        skill_reasoning_span_count=len(skill_reasoning_spans),
        version=hard_span_version,
    )
    return hard_mask, stats


def _normalize_version(version: str | None) -> str:
    normalized = (version or "v1").strip().lower()
    if normalized in {"4", "v4", "privileged_source_guard"}:
        return "v4"
    if normalized in {"3", "v3", "wide_reasoning"}:
        return "v3"
    if normalized in {"2", "v2", "scrubbed_skill_reasoning"}:
        return "v2"
    return "v1"


def _action_linked_reasoning_spans(
    text: str,
    action_spans: Sequence[tuple[int, int]],
    excluded_spans: Sequence[tuple[int, int]],
    *,
    max_chars: int,
    require_useful: bool,
    useful_pattern: re.Pattern[str],
) -> list[tuple[int, int]]:
    if not action_spans:
        return []
    turn_spans = _assistant_turn_spans(text)
    spans: list[tuple[int, int]] = []
    for action_start, action_end in action_spans:
        turn = _containing_span(action_start, action_end, turn_spans)
        if turn is None:
            continue
        turn_start, turn_end = turn
        prev_action_end = max((end for _start, end in action_spans if turn_start <= end <= action_start), default=turn_start)
        segment_start = max(turn_start, prev_action_end)
        segment_end = min(action_start, turn_end)
        for start, end in _subtract_spans([(segment_start, segment_end)], excluded_spans):
            trimmed = _trim_span(text, start, end)
            if trimmed is None:
                continue
            start, end = trimmed
            segment = text[start:end]
            if require_useful and not useful_pattern.search(segment):
                continue
            if max_chars > 0 and end - start > max_chars:
                start = end - max_chars
                trimmed = _trim_span(text, start, end)
                if trimmed is None:
                    continue
                start, end = trimmed
            spans.append((start, end))
    return _merge_spans(spans)


def _v3_action_context_reasoning_spans(
    text: str,
    action_spans: Sequence[tuple[int, int]],
    excluded_spans: Sequence[tuple[int, int]],
    *,
    max_chars: int,
    require_useful: bool,
    useful_pattern: re.Pattern[str],
    forbidden_pattern: re.Pattern[str] = V3_SKILL_DERIVED_PAT,
) -> list[tuple[int, int]]:
    """Keep broad non-skill prose that is grounded by a nearby action.

    v1/v2 intentionally keep only short useful-looking spans.  v3 is wider:
    if a same-turn prose segment before a tool call contains any concrete
    grounding signal, keep its non-filler, non-skill lines.  This preserves
    plans, hypotheses, bug localization, and test rationale while still
    excluding privileged skill disclosures and observations.
    """

    if not action_spans:
        return []
    turn_spans = _assistant_turn_spans(text)
    spans: list[tuple[int, int]] = []
    for action_start, action_end in action_spans:
        turn = _containing_span(action_start, action_end, turn_spans)
        if turn is None:
            continue
        turn_start, turn_end = turn
        prev_action_end = max((end for _start, end in action_spans if turn_start <= end <= action_start), default=turn_start)
        segment_start = max(turn_start, prev_action_end)
        segment_end = min(action_start, turn_end)
        for start, end in _subtract_spans([(segment_start, segment_end)], excluded_spans):
            spans.extend(
                _v3_reasoning_line_spans(
                    text,
                    start,
                    end,
                    max_chars=max_chars,
                    require_useful=require_useful,
                    useful_pattern=useful_pattern,
                    forbidden_pattern=forbidden_pattern,
                )
            )
    return _merge_spans(spans)


def _v3_reasoning_line_spans(
    text: str,
    start: int,
    end: int,
    *,
    max_chars: int,
    require_useful: bool,
    useful_pattern: re.Pattern[str],
    forbidden_pattern: re.Pattern[str] = V3_SKILL_DERIVED_PAT,
) -> list[tuple[int, int]]:
    trimmed = _trim_span(text, start, end)
    if trimmed is None:
        return []
    start, end = trimmed
    segment = text[start:end]
    if not _v3_segment_has_grounding(
        segment,
        require_useful=require_useful,
        useful_pattern=useful_pattern,
        forbidden_pattern=forbidden_pattern,
    ):
        return []

    spans: list[tuple[int, int]] = []
    for raw_line_start, raw_line_end in _line_spans_in_range(text, start, end):
        line_trimmed = _trim_span(text, raw_line_start, raw_line_end)
        if line_trimmed is None:
            continue
        line_start, content_end = line_trimmed
        line = text[line_start:content_end]
        if _v3_skip_reasoning_line(line, forbidden_pattern=forbidden_pattern):
            continue
        # Keep adjacent accepted lines contiguous so BC sees coherent prose
        # chunks rather than isolated line fragments.
        spans.append((line_start, min(raw_line_end, end)))
    return _cap_spans_from_end(_merge_spans(spans), max_chars)


def _v3_segment_has_grounding(
    segment: str,
    *,
    require_useful: bool,
    useful_pattern: re.Pattern[str],
    forbidden_pattern: re.Pattern[str] = V3_SKILL_DERIVED_PAT,
) -> bool:
    if forbidden_pattern.search(segment):
        scrubbed = "\n".join(line for line in segment.splitlines() if not forbidden_pattern.search(line))
    else:
        scrubbed = segment
    if not scrubbed.strip():
        return False
    if not require_useful:
        return True
    if useful_pattern.search(scrubbed):
        return True
    if "`" in scrubbed or "```" in scrubbed:
        return True
    if re.search(r"(?:^|[\s`'\"])(?:[\w.-]+/)+[\w.-]+", scrubbed):
        return True
    if re.search(r"\b[a-zA-Z_][\w]*\([^)]*\)", scrubbed):
        return True
    return False


def _v3_skip_reasoning_line(line: str, *, forbidden_pattern: re.Pattern[str] = V3_SKILL_DERIVED_PAT) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if SKILL_LINE_PAT.search(stripped):
        return True
    if forbidden_pattern.search(stripped):
        return True
    if V3_MARKUP_OR_CONTROL_PAT.match(stripped):
        return True
    if V3_GENERIC_FILLER_PAT.match(stripped):
        return True
    if len(stripped) < 4:
        return True
    return False


def _v4_has_privileged_linked_action_context(
    text: str,
    action_start: int,
    action_end: int,
    action_spans: Sequence[tuple[int, int]],
    excluded_spans: Sequence[tuple[int, int]],
) -> bool:
    turn = _containing_span(action_start, action_end, _assistant_turn_spans(text))
    if turn is None:
        return False
    turn_start, _turn_end = turn
    prev_action_end = max((end for _start, end in action_spans if turn_start <= end <= action_start), default=turn_start)
    segment_start = max(turn_start, prev_action_end)
    segment_end = action_start
    if segment_end <= segment_start:
        return False
    for start, end in _subtract_spans([(segment_start, segment_end)], excluded_spans):
        segment = text[start:end]
        if V4_LINKED_ACTION_COPY_PAT.search(segment):
            return True
    return False


def _scrubbed_skill_reasoning_spans(
    text: str,
    skill_block_spans: Sequence[tuple[int, int]],
    excluded_spans: Sequence[tuple[int, int]],
    *,
    max_chars: int,
    require_useful: bool,
    useful_pattern: re.Pattern[str],
    forbidden_pattern: re.Pattern[str] = SKILL_LINE_PAT,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    remaining = max_chars if max_chars > 0 else None
    for block_start, block_end in skill_block_spans:
        for start, end in _subtract_spans([(block_start, block_end)], excluded_spans):
            for line_start, line_end in _line_spans_in_range(text, start, end):
                trimmed = _trim_span(text, line_start, line_end)
                if trimmed is None:
                    continue
                line_start, line_end = trimmed
                segment = text[line_start:line_end]
                if forbidden_pattern.search(segment):
                    continue
                if require_useful and not useful_pattern.search(segment):
                    continue
                if remaining is not None:
                    if remaining <= 0:
                        return _merge_spans(spans)
                    take = min(line_end - line_start, remaining)
                    line_end = line_start + take
                    trimmed = _trim_span(text, line_start, line_end)
                    if trimmed is None:
                        continue
                    line_start, line_end = trimmed
                    remaining -= line_end - line_start
                spans.append((line_start, line_end))
    return _merge_spans(spans)


def _cap_spans_from_end(spans: Sequence[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    spans = _merge_spans(spans)
    if max_chars <= 0 or not spans:
        return list(spans)
    capped: list[tuple[int, int]] = []
    remaining = max_chars
    for start, end in reversed(spans):
        if remaining <= 0:
            break
        take = min(end - start, remaining)
        capped.append((end - take, end))
        remaining -= take
    return _merge_spans(list(reversed(capped)))


def _final_answer_spans(
    text: str,
    action_spans: Sequence[tuple[int, int]],
    tool_response_spans: Sequence[tuple[int, int]],
    excluded_spans: Sequence[tuple[int, int]],
    *,
    max_chars: int,
) -> list[tuple[int, int]]:
    after = 0
    if action_spans:
        after = max(after, max(end for _start, end in action_spans))
    if tool_response_spans:
        after = max(after, max(end for _start, end in tool_response_spans))

    for start, end in reversed(_assistant_turn_spans(text)):
        if start < after:
            continue
        if find_action_spans(text[start:end], mode="tool_call"):
            continue
        candidate = _subtract_spans([(start, end)], excluded_spans)
        candidate = [span for span in (_trim_span(text, s, e) for s, e in candidate) if span is not None]
        if not candidate:
            continue
        merged = _merge_spans(candidate)
        if max_chars > 0:
            capped: list[tuple[int, int]] = []
            remaining = max_chars
            for s, e in reversed(merged):
                if remaining <= 0:
                    break
                take = min(e - s, remaining)
                capped.append((e - take, e))
                remaining -= take
            merged = list(reversed(capped))
        return _merge_spans(merged)
    return []


def _assistant_turn_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    first_end = text.find(IM_END)
    first_marker = text.find("<|im_start|>")
    if text and (first_marker < 0 or first_marker > 0):
        spans.append((0, first_end if first_end >= 0 else len(text)))
    for match in ASSISTANT_START_RE.finditer(text):
        start = match.end()
        end = text.find(IM_END, start)
        if end < 0:
            end = len(text)
        spans.append((start, end))
    return _merge_spans(spans)


def _spans_to_token_mask(
    text: str,
    n_tokens: int,
    spans: Sequence[tuple[int, int]],
    tokenizer: Any,
    base_mask: Sequence[bool],
) -> list[bool]:
    mask = [False] * n_tokens
    spans = _merge_spans(spans)
    if n_tokens <= 0 or not spans:
        return mask

    offsets = _token_offsets(tokenizer, text)
    if offsets is not None:
        for idx, (start, end) in enumerate(offsets[:n_tokens]):
            if idx >= len(base_mask) or not base_mask[idx]:
                continue
            if end <= start:
                point = start
                overlap = any(span_start <= point < span_end for span_start, span_end in spans)
            else:
                overlap = any(start < span_end and end > span_start for span_start, span_end in spans)
            if overlap:
                mask[idx] = True
        return mask

    for start_char, end_char in spans:
        start_token = _safe_token_count(tokenizer, text[:start_char], n_tokens)
        end_token = _safe_token_count(tokenizer, text[:end_char], n_tokens)
        for idx in range(start_token, end_token):
            if idx < len(base_mask) and base_mask[idx]:
                mask[idx] = True
    return mask


def _token_offsets(tokenizer: Any, text: str) -> list[tuple[int, int]] | None:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        return None
    offsets = encoded["offset_mapping"] if isinstance(encoded, dict) else getattr(encoded, "offset_mapping", None)
    if offsets is None:
        return None
    return [(int(start), int(end)) for start, end in offsets]


def _safe_token_count(tokenizer: Any, text: str, upper_bound: int) -> int:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        count = len(input_ids)
    except Exception:
        count = 0
    return max(0, min(int(count), upper_bound))


def build_skill_register_token_mask(
    response: str,
    *,
    response_token_count: int,
    tokenizer: Any,
) -> tuple[list[bool], dict[str, float]]:
    """Token mask of skill-register text for OPSD distill-term gating.

    True marks tokens inside skill-register text: <skill_reasoning> blocks and
    stray tags, preloaded-skill XML, skill-referencing lines (SKILL_LINE_PAT /
    V3_SKILL_DERIVED_PAT), and privileged path/source prose (V4 guards) --
    the same keyword recipes as hard-span v1-v4. Unlike hard-span this is a
    negative mask applied ONLY to the OPSD distillation term; the GRPO loss
    is unaffected.
    """
    n_tokens = max(int(response_token_count or 0), 0)
    stats = {"register_token_count": 0.0, "register_token_frac": 0.0}
    if n_tokens <= 0 or not response:
        return [False] * n_tokens, stats
    spans = _merge_spans(
        _regex_spans(SKILL_BLOCK_RE, response)
        + _regex_spans(PRELOADED_SKILL_XML_RE, response)
        + _regex_spans(SKILL_REASONING_TAG_RE, response)
        + _line_spans_matching(response, SKILL_LINE_PAT)
        + _line_spans_matching(response, V3_SKILL_DERIVED_PAT)
        + _line_spans_matching(response, V4_PRIVILEGED_PATH_PAT)
        + _line_spans_matching(response, V4_PRIVILEGED_SOURCE_PAT)
    )
    if not spans:
        return [False] * n_tokens, stats
    mask = _spans_to_token_mask(response, n_tokens, spans, tokenizer, [True] * n_tokens)
    count = float(sum(mask))
    stats["register_token_count"] = count
    stats["register_token_frac"] = count / n_tokens if n_tokens else 0.0
    return mask, stats


def _regex_spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _line_spans_matching(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        end = pos + len(line)
        if pattern.search(line):
            spans.append((pos, end))
        pos = end
    return spans


def _line_spans_in_range(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        newline = text.find("\n", pos, end)
        line_end = end if newline < 0 else newline + 1
        spans.append((pos, line_end))
        pos = line_end
    return spans


def _has_malformed_tags(text: str) -> bool:
    action_text = TOOL_RESPONSE_RE.sub("", text)
    for open_pat, close_tag in MALFORMED_TAGS:
        haystack = text if close_tag == "</tool_response>" else action_text
        if _count_structural_matches(open_pat, haystack) != _count_structural_literal(close_tag, haystack):
            return True
    return False


def _count_structural_matches(pattern: re.Pattern[str], text: str) -> int:
    return sum(1 for match in pattern.finditer(text) if not _inside_inline_code(text, match.start()))


def _count_structural_literal(literal: str, text: str) -> int:
    pattern = re.compile(re.escape(literal), re.IGNORECASE)
    return _count_structural_matches(pattern, text)


def _tail_open_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    lower = text.lower()
    for open_text, close_text in (
        ("<skill_reasoning", "</skill_reasoning>"),
        ("<think", "</think>"),
        ("<tool_response", "</tool_response>"),
        ("<tool_call", "</tool_call>"),
        ("<function=", "</function>"),
        ("<preloaded_oracle_skill", "</preloaded_oracle_skill>"),
        ("<preloaded_top1_skill", "</preloaded_top1_skill>"),
    ):
        open_positions = [
            match.start()
            for match in re.finditer(re.escape(open_text), lower)
            if not _inside_inline_code(text, match.start())
        ]
        close_positions = [match.end() for match in re.finditer(re.escape(close_text), lower)]
        close_cursor = 0
        unmatched: list[int] = []
        for pos in open_positions:
            while close_cursor < len(close_positions) and close_positions[close_cursor] < pos:
                close_cursor += 1
            if close_cursor < len(close_positions):
                close_cursor += 1
            else:
                unmatched.append(pos)
        if unmatched:
            spans.append((unmatched[-1], len(text)))
    return _merge_spans(spans)


def _inside_inline_code(text: str, pos: int) -> bool:
    line_start = text.rfind("\n", 0, pos) + 1
    before = text[line_start:pos]
    return before.count("`") % 2 == 1


def _containing_span(start: int, end: int, spans: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    for span_start, span_end in spans:
        if span_start <= start and end <= span_end:
            return span_start, span_end
    return None


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return start, end


def _subtract_spans(
    spans: Sequence[tuple[int, int]],
    excluded: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    remaining = _merge_spans(spans)
    for ex_start, ex_end in _merge_spans(excluded):
        next_remaining: list[tuple[int, int]] = []
        for start, end in remaining:
            if end <= ex_start or start >= ex_end:
                next_remaining.append((start, end))
                continue
            if start < ex_start:
                next_remaining.append((start, ex_start))
            if ex_end < end:
                next_remaining.append((ex_end, end))
        remaining = next_remaining
    return _merge_spans(remaining)


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((max(int(s), 0), max(int(e), 0)) for s, e in spans if e > s)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _overlaps(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _base_mask(values: Sequence[int | bool] | None, n_tokens: int) -> list[bool]:
    if values is None:
        return [True] * n_tokens
    base = [bool(value) for value in list(values)[:n_tokens]]
    if len(base) < n_tokens:
        base.extend([False] * (n_tokens - len(base)))
    return base


def _count(mask: Sequence[bool]) -> float:
    return float(sum(1 for value in mask if value))


def _stats(
    hard_mask: Sequence[bool],
    action_mask: Sequence[bool],
    reasoning_mask: Sequence[bool],
    final_mask: Sequence[bool],
    skill_mask: Sequence[bool],
    think_mask: Sequence[bool],
    tool_response_mask: Sequence[bool],
    n_tokens: int,
    *,
    drop_reason: str,
    action_span_count: int,
    reasoning_span_count: int,
    final_span_count: int,
    contaminated_action_span_count: int,
    skill_reasoning_span_count: int,
    version: str,
) -> dict[str, float | str]:
    hard_count = _count(hard_mask)
    return {
        "drop_reason": drop_reason,
        "version": version,
        "token_count": hard_count,
        "token_frac": hard_count / float(n_tokens) if n_tokens else 0.0,
        "action_token_count": _count(action_mask),
        "reasoning_token_count": _count(reasoning_mask),
        "final_token_count": _count(final_mask),
        "excluded_skill_token_count": _count(skill_mask),
        "excluded_think_token_count": _count(think_mask),
        "excluded_tool_response_token_count": _count(tool_response_mask),
        "span_count": float(action_span_count + reasoning_span_count + final_span_count),
        "action_span_count": float(action_span_count),
        "reasoning_span_count": float(reasoning_span_count),
        "skill_reasoning_span_count": float(skill_reasoning_span_count),
        "final_span_count": float(final_span_count),
        "contaminated_action_span_count": float(contaminated_action_span_count),
    }

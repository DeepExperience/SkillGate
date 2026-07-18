"""Utilities for SIRI-style action-token BC masks.

The prompt-only oracle BC branch should not imitate oracle-conditioned prose or
``<skill_reasoning>`` text.  For the first action-only variant we supervise only
serialized tool calls, which are the executable action spans in the agent trace.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any


_TOOL_CALL_RE = re.compile(r"<tool_call\b[^>]*>.*?</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=[^>]+>.*?</function>", re.IGNORECASE | re.DOTALL)


def find_action_spans(response: str, *, mode: str = "tool_call") -> list[tuple[int, int]]:
    """Return character spans for executable action blocks in a response.

    ``mode`` currently supports ``tool_call`` / ``tool_call_or_function``.  The
    second spelling is accepted for explicitness, but both behave the same:
    prefer complete ``<tool_call>`` blocks and fall back to bare
    ``<function=...>`` blocks only when no outer tool-call wrapper is present.
    """

    if not response:
        return []

    normalized_mode = (mode or "tool_call").strip().lower()
    if normalized_mode not in {"tool_call", "tool_call_or_function", "function"}:
        raise ValueError(
            f"Unknown RELAX_SHADOW_ACTION_MASK_MODE={mode!r}; expected tool_call or tool_call_or_function"
        )

    spans = [(match.start(), match.end()) for match in _TOOL_CALL_RE.finditer(response)]
    if spans and normalized_mode != "function":
        return _merge_spans(spans)

    fallback_spans = [(match.start(), match.end()) for match in _FUNCTION_RE.finditer(response)]
    return _merge_spans(fallback_spans)


def build_action_token_mask(
    response: str,
    *,
    response_token_count: int,
    tokenize_len: Callable[[str], int],
    mode: str = "tool_call",
) -> tuple[list[bool], dict[str, float]]:
    """Build a response-token mask from action character spans.

    The rollout trace stores response text, while training loss is indexed by
    response tokens.  We map each action character span to token offsets by
    tokenizing response prefixes with the same HF tokenizer used by the run.
    This is deliberately conservative and clamps to the known response length.
    """

    n_tokens = max(int(response_token_count or 0), 0)
    mask = [False] * n_tokens
    spans = find_action_spans(response, mode=mode)
    if n_tokens <= 0 or not spans:
        return mask, {
            "span_count": float(len(spans)),
            "action_token_count": 0.0,
            "action_token_frac": 0.0,
            "char_span_frac": 0.0,
        }

    char_span_chars = sum(max(end - start, 0) for start, end in spans)
    response_chars = max(len(response), 1)

    for start_char, end_char in spans:
        start_token = _safe_token_count(tokenize_len, response[:start_char], n_tokens)
        end_token = _safe_token_count(tokenize_len, response[:end_char], n_tokens)
        if end_token <= start_token:
            continue
        for idx in range(start_token, end_token):
            mask[idx] = True

    action_token_count = float(sum(1 for value in mask if value))
    return mask, {
        "span_count": float(len(spans)),
        "action_token_count": action_token_count,
        "action_token_frac": action_token_count / float(n_tokens) if n_tokens else 0.0,
        "char_span_frac": float(char_span_chars) / float(response_chars),
    }


def make_tokenize_len(tokenizer: Any) -> Callable[[str], int]:
    """Return a tokenizer length callable with no special-token insertion."""

    def tokenize_len(text: str) -> int:
        if not text:
            return 0
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return len(input_ids)

    return tokenize_len


def _safe_token_count(tokenize_len: Callable[[str], int], text: str, upper_bound: int) -> int:
    try:
        count = int(tokenize_len(text))
    except Exception:
        count = 0
    return max(0, min(count, upper_bound))


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
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

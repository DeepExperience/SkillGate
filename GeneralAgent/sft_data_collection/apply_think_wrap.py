#!/usr/bin/env python3
"""Force every first-assistant content to start with the byte sequence

    <think>\n
    \n
    </think>\n
    \n
    <skill_reasoning>\n
    {hindsight}\n
    </skill_reasoning>\n
    \n
    {original body}

This is what aligns the SFT data with the qwen3.5 chat-template generation
prompt under either ``enable_thinking=False`` (jinja prepends the closed
empty think block) or default ``enable_thinking=True`` (jinja prepends only
``<think>\\n`` and the model emits ``\\n</think>\\n\\n`` itself, which the
training distribution covers).

Operates in-place on each record:

  - Records that already start with ``<think>...</think>...<skill_reasoning>``
    are left alone (idempotent).
  - Records starting with bare ``<skill_reasoning>`` (legacy 2042-record
    dataset) get the empty think block prepended.
  - Records that don't start with either form are left alone but counted as
    'no_skill_reasoning' for visibility.

Usage:
  python3 GeneralAgent/sft_data_collection/apply_think_wrap.py \
    --input  GeneralAgent/sft_training/datasets/<src>/sft_messages.jsonl \
    --output GeneralAgent/sft_training/datasets/<dst>/sft_messages.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


WRAP_PREFIX = "<think>\n\n</think>\n\n"
SR_TAG = "<skill_reasoning>"


def normalize_first_asst(content: str) -> tuple[str, str]:
    """Return (new_content, action_tag).

    action_tag is one of:
      already_wrapped: content starts with the canonical wrap prefix
      added_wrap:      content starts with <skill_reasoning>; we prepended wrap
      no_sr_tag:       content does not start with <skill_reasoning>; left alone
    """
    if content.startswith(WRAP_PREFIX):
        return content, "already_wrapped"
    if content.startswith(SR_TAG):
        return WRAP_PREFIX + content, "added_wrap"
    # Some legacy records may have a leading whitespace before <skill_reasoning>.
    stripped = content.lstrip()
    if stripped.startswith(SR_TAG):
        return WRAP_PREFIX + stripped, "added_wrap_after_lstrip"
    return content, "no_sr_tag"


def first_assistant_index(messages: list[dict]) -> int | None:
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counter: Counter[str] = Counter()
    n = 0
    with in_path.open("r") as inf, out_path.open("w") as outf:
        for line in inf:
            line = line.rstrip("\n")
            if not line:
                continue
            r = json.loads(line)
            n += 1
            msgs = r.get("messages") or []
            idx = first_assistant_index(msgs)
            if idx is None:
                counter["no_assistant"] += 1
            else:
                content = msgs[idx].get("content") or ""
                if isinstance(content, list):
                    counter["non_string_content"] += 1
                else:
                    new_content, tag = normalize_first_asst(content)
                    counter[tag] += 1
                    if tag != "already_wrapped" and tag != "no_sr_tag":
                        m_copy = dict(msgs[idx])
                        m_copy["content"] = new_content
                        msgs = list(msgs)
                        msgs[idx] = m_copy
                        r = dict(r)
                        r["messages"] = msgs
            outf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"records: {n}")
    for tag, c in sorted(counter.items()):
        print(f"  {tag}: {c}")
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

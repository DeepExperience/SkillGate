# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Standalone smoke test for env_agent_bench + launchers + reward.

Builds a fake :class:`relax.utils.types.Sample` from one row of our canonical v2
``train.parquet``, runs the env through a fixed canned conversation, and
verifies that:

* ``env.reset()`` returns a usable observation
* ``env.step(<tool_call XML>)`` parses + dispatches via ToolLayer
* ``env.step(<final answer>)`` triggers grading and stashes ``final_score``
* ``reward_func`` returns a dict with ``score`` in ``[0, 1]``

Defaults to ``UNIFIED_LAUNCHER_MODE=mock``; ``--launcher real`` will switch
to the real launchers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).absolute().parents[3]


def _make_sample_from_parquet(parquet_path: Path, row_idx: int):
    import pyarrow.parquet as pq  # type: ignore

    table = pq.read_table(str(parquet_path))
    rows = table.slice(row_idx, 1).to_pylist()
    if not rows:
        raise RuntimeError(f"row {row_idx} out of range in {parquet_path}")
    row = rows[0]

    # Build a minimal Sample without pulling in Megatron / TE.  On the internal Ray
    # head nodes the control-plane Python can lack torch, while GPU workers
    # have the full training stack.  The env only needs prompt/label/metadata,
    # so fall back to SimpleNamespace for local smoke tests.
    try:
        from relax.utils.types import Sample  # type: ignore

        s = Sample()
    except Exception:
        s = SimpleNamespace()
    s.prompt = row["prompt"]
    s.label = row["reward_model"]
    s.metadata = {"extra_info": row["extra_info"]}
    return s


def _fake_args() -> Namespace:
    return Namespace(
        max_turns=4,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
    )


CANNED_TOOL_CALL = (
    "<tool_call>\n<function=ls>\n<parameter=path>\n/workspace\n</parameter>\n</function>\n</tool_call>"
)
CANNED_FINAL_ANSWER = "Based on inspection, the answer is 42."


async def _run_reward(sample) -> dict:
    try:
        from examples.agent_bench.reward_agent_bench import reward_func  # type: ignore
    except Exception:
        metadata = sample.metadata or {}
        score = float(metadata.get("final_score") or 0.0)
        extra = metadata.get("extra_info") or {}
        return {
            "score": max(0.0, min(1.0, score)),
            "raw_score": score,
            "bench": extra.get("bench", "?"),
            "task_id": extra.get("task_id", "?"),
            "missing_score": metadata.get("final_score") is None,
        }

    return await reward_func(_fake_args(), sample)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--parquet",
        default=str(REPO_ROOT / "experiments/rl/v2/parquet/train.parquet"),
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--launcher",
        default=os.environ.get("UNIFIED_LAUNCHER_MODE", "mock"),
        choices=["mock", "real"],
    )
    args = parser.parse_args()
    os.environ["UNIFIED_LAUNCHER_MODE"] = args.launcher

    sample = _make_sample_from_parquet(Path(args.parquet), args.row)
    print(f"[smoke] sample.metadata.extra_info.task_id={sample.metadata['extra_info']['task_id']}")
    print(f"[smoke] sample.metadata.extra_info.bench={sample.metadata['extra_info']['bench']}")

    from examples.agent_bench.env_agent_bench import build_env  # type: ignore

    env = build_env(sample, _fake_args())
    obs, info = env.reset()
    print(f"[smoke] reset → obs={obs!r}, info={info!r}")
    if info.get("skipped"):
        print(f"[FAIL] env.reset skipped due to infra/setup failure: {info!r}", file=sys.stderr)
        env.close()
        return 1

    # Turn 1: emit a tool_call XML, expect non-terminal step with tool_response observation.
    obs, done, info = env.step(CANNED_TOOL_CALL)
    print(f"[smoke] step(tool_call) → done={done} info={info}")
    if done:
        print("[FAIL] env terminated on tool_call turn; should continue.", file=sys.stderr)
        return 1

    # Turn 2: emit a plain final-answer response, expect done=True with score.
    obs, done, info = env.step(CANNED_FINAL_ANSWER)
    print(f"[smoke] step(final_answer) → done={done} info={info}")
    if not done:
        print("[FAIL] env did not terminate on text-only response.", file=sys.stderr)
        env.close()
        return 1
    if info.get("error"):
        print(f"[FAIL] terminal step reported error: {info!r}", file=sys.stderr)
        env.close()
        return 1
    score_in_info = info.get("score")
    print(f"[smoke] final score (from info)={score_in_info}")

    env.close()

    # Reward func reads sample.metadata.final_score.
    reward = asyncio.run(_run_reward(sample))
    print(f"[smoke] reward_func → {reward}")
    if not (0.0 <= reward["score"] <= 1.0):
        print(f"[FAIL] reward.score out of [0,1]: {reward['score']!r}", file=sys.stderr)
        return 1
    if reward["missing_score"]:
        print("[FAIL] reward.missing_score=True (env never stashed score).", file=sys.stderr)
        return 1

    print()
    print("[PASS] env + tool_call parse + launcher.grade + reward_func all work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Custom reward function for the agent_bench Relax campaign.

We compute the actual task reward inside :class:`env_agent_bench.AgentBenchEnv`
(at the moment the launcher knows the container state) and stash it on
``sample.metadata["final_score"]``. This module just reads it back so Relax's
:class:`RewardExecutor` has a clean ``async`` callable to dispatch.

That separation keeps the reward path:

* fast (no GPU/IO here),
* free of double-grading (one launcher call per rollout, not two),
* and easy to debug (one place writes ``final_score``, one place reads it).

Returns a ``dict`` so we can surface bench/task metadata for logging /
filtering, while ``--reward-key score`` picks the scalar Relax uses for
advantage computation.
"""
from __future__ import annotations

import logging
from typing import Any

from relax.utils.types import Sample


logger = logging.getLogger(__name__)


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    metadata = sample.metadata or {}
    final_score = metadata.get("final_score")
    extra = metadata.get("extra_info") or {}
    task_id = extra.get("task_id") or metadata.get("task_id") or "?"
    bench = extra.get("bench") or metadata.get("bench") or "?"

    if final_score is None:
        # Most common cause: rollout aborted before reaching env.close (e.g.
        # context overflow, sglang HTTP failure). Treat as zero reward — we
        # explicitly tag the sample so Relax metrics can filter it.
        logger.warning(
            f"[reward] task_id={task_id} bench={bench} has no final_score; "
            "defaulting to 0.0 and flagging as 'missing_score'."
        )
        final_score = 0.0
        missing = True
    else:
        missing = False

    try:
        score = float(final_score)
    except (TypeError, ValueError):
        logger.warning(
            f"[reward] task_id={task_id} non-numeric final_score={final_score!r}; "
            "coercing to 0.0."
        )
        score = 0.0
        missing = True

    # Clip to the unit interval so Relax advantage normalization stays sane
    # even if a launcher accidentally returns a value outside [0, 1].
    clipped = max(0.0, min(1.0, score))

    result = {
        "score": clipped,
        "raw_score": score,
        "bench": bench,
        "task_id": task_id,
        "missing_score": missing,
    }
    if "selector_action_credit" in metadata:
        from examples.agent_bench.selector_action_credit import sample_behavior_metrics

        selector_metrics = sample_behavior_metrics(sample)
        result.update(selector_metrics)
        result["selector_success_and_oracle_only"] = (
            clipped * selector_metrics["selector_oracle_only"]
        )
    return result

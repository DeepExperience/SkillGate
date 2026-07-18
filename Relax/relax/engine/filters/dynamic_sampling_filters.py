# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.types import Sample


__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = []
    for sample in samples:
        if sample.reward is None:
            return DynamicFilterOutput(keep=False, reason="missing_reward")
        rewards.append(sample.get_reward_value(args))
    keep = torch.tensor(rewards, dtype=torch.float).std() > 0.0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )

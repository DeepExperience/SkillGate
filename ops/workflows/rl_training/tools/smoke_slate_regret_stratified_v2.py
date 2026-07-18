#!/usr/bin/env python3
"""CPU-only smoke test for gold-only SlateRL stratified advantage.

No Ray, Docker, SGLang, model weights, or GPUs are used.  The test covers:
- disabled-path equivalence with the original slate-regret postprocessor;
- strict oracle/misleading/no-read/other attribution;
- misleading precedence when both misleading and oracle are read;
- shrinkage, clipping, and weighted-zero-mean behavior advantage;
- raw verifier reward preservation and fail-fast all-gold metadata checks.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import types
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "Relax"))

# This workspace host intentionally does not install the training container's
# PyTorch.  The deterministic reward modules only need torch names for type
# annotations at import time; no tensor operation is used by this smoke test.
if importlib.util.find_spec("torch") is None:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.dtype = object
    torch_stub.Size = tuple
    torch_stub.float = float
    sys.modules["torch"] = torch_stub

from examples.agent_bench.slate_regret_gating import (  # noqa: E402
    post_process_rewards as old_post_process_rewards,
)
from examples.agent_bench.slate_regret_stratified_gating import (  # noqa: E402
    behavior_stratum,
    post_process_rewards,
    strict_read_skill_names,
)
from relax.utils.types import Sample  # noqa: E402


ORACLE = "oracle-skill"
MISLEADING = [f"misleading-{index}" for index in range(5)]
RELEVANT = [f"relevant-{index}" for index in range(5)]
IRRELEVANT = [f"irrelevant-{index}" for index in range(5)]
ADVERTISED = [ORACLE, *MISLEADING, *RELEVANT, *IRRELEVANT]


def _read_call(*names: str) -> str:
    return "\n".join(
        "<tool_call><function=read><parameter=path>"
        f"/root/.claude/skills/{name}/SKILL.md"
        "</parameter></function></tool_call>"
        for name in names
    )


def _sample(score: float, names: tuple[str, ...], index: int, *, has_gold: bool = True) -> Sample:
    extra = {
        "update_kind": "slate_grpo",
        "hybrid_update_kind": "slate_grpo",
        "task_id": "smoke-task",
        "bench": "seta_synth",
        "slate_contains_gold": 1.0 if has_gold else 0.0,
        "slate_gold_name": ORACLE if has_gold else "",
        "slate_oracle_names": [ORACLE] if has_gold else [],
        "slate_misleading_names": MISLEADING,
        "retrieval_skills_top_n": ADVERTISED if has_gold else ADVERTISED[1:],
        "relax_pair_no_skill_mean_reward": 0.5,
        "hybrid_grpo_weight": 1.0,
    }
    return Sample(
        index=index,
        response=_read_call(*names) if names else "Solve directly.",
        response_length=32,
        reward={"score": score, "raw_score": score},
        metadata={"extra_info": extra},
        train_metadata={},
        status=Sample.Status.COMPLETED,
    )


def _group() -> list[Sample]:
    return [
        _sample(1.0, (ORACLE,), 0),
        _sample(1.0, (ORACLE,), 1),
        _sample(0.0, (MISLEADING[0],), 2),
        _sample(0.0, (ORACLE, MISLEADING[1]), 3),
        _sample(1.0, (), 4),
        _sample(0.0, (), 5),
        _sample(1.0, (RELEVANT[0],), 6),
        _sample(0.0, (IRRELEVANT[0],), 7),
    ]


def _args() -> Namespace:
    return Namespace(
        reward_key="score",
        n_samples_per_prompt=8,
        rewards_normalization=False,
        grpo_std_normalization=False,
    )


def _set_env() -> None:
    os.environ["RELAX_SKILL_GROUP_REWARD"] = "0"
    os.environ["RELAX_SLATE_REGRET_GRPO"] = "1"
    os.environ["RELAX_SLATE_REGRET_COEF"] = "0.5"
    os.environ["RELAX_SLATE_STRATIFIED_ADVANTAGE"] = "1"
    os.environ["RELAX_SLATE_STRATIFIED_ADV_COEF"] = "1.0"
    os.environ["RELAX_SLATE_STRATIFIED_SHRINKAGE"] = "1.0"
    os.environ["RELAX_SLATE_STRATIFIED_ADV_CLIP"] = "0.5"


def test_attribution() -> None:
    group = _group()
    assert behavior_stratum(group[0])[0] == "oracle"
    assert behavior_stratum(group[2])[0] == "misleading"
    assert behavior_stratum(group[3])[0] == "misleading", "misleading must take precedence"
    assert behavior_stratum(group[4])[0] == "no_read"
    assert behavior_stratum(group[6])[0] == "other"
    assert strict_read_skill_names(group[3]) == {ORACLE, MISLEADING[1]}


def test_disabled_equivalence() -> None:
    os.environ["RELAX_SLATE_STRATIFIED_ADVANTAGE"] = "0"
    old_group = _group()
    new_group = _group()
    old_raw, old_processed = old_post_process_rewards(_args(), old_group)
    new_raw, new_processed = post_process_rewards(_args(), new_group)
    assert new_raw == old_raw
    assert new_processed == old_processed
    os.environ["RELAX_SLATE_STRATIFIED_ADVANTAGE"] = "1"


def test_advantage_math_and_reward_preservation() -> None:
    group = _group()
    original_rewards = [copy.deepcopy(sample.reward) for sample in group]
    raw, processed = post_process_rewards(_args(), group)

    # Eligible strata each have n=2 and eligible mean=0.5.  With tau=1:
    # oracle -> (2*1 + 1*.5)/3 - .5 = +1/3
    # misleading -> (2*0 + 1*.5)/3 - .5 = -1/3
    # no_read -> 0.  Other is deliberately unshaped.
    expected_additions = [1 / 3, 1 / 3, -1 / 3, -1 / 3, 0.0, 0.0, 0.0, 0.0]
    base = [float(item["score"]) for item in original_rewards]
    assert raw == base
    for actual, expected in zip(processed, (x + y for x, y in zip(base, expected_additions))):
        assert abs(actual - expected) < 1e-9, (actual, expected)
    assert abs(sum(expected_additions)) < 1e-12
    assert [sample.reward["raw_score"] for sample in group] == base
    assert [sample.reward["score"] for sample in group] == base
    assert group[0].train_metadata["slate_stratified_stratum"] == "oracle"
    assert group[3].train_metadata["slate_stratified_stratum"] == "misleading"


def test_clip() -> None:
    os.environ["RELAX_SLATE_STRATIFIED_ADV_CLIP"] = "0.1"
    group = _group()
    _, _ = post_process_rewards(_args(), group)
    additions = [abs(sample.train_metadata["slate_stratified_adv_addition"]) for sample in group]
    assert max(additions) <= 0.1 + 1e-12
    os.environ["RELAX_SLATE_STRATIFIED_ADV_CLIP"] = "0.5"


def test_gold_guard() -> None:
    bad_group = [_sample(float(index % 2), (), index, has_gold=False) for index in range(8)]
    try:
        post_process_rewards(_args(), bad_group)
    except RuntimeError as error:
        assert "requires slate_contains_gold=1" in str(error)
    else:
        raise AssertionError("gold-absent slate group did not fail fast")


def main() -> None:
    _set_env()
    test_attribution()
    test_disabled_equivalence()
    test_advantage_math_and_reward_preservation()
    test_clip()
    test_gold_guard()
    print("slate-regret gold-stratified v2 smoke: OK")


if __name__ == "__main__":
    main()

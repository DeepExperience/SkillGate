# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Agent benchmark adapter for Relax GRPO training.

Plugs OpenClaw tool-calling agent benchmarks (SkillsBench, Terminal-Bench 2.0,
SETA, SWE-Gym, Claw-Eval) into Relax via the three standard hook points:
``--custom-generate-function-path`` (``rollout.generate``),
``--custom-rm-path`` (``reward_agent_bench.reward_func``),
and the :class:`BaseInteractionEnv` subclass in :mod:`env_agent_bench`.
"""

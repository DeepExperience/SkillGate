# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Per-bench sandbox launchers used by env_agent_bench.AgentBenchEnv.

Each launcher exposes the same lifecycle interface:

    launcher = <Launcher>(task_id: str, task_kwargs: dict)
    container = launcher.start()        # spin up docker / sandbox; return container name or handle
    score     = launcher.grade(final_answer=None, container_state=False)  # in [0, 1] (or {0, 1})
    launcher.teardown()                 # tear down container / free port

The launcher *owns* the container lifecycle and the final reward extraction
so the surrounding rollout (`env_agent_bench.AgentBenchEnv`) only needs to
sequence ``start → tool dispatch loop → grade → teardown``.
"""

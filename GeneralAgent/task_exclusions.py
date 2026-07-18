"""Central task exclusion list for broken benchmark environments.

These exclusions are task-level, not trajectory-level. They are used by SFT
collection/export, evaluation queue builders, and RL parquet generation so a
known-broken Docker environment is not reintroduced by later data regeneration.
"""

from __future__ import annotations

from typing import Iterable


# Confirmed Dockerfile/build-context or pinned-dependency failures. Do not add
# mere timeouts here; timeout-heavy tasks stay in the prebuild backlog and are
# handled by runtime setup caps.
CONFIRMED_BAD_DOCKER_TASKS: dict[tuple[str, str], str] = {
    ("seta_synth", "244"): "previously confirmed broken SETA Dockerfile during RL data prep",
    ("seta_synth", "436"): "previously confirmed broken SETA Dockerfile during RL data prep",
    ("seta_synth", "729"): "previously confirmed broken SETA Dockerfile during RL data prep",
    ("seta_synth", "25"): "Dockerfile copies sample.log, but sample.log is absent from build context",
    (
        "sb_ns",
        "speaker-diarization-subtitles",
    ): "pinned openai-whisper==20231117 fails current pip build isolation with missing pkg_resources",
    (
        "sb_ns",
        "multilingual-video-dubbing",
    ): "Dockerfile build fails at Kokoro KPipeline warmup step; confirmed structural image build failure",
    ("seta_synth", "1132"): "dataset task has no usable verifier test.sh; confirmed RL preflight runner_or_launcher_bug",
}


_BENCH_ALIASES = {
    "seta": "seta_synth",
    "seta-synth": "seta_synth",
    "seta_synth": "seta_synth",
    "skillsbench": "sb_ns",
    "skillsbench-no-skills": "sb_ns",
    "sb_ns": "sb_ns",
    "swe": "swe_lite",
    "swe_lite": "swe_lite",
    "tb2": "tb2",
    "claw": "claw",
}


def canonical_bench(bench: str) -> str:
    return _BENCH_ALIASES.get(str(bench), str(bench))


def is_bad_task(bench: str, task_id: str | int) -> bool:
    return (canonical_bench(bench), str(task_id)) in CONFIRMED_BAD_DOCKER_TASKS


def bad_reason(bench: str, task_id: str | int) -> str:
    return CONFIRMED_BAD_DOCKER_TASKS.get((canonical_bench(bench), str(task_id)), "")


def bad_task_ids(bench: str) -> set[str]:
    canonical = canonical_bench(bench)
    return {
        task_id
        for (bad_bench, task_id), _reason in CONFIRMED_BAD_DOCKER_TASKS.items()
        if bad_bench == canonical
    }


def filter_bad_tasks(bench: str, task_ids: Iterable[str | int]) -> list[str]:
    return [str(task_id) for task_id in task_ids if not is_bad_task(bench, task_id)]

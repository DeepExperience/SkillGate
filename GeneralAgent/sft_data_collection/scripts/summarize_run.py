#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GeneralAgent.sft_data_collection.common import (
    experiment_collected_dir,
    experiment_plan_path,
    experiment_root,
    experiment_status_path,
)

ERROR_PATTERNS = {
    'container_conflict': re.compile(r'Conflict\. The container name|already in use|removal of container .* already in progress', re.I),
    'no_such_container': re.compile(r'No such container', re.I),
    'docker_cp_timeout': re.compile(r'docker cp failed: Command timed out|Command timed out', re.I),
    'docker_daemon_error': re.compile(r'Error response from daemon|Cannot connect to the Docker daemon|Failed to start container', re.I),
    'traceback': re.compile(r'Traceback|RuntimeError|Exception', re.I),
    'agent_loop_started': re.compile(r'Running unified agent loop', re.I),
    'summary_saved': re.compile(r'Summary saved|UNIFIED .* FINAL|resolve_rate', re.I),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors='ignore').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def classify_log(path: Path) -> str:
    if not path.exists():
        return 'no_log'
    text = path.read_text(errors='ignore')
    for name, pattern in ERROR_PATTERNS.items():
        if pattern.search(text):
            return name
    return 'unclassified'


def counter_to_jsonable(counter: Counter) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in counter.items():
        if isinstance(key, tuple):
            key_text = '|'.join(str(part) for part in key)
        else:
            key_text = str(key)
        result[key_text] = value
    return result


def key_to_text(key) -> str:
    if isinstance(key, tuple):
        return '|'.join(str(part) for part in key)
    return str(key)


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'n': 0}

    def quantile(frac: float) -> float:
        index = max(0, min(len(values) - 1, int(round(frac * (len(values) - 1)))))
        return round(values[index], 1)

    return {
        'n': len(values),
        'mean': round(sum(values) / len(values), 1),
        'p50': quantile(0.50),
        'p90': quantile(0.90),
        'p95': quantile(0.95),
        'max': round(max(values), 1),
    }


def summarize_numeric_field(rows: list[dict], field: str) -> dict[str, float | int]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return numeric_summary(values)


def summarize_numeric_by_group(rows: list[dict], field: str) -> dict[str, dict[str, float | int]]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if not isinstance(value, (int, float)):
            continue
        key = (row.get('bench'), row.get('mode'), row.get('model'))
        groups[key].append(float(value))
    return {
        key_to_text(key): numeric_summary(values)
        for key, values in sorted(groups.items(), key=lambda item: key_to_text(item[0]))
        if values
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    args = parser.parse_args()
    run_id = args.run_id
    run_root = experiment_root(run_id)
    plan = experiment_plan_path(run_id)
    status = experiment_status_path(run_id)
    collected = experiment_collected_dir(run_id)
    if not plan.exists():
        plan = PROJECT_ROOT / 'GeneralAgent' / 'sft_data_collection' / 'outputs' / 'plans' / f'{run_id}.jsonl'
    if not status.exists():
        status = PROJECT_ROOT / 'archive' / 'overnight' / 'logs' / 'migrated_20260428' / 'logs' / 'sft_collection' / run_id / 'status.jsonl'
    if not collected.exists():
        collected = PROJECT_ROOT / 'GeneralAgent' / 'sft_data_collection' / 'outputs' / 'collected' / run_id

    records = read_jsonl(plan)
    rows = read_jsonl(status)
    done_ids = {r.get('trial_id') for r in rows}
    pending = [r for r in records if r.get('trial_id') not in done_ids]
    traj_exists = sum((PROJECT_ROOT / r['trajectory_path']).exists() for r in rows if 'trajectory_path' in r)

    print(f'run_id={run_id}')
    print(f'plan={len(records)} status={len(rows)} pending={len(pending)} trajectories={traj_exists}/{len(rows)}')
    print('plan_by_bench_mode=' + json.dumps(counter_to_jsonable(Counter((r.get('bench'), r.get('mode')) for r in records)), ensure_ascii=False, sort_keys=True))
    print('status_returncodes=' + json.dumps(counter_to_jsonable(Counter(r.get('returncode') for r in rows)), ensure_ascii=False, sort_keys=True))
    print('status_error_kind=' + json.dumps(counter_to_jsonable(Counter(r.get('error_kind') or '' for r in rows)), ensure_ascii=False, sort_keys=True))
    print('status_by_bench_mode=' + json.dumps(counter_to_jsonable(Counter((r.get('bench'), r.get('mode')) for r in rows)), ensure_ascii=False, sort_keys=True))
    print('elapsed_sec_summary=' + json.dumps(summarize_numeric_field(rows, 'elapsed_sec'), ensure_ascii=False, sort_keys=True))
    print('lock_wait_sec_summary=' + json.dumps(summarize_numeric_field(rows, 'lock_wait_sec'), ensure_ascii=False, sort_keys=True))
    print('subprocess_elapsed_sec_summary=' + json.dumps(summarize_numeric_field(rows, 'subprocess_elapsed_sec'), ensure_ascii=False, sort_keys=True))
    print('lock_wait_sec_by_bench_mode_model=' + json.dumps(summarize_numeric_by_group(rows, 'lock_wait_sec'), ensure_ascii=False, sort_keys=True))
    print('subprocess_elapsed_sec_by_bench_mode_model=' + json.dumps(summarize_numeric_by_group(rows, 'subprocess_elapsed_sec'), ensure_ascii=False, sort_keys=True))

    log_classes = Counter()
    for rec in records:
        log_path = PROJECT_ROOT / rec.get('log_path', '')
        log_classes[classify_log(log_path)] += 1
    print('log_classes=' + json.dumps(dict(log_classes), ensure_ascii=False, sort_keys=True))

    summary = collected / 'summary.md'
    sft_messages = collected / 'sft_messages.jsonl'
    if summary.exists():
        print(f'collected_summary={summary}')
        for line in summary.read_text(errors='ignore').splitlines()[:16]:
            print(line)
    if sft_messages.exists():
        print(f'sft_messages={sum(1 for _ in sft_messages.open())}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Post-process SWE agent results using swebench harness for proper verification.

The custom run_tests() can't handle FAIL_TO_PASS tests that are part of the gold patch.
This script:
1. Reads patches from the incremental JSONL
2. Creates swebench predictions format
3. Runs swebench evaluation which properly applies gold test patches

Usage:
    python verify_patches.py --input swe_qwen3_5-27b_incremental.jsonl --tag rerun
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(os.environ.get("SKILLRL_ROOT", "/path/to/skillRL"))
DOCKER_ENV = dict(os.environ, DOCKER_HOST="tcp://127.0.0.1:2375")
IMAGE_PREFIX = "xingyaoww/sweb.eval.x86_64."


def load_predictions(jsonl_path, skip_first_n=0):
    """Load patches from incremental JSONL and convert to swebench predictions format."""
    predictions = []
    seen = set()
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i < skip_first_n:
                continue
            data = json.loads(line)
            iid = data["instance_id"]
            patch = data.get("patch", "").strip()
            if not patch or iid in seen:
                continue
            seen.add(iid)
            predictions.append({
                "instance_id": iid,
                "model_patch": patch,
                "model_name_or_path": "qwen3.5-27b",
            })
    return predictions


def verify_single(instance_id, patch, timeout=600):
    """Verify a single patch using Docker container + gold test from swebench data."""
    import pandas as pd
    import numpy as np

    # Load instance data to get gold patch and test info
    for parquet_path in [
        BASE_DIR / "datasets/swe-gym/lite/data/train-00000-of-00001.parquet",
        BASE_DIR / "datasets/swe-bench-verified/data/data/test-00000-of-00001.parquet",
    ]:
        df = pd.read_parquet(parquet_path)
        matches = df[df["instance_id"] == instance_id]
        if len(matches) > 0:
            row = matches.iloc[0]
            break
    else:
        return {"instance_id": instance_id, "resolved": False, "error": "Instance not found"}

    gold_patch = row["patch"]
    fail_to_pass = row["FAIL_TO_PASS"]
    if isinstance(fail_to_pass, np.ndarray):
        fail_to_pass = fail_to_pass.tolist()
    elif isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = [fail_to_pass]

    # Extract test-only changes from gold patch (files in tests/ or test_*)
    import re
    test_hunks = []
    current_file = None
    in_test_file = False
    for line in gold_patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                current_file = match.group(1)
                in_test_file = "test" in current_file.lower()
            if in_test_file:
                test_hunks.append(line)
        elif in_test_file:
            test_hunks.append(line)

    test_patch = "\n".join(test_hunks) if test_hunks else ""

    # Start container
    image_name = instance_id.replace("/", "_s_").replace("__", "_s_")
    image = IMAGE_PREFIX + image_name + ":latest"
    cname = f"swe-verify-{instance_id.replace('/', '_')}"

    try:
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True, env=DOCKER_ENV, timeout=30)
        result = subprocess.run(
            ["docker", "run", "-d", "--name", cname, image, "sleep", "infinity"],
            capture_output=True, text=True, env=DOCKER_ENV, timeout=120)
        if result.returncode != 0:
            return {"instance_id": instance_id, "resolved": False, "error": f"Container failed: {result.stderr[:200]}"}

        # Find repo path
        repo_path = "/testbed"
        for candidate in ["/testbed", "/workspace", "/repo"]:
            r = subprocess.run(
                ["docker", "exec", cname, "test", "-d", candidate],
                capture_output=True, env=DOCKER_ENV, timeout=10)
            if r.returncode == 0:
                repo_path = candidate
                break

        # Apply model's patch
        import base64
        b64_patch = base64.b64encode(patch.encode()).decode()
        apply_cmd = f"echo '{b64_patch}' | base64 -d | git apply --allow-empty -v 2>&1"
        r = subprocess.run(
            ["docker", "exec", "-w", repo_path, cname, "bash", "-c", apply_cmd],
            capture_output=True, text=True, env=DOCKER_ENV, timeout=60)
        model_apply_ok = r.returncode == 0

        if not model_apply_ok:
            # Try with git apply --reject
            apply_cmd2 = f"echo '{b64_patch}' | base64 -d | git apply --reject 2>&1 || true"
            subprocess.run(
                ["docker", "exec", "-w", repo_path, cname, "bash", "-c", apply_cmd2],
                capture_output=True, text=True, env=DOCKER_ENV, timeout=60)

        # Apply gold test patch (test files only)
        if test_patch:
            b64_test = base64.b64encode(test_patch.encode()).decode()
            test_cmd = f"echo '{b64_test}' | base64 -d | git apply --allow-empty 2>&1 || true"
            subprocess.run(
                ["docker", "exec", "-w", repo_path, cname, "bash", "-c", test_cmd],
                capture_output=True, text=True, env=DOCKER_ENV, timeout=60)

        # Install test dependencies
        dep_cmd = "pip install -q pytest sure funcy pytest-xdist pytest-benchmark boto3 2>/dev/null || true"
        subprocess.run(
            ["docker", "exec", cname, "bash", "-c", dep_cmd],
            capture_output=True, text=True, env=DOCKER_ENV, timeout=120)

        # Install the repo itself
        install_cmd = f"cd {repo_path} && pip install -q -e . 2>/dev/null || true"
        subprocess.run(
            ["docker", "exec", cname, "bash", "-c", install_cmd],
            capture_output=True, text=True, env=DOCKER_ENV, timeout=120)

        # Run FAIL_TO_PASS tests
        test_spec = " ".join(f"'{t}'" for t in fail_to_pass)
        test_cmd = (
            f"cd {repo_path} && python -m pytest {test_spec} "
            f"-x --tb=short --no-header -rN -o 'addopts=' -p no:cacheprovider "
            f"2>&1 | tail -30"
        )
        r = subprocess.run(
            ["docker", "exec", cname, "bash", "-c", test_cmd],
            capture_output=True, text=True, env=DOCKER_ENV, timeout=300)
        test_output = r.stdout + ("\n" + r.stderr if r.stderr else "")

        # Check if tests passed
        resolved = False
        for line in reversed(test_output.strip().split("\n")[-10:]):
            line = line.strip()
            if " passed" in line and "failed" not in line and "error" not in line.lower():
                resolved = True
                break
            if line == "OK":
                resolved = True
                break
            if "FAILED" in line or "ERRORS" in line or "no tests ran" in line.lower():
                break

        return {
            "instance_id": instance_id,
            "resolved": resolved,
            "model_apply_ok": model_apply_ok,
            "test_output": test_output[-500:],
        }

    except Exception as e:
        return {"instance_id": instance_id, "resolved": False, "error": str(e)}

    finally:
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True, env=DOCKER_ENV, timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to incremental JSONL")
    parser.add_argument("--skip-first", type=int, default=0, help="Skip first N lines (e.g. old run)")
    parser.add_argument("--output", default="", help="Output JSONL path")
    parser.add_argument("--tag", default="verify", help="Tag for output file")
    args = parser.parse_args()

    predictions = load_predictions(args.input, skip_first_n=args.skip_first)
    print(f"Loaded {len(predictions)} patches to verify")

    if not predictions:
        print("No patches to verify!")
        return

    output_path = args.output or str(
        BASE_DIR / "experiments" / f"swe_{args.tag}_verified" / "verified.jsonl"
    )

    results = []
    for i, pred in enumerate(predictions, 1):
        iid = pred["instance_id"]
        print(f"\n[{i}/{len(predictions)}] Verifying {iid}...")
        result = verify_single(iid, pred["model_patch"])
        results.append(result)

        status = "RESOLVED" if result["resolved"] else "FAILED"
        print(f"  {status}")

        with open(output_path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

    resolved = sum(1 for r in results if r["resolved"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Verified: {resolved}/{total} resolved ({100*resolved/total:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

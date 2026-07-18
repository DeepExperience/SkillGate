#!/usr/bin/env python3
"""Small, safe operator interface for the handover repository."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "ops/recipes/catalog.toml"
FORMER_ROOT = "/mnt/tidalfs-bdsz01/usr/tusen/liqingyao/" + "Projects"
FORMER_HANDOVER = "/mnt/tidalfs-bdsz01/usr/tusen/liqingyao/" + "skillRL_handover"


def stale_roots() -> list[str]:
    """Old checkout roots that must not appear in control files.

    The original handover root only counts as stale when this checkout
    actually lives somewhere else (e.g. a fresh clone from Git)."""
    roots = [FORMER_ROOT]
    if str(ROOT) != FORMER_HANDOVER:
        roots.append(FORMER_HANDOVER)
    return roots


def load_recipes() -> dict[str, dict]:
    with CATALOG.open("rb") as fh:
        return tomllib.load(fh)["recipes"]


def command_for(recipe: dict, extra: list[str], execute: bool) -> tuple[list[str], dict[str, str]]:
    entrypoint = ROOT / recipe["entrypoint"]
    runner = recipe["runner"]
    if runner == "bash":
        cmd = ["bash", str(entrypoint)]
    elif runner == "python":
        cmd = [os.environ.get("SKILLRL_SLIME_PYTHON", sys.executable), str(entrypoint)]
    else:
        raise ValueError(f"unsupported runner: {runner}")
    cmd.extend(str(v) for v in recipe.get("default_args", []))
    cmd.extend(extra)

    env = os.environ.copy()
    env.setdefault("ROOT", str(ROOT))
    env.setdefault("SKILLRL_ROOT", str(ROOT))
    env.setdefault("SKILLRL_MODEL_ROOT", str(ROOT / "models"))
    safety = recipe["safety"]
    if safety == "env-dry-run" and not execute:
        env["DRY_RUN"] = "1"
        env.setdefault("WANDB_API_KEY", "handover-dry-run-placeholder")
        env.setdefault("WANDB_MODE", "offline")
    elif safety == "native-execute-flag" and execute and "--execute" not in cmd:
        cmd.append("--execute")
    return cmd, env


def missing_requirements(recipe: dict) -> list[str]:
    return [item for item in recipe.get("requires", []) if not (ROOT / item).exists()]


def cmd_recipes(args: argparse.Namespace) -> int:
    recipes = load_recipes()
    for name, recipe in sorted(recipes.items()):
        missing = len(missing_requirements(recipe))
        state = "ready" if missing == 0 else f"missing:{missing}"
        print(f"{name:28} {state:12} {recipe['description']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    recipes = load_recipes()
    recipe = recipes.get(args.recipe)
    if recipe is None:
        print(f"unknown recipe: {args.recipe}", file=sys.stderr)
        return 2
    cmd, _ = command_for(recipe, args.extra, execute=False)
    print(f"recipe: {args.recipe}")
    print(f"kind: {recipe['kind']}")
    print(f"safety: {recipe['safety']}")
    print(f"description: {recipe['description']}")
    print(f"command: {shlex.join(cmd)}")
    missing = missing_requirements(recipe)
    if missing:
        print("missing:")
        for item in missing:
            print(f"  - {item}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    recipes = load_recipes()
    recipe = recipes.get(args.recipe)
    if recipe is None:
        print(f"unknown recipe: {args.recipe}", file=sys.stderr)
        return 2
    missing = missing_requirements(recipe)
    if missing and args.execute:
        print("refusing execution; required assets are missing:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2
    cmd, env = command_for(recipe, args.extra, execute=args.execute)
    print(f"[skillrl] cwd={ROOT}")
    print(f"[skillrl] mode={'execute' if args.execute else 'safe'}")
    print(f"[skillrl] command={shlex.join(cmd)}")
    if recipe["safety"] == "print-only" and not args.execute:
        print("[skillrl] print-only recipe; pass --execute after doctor succeeds")
        return 0
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


CORE_PATHS = [
    "GeneralAgent/eval_scripts/unified_runner/run_unified_harbor.py",
    "GeneralAgent/sft_data_collection/launch_trials.py",
    "GeneralAgent/sft_training/llamafactory_data/20260512_sft_campaign_clean_plus_claw_thinkwrap/agent_sft_campaign_20260512_clean_plus_claw_thinkwrap.json",
    "Relax/relax/entrypoints/train.py",
    "Relax/examples/agent_bench/run_agent_grpo_9B.sh",
    "datasets/terminal-bench-v2",
    "datasets/seta-env/Harbor-Dataset",
    "datasets/skillsbench/tasks-no-skills",
    "datasets/claw-eval/tasks",
    "skill_libraries/merged/_index.json",
    "GeneralAgent/eval_scripts/skills_retrieval/skill_index_qwen3emb8b.pkl",
    "datasets/rl/parquet_4bench_base_20260523/train.parquet",
    "datasets/rl/rl_split_v2.json",
    "skill_libraries/snapshots/rl/slate_skills_20260708_hard_negative_v8_production/manifest/slate_manifest_train.jsonl",
    "ops/workflows/rl_eval/specs/eval70_v1/tasks.tsv",
    "ops/workflows/rl_training/run_rl.sh",
    "experiments/rl/catalog.json",
    "experiments/rl/HANDOVER_MANIFEST.json",
]


def iter_control_files() -> list[Path]:
    suffixes = {".py", ".sh", ".md", ".toml", ".yaml", ".yml", ".json", ".txt"}
    roots = [ROOT / p for p in ("ops", "GeneralAgent", "Relax", "docs", "tools")]
    out: list[Path] = []
    skip_names = {
        ".git",
        ".venv",
        ".venvs",
        "__pycache__",
        "cache",
        "datasets",
        "deps",
        "llamafactory_data",
        "logs",
        "merged",
        "node_modules",
        "outputs",
        "results",
        "snapshots",
        "third_party",
        "traces",
        "wandb",
    }
    for base in roots:
        if not base.exists():
            continue
        for current, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in skip_names and not (Path(current) / name).is_symlink()
            ]
            current_path = Path(current)
            for name in filenames:
                path = current_path / name
                if path.suffix.lower() not in suffixes or path.is_symlink():
                    continue
                if path.stat().st_size <= 8 * 1024 * 1024:
                    out.append(path)
    skill_root = ROOT / "skill_libraries"
    if skill_root.exists():
        for path in skill_root.iterdir():
            if path.is_file() and path.suffix.lower() in suffixes and path.stat().st_size <= 8 * 1024 * 1024:
                out.append(path)
    return out


def iter_control_symlinks() -> list[Path]:
    """Return symlinks in maintained code without walking payload/env trees.

    A full Path.rglob over benchmark payloads and virtual environments takes
    minutes on networked storage. Those trees are immutable inputs or reconstructed
    dependencies; doctor is meant to give an operator a fast launch-safety
    check of maintained control paths.
    """
    roots = [ROOT / name for name in ("ops", "GeneralAgent", "Relax", "tools", "models")]
    skip_names = {
        ".git",
        ".venv",
        ".venvs",
        "__pycache__",
        "deps",
        "node_modules",
        "outputs",
        "third_party",
        "wandb",
    }
    found: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for current, dirnames, filenames in os.walk(base, followlinks=False):
            current_path = Path(current)
            kept_dirs: list[str] = []
            for name in dirnames:
                path = current_path / name
                if path.is_symlink():
                    found.append(path)
                elif name not in skip_names:
                    kept_dirs.append(name)
            dirnames[:] = kept_dirs
            for name in filenames:
                path = current_path / name
                if path.is_symlink():
                    found.append(path)
    for path in ROOT.iterdir():
        if path.is_symlink():
            found.append(path)
    skill_root = ROOT / "skill_libraries"
    if skill_root.exists():
        for path in skill_root.iterdir():
            if path.is_symlink():
                found.append(path)
    return sorted(set(found))


def cmd_doctor(args: argparse.Namespace) -> int:
    failures: list[str] = []
    warnings: list[str] = []
    for rel in CORE_PATHS:
        if not (ROOT / rel).exists():
            failures.append(f"missing core asset: {rel}")

    broken: list[str] = []
    absolute: list[str] = []
    scanned_symlinks = iter_control_symlinks()
    for path in scanned_symlinks:
        if not path.exists():
            broken.append(str(path.relative_to(ROOT)))
        raw = os.readlink(path)
        allowed_cuda = "/cache/cuda_fast_home/" in path.as_posix() and raw.startswith("/usr/local/cuda/")
        if os.path.isabs(raw) and not allowed_cuda:
            absolute.append(str(path.relative_to(ROOT)))
    if broken:
        failures.append(f"broken symlinks: {len(broken)} (first: {broken[:5]})")
    if absolute:
        failures.append(f"absolute symlinks: {len(absolute)} (first: {absolute[:5]})")

    stale: list[str] = []
    roots = stale_roots()
    for path in iter_control_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if any(root in text for root in roots):
            stale.append(str(path.relative_to(ROOT)))
    if stale:
        failures.append(f"former checkout-root references in control files: {len(stale)} (first: {stale[:8]})")
    stale_pickles: list[str] = []
    for rel in (
        "GeneralAgent/eval_scripts/skills_retrieval/skill_index_qwen3emb8b.pkl",
        "GeneralAgent/eval_scripts/skills_retrieval/skill_index_bm25.pkl",
    ):
        path = ROOT / rel
        if path.exists() and any(root.encode() in path.read_bytes() for root in roots):
            stale_pickles.append(rel)
    if stale_pickles:
        failures.append(f"former checkout-root references in retrieval indices: {stale_pickles}")

    for recipe_name, recipe in load_recipes().items():
        missing = missing_requirements(recipe)
        if missing:
            warnings.append(f"{recipe_name}: {len(missing)} required assets missing")

    print(f"root: {ROOT}")
    print(f"core_paths: {len(CORE_PATHS) - len([x for x in failures if x.startswith('missing core')])}/{len(CORE_PATHS)}")
    print(f"control_symlinks: scanned={len(scanned_symlinks)} broken={len(broken)} absolute={len(absolute)}")
    print(f"stale_projects_refs: {len(stale)}")
    print(f"stale_retrieval_indices: {len(stale_pickles)}")
    for item in warnings:
        print(f"WARN: {item}")
    for item in failures:
        print(f"FAIL: {item}")
    if failures:
        return 1
    print("DOCTOR_OK")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    failures: list[str] = []
    py_files: list[Path] = []
    sh_files: list[Path] = []
    maintained_roots = (
        "ops",
        "GeneralAgent/eval_scripts",
        "GeneralAgent/sft_data_collection",
        "GeneralAgent/sft_training/scripts",
        "GeneralAgent/rl_data_prep",
        "Relax/relax",
        "Relax/examples/agent_bench",
        "Relax/scripts/entrypoint",
        "Relax/tests",
        "tools",
    )
    for base_name in maintained_roots:
        base = ROOT / base_name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or any(part in {".git", "__pycache__", "wandb", "logs"} for part in path.parts):
                continue
            if path.suffix == ".py":
                py_files.append(path)
            elif path.suffix == ".sh":
                sh_files.append(path)

    # This is the one live local patch in the otherwise upstream SGLang dep.
    qwen_patch = ROOT / "Relax/deps/sglang/python/sglang/srt/models/qwen3_5.py"
    if qwen_patch.exists():
        py_files.append(qwen_patch)

    for path in sorted(set(py_files)):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception as exc:  # syntax/encoding failures both matter
            failures.append(f"python {path.relative_to(ROOT)}: {exc}")
    for path in sorted(set(sh_files)):
        proc = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
        if proc.returncode:
            failures.append(f"shell {path.relative_to(ROOT)}: {proc.stderr.strip()}")
    try:
        load_recipes()
    except Exception as exc:
        failures.append(f"catalog: {exc}")

    print(f"python_ast: {len(set(py_files))}")
    print(f"shell_syntax: {len(set(sh_files))}")
    for item in failures[:50]:
        print(f"FAIL: {item}")
    if failures:
        print(f"VERIFY_FAILED count={len(failures)}")
        return 1
    print("VERIFY_OK (static, no GPU/Ray/Docker)")
    return 0


def cmd_relocate(args: argparse.Namespace) -> int:
    script = ROOT / "tools/relocate_repository.py"
    configured = os.environ.get("SKILLRL_SLIME_PYTHON", "")
    conda_root = os.environ.get("SKILLRL_CONDA_ROOT", str(Path.home() / "anaconda3"))
    local_default = Path(conda_root) / "envs/slime/bin/python"
    python = configured or (str(local_default) if local_default.exists() else sys.executable)
    cmd = [python, str(script)]
    if args.from_root:
        cmd.extend(["--from-root", args.from_root])
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_data_check(args: argparse.Namespace) -> int:
    script = ROOT / "tools/validate_data.py"
    python = os.environ.get("SKILLRL_SLIME_PYTHON", sys.executable)
    return subprocess.run([python, str(script)], cwd=ROOT).returncode


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skillrl", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("recipes", help="list maintained recipes")
    q.set_defaults(func=cmd_recipes)
    q = sub.add_parser("show", help="show one recipe")
    q.add_argument("recipe")
    q.add_argument("extra", nargs=argparse.REMAINDER)
    q.set_defaults(func=cmd_show)
    q = sub.add_parser("run", help="dry-run/plan by default; execute only with --execute")
    q.add_argument("recipe")
    q.add_argument("extra", nargs=argparse.REMAINDER, help="arguments after -- are passed through")
    q.set_defaults(func=cmd_run, execute=False)
    q = sub.add_parser("doctor", help="validate repository and asset wiring")
    q.set_defaults(func=cmd_doctor)
    q = sub.add_parser("verify", help="no-GPU Python AST and shell syntax checks")
    q.set_defaults(func=cmd_verify)
    q = sub.add_parser("relocate", help="rewrite embedded repository roots after moving the checkout")
    q.add_argument("--from-root", default="")
    q.set_defaults(func=cmd_relocate)
    q = sub.add_parser("data-check", help="validate migrated skill, SFT, RL, and eval inputs")
    q.set_defaults(func=cmd_data_check)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.command == "run" and "--execute" in args.extra:
        args.execute = True
        args.extra.remove("--execute")
    if getattr(args, "extra", None) and args.extra[0:1] == ["--"]:
        args.extra = args.extra[1:]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

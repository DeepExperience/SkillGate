"""Re-export points from ``sft_data_collection`` / ``eval_scripts``.

Centralising imports here keeps the RL pipeline a one-way dependency on SFT
helpers — we never modify SFT code, only call into it. If an SFT module
relocates, only this file needs to change.
"""
from __future__ import annotations

import sys
from pathlib import Path


# Make sibling project modules importable when this package is invoked as
# ``python -m GeneralAgent.rl_data_prep.<x>`` from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SFT_DATA_DIR = _PROJECT_ROOT / "GeneralAgent" / "sft_data_collection"
for _p in (_PROJECT_ROOT, _SFT_DATA_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# --- from sft_data_collection.build_splits ----------------------------------
# ``sft_data_collection/build_splits.py`` uses bare ``from common import ...``
# (its sibling ``common.py``), so we add the directory to sys.path above
# before importing.
# ``split_even`` produces a deterministic train/holdout split from a sorted
# task list via even-spaced index picking, mirroring SFT split logic exactly.
from build_splits import split_even  # type: ignore  # noqa: E402


__all__ = [
    "split_even",
]

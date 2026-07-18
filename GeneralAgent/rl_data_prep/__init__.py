"""RL data preparation utilities.

This package builds RL train/eval splits, converts task lists to Relax parquet
format, and pre-generates retrieval caches. It is strictly separated from
``sft_data_collection/`` and ``sft_training/`` — any SFT helper functions are
imported (never modified in place) via ``_imports.py``.
"""

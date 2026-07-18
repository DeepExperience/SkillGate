#!/usr/bin/env bash
# NOTE: Canonical workflow wrapper for RL split/parquet preparation.
# Maintained here so data-prep orchestration is centralized while Python logic stays in GeneralAgent/rl_data_prep.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
cd "${ROOT}"
PY=${PY:-python3}
${PY} GeneralAgent/rl_data_prep/build_rl_split_v2.py "$@"
${PY} GeneralAgent/rl_data_prep/convert_to_relax_data_v2.py

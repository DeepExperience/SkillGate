#!/bin/bash
# One-shot: prebake SWE-Gym-lite 100 + SETA synth_data 300 image cache.
#
# Safe-mode default: --dry-run prints plan + cost estimate without touching anything.
# Real run: strip --dry-run. Hard limit via sequential order (SWE first, SETA second)
# so a clash-quota break stops SETA before it also starts.
#
# Usage:
#   bash prebake_all.sh --dry-run       # see plan + traffic estimate
#   bash prebake_all.sh                 # actually run (SWE first, SETA second)
#   bash prebake_all.sh --swe-only
#   bash prebake_all.sh --seta-only
#
# Each step is idempotent: re-running after partial success only completes remaining items.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DO_SWE=1
DO_SETA=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY_RUN=1; shift ;;
    --swe-only)  DO_SETA=0; shift ;;
    --seta-only) DO_SWE=0; shift ;;
    *)           echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Sanity: lists exist
if [[ $DO_SWE -eq 1 && ! -f "$SCRIPT_DIR/swe_lite_100_images.txt" ]]; then
  echo "[gen] generating swe_lite_100 list..."
  python3 "$SCRIPT_DIR/select_swe_lite_100.py"
fi
if [[ $DO_SETA -eq 1 && ! -f "$SCRIPT_DIR/seta_300.txt" ]]; then
  echo "[gen] generating seta_300 list..."
  python3 "$SCRIPT_DIR/select_seta_300.py"
fi

echo "================================================================"
echo "  PREBAKE: SWE-lite 100 + SETA synth 300"
echo "================================================================"

# ---- Stage 1: SWE ----
if [[ $DO_SWE -eq 1 ]]; then
  echo
  echo "=== Stage 1/2: SWE-Gym-lite 100 images (docker pull) ==="
  if [[ $DRY_RUN -eq 1 ]]; then
    bash "$SCRIPT_DIR/prebake_swe_lite_100.sh" --dry-run
  else
    bash "$SCRIPT_DIR/prebake_swe_lite_100.sh"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo
      echo "⚠ SWE prebake exited non-zero ($rc). Stopping before SETA (flip --seta-only after fixing)."
      exit $rc
    fi
  fi
fi

# ---- Stage 2: SETA ----
if [[ $DO_SETA -eq 1 ]]; then
  echo
  echo "=== Stage 2/2: SETA synth_data 300 images (docker build) ==="
  if [[ $DRY_RUN -eq 1 ]]; then
    bash "$SCRIPT_DIR/prebake_seta_300.sh" --dry-run
  else
    bash "$SCRIPT_DIR/prebake_seta_300.sh"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo
      echo "⚠ SETA prebake exited non-zero ($rc). See /tmp/prebake_seta_failures.txt"
      exit $rc
    fi
  fi
fi

echo
echo "================================================================"
echo "  PREBAKE COMPLETE"
echo "================================================================"

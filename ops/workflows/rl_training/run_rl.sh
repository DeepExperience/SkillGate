#!/usr/bin/env bash
# Canonical RL entrypoint. One scientific experiment owns all of its restart
# segments, model exports, and evals under experiments/rl/runs/EXPERIMENT_ID.
set -Eeuo pipefail

ROOT="${ROOT:-/path/to/skillRL}"
export ROOT
cd "${ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash ops/workflows/rl_training/run_rl.sh PROFILE [--dry-run]

Profiles:
  no_skill         No advertised/injected skills; standard task-reward GRPO.
  mixed_task_reward 16 mixed skills; standard task-reward-only GRPO control.
  mixed_separated  16 mixed skills; separate outcome-stratified selector advantage.
  selector_action_credit 16 mixed skills; token-local oracle-vs-distractor selector credit.
  selector_clean_oracle_action_credit 16 V8-production skills; exactly-one-oracle-read token-local selector credit.
  hybrid_slate     Paired no-skill/mixed hybrid slate regret + stratified advantage.

For a first launch, EXPERIMENT_ID and RUN_NAME are generated automatically.
Resume by reusing EXPERIMENT_ID and setting LOAD_DIR, EXPECTED_LATEST_CKPT,
START_ROLLOUT_ID, and a new RUN_NAME (RUN_NAME is the segment id).
EOF
}

RL_PROFILE="${1:-}"
[[ -n "${RL_PROFILE}" ]] || { usage; exit 2; }
shift
case "${1:-}" in
  --dry-run) export DRY_RUN=1; shift ;;
  '') ;;
  *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
esac
[[ $# -eq 0 ]] || { echo "Unexpected arguments: $*" >&2; exit 2; }

export RL_LAUNCH_STAMP="${RL_LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
if [[ -n "${EXPERIMENT_ID+x}" ]]; then
  export RL_EXPERIMENT_ID_WAS_EXPLICIT=1
else
  export RL_EXPERIMENT_ID_WAS_EXPLICIT=0
fi

RUNTIME="${ROOT}/ops/workflows/rl_training/lib/runtime.sh"
PROFILE_FILE="${ROOT}/ops/workflows/rl_training/profiles/${RL_PROFILE}.sh"
[[ -f "${PROFILE_FILE}" ]] || { echo "Unknown RL profile: ${RL_PROFILE}" >&2; usage; exit 2; }

# shellcheck source=lib/runtime.sh
source "${RUNTIME}"
rl_reset_algorithm_env
# shellcheck source=/dev/null
source "${PROFILE_FILE}"
rl_profile_configure
rl_configure_identity
rl_apply_common_defaults
rl_profile_prepare
if declare -F rl_profile_validate_algorithm >/dev/null; then
  rl_profile_validate_algorithm
fi
rl_validate_common_config
rl_resolve_nodes

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  rl_dump_resolved_config
  echo "DRY_RUN_OK profile=${RL_PROFILE}; direct delegate=Relax/examples/agent_bench/run_agent_grpo_9B.sh"
  exit 0
fi

rl_load_wandb_credentials
rl_init_run_dir
# Capture validation, preflight, entrypoint preparation, guard startup, and
# training in the segment's single canonical log.  This survives detached tmux
# exit and makes failed infrastructure retries diagnosable.
exec > >(tee -a "${RUN_DIR}/driver.log") 2>&1
RL_MANIFEST_RECORDED=0
RL_MANIFEST_FINALIZED=0
rl_record_launch
rl_exit_handler() {
  local rc=$?
  set +e
  rl_stop_guards
  rl_record_finish "${rc}"
}
trap rl_exit_handler EXIT
set +e
rl_preflight
preflight_rc=$?
set -e
if (( preflight_rc != 0 )); then
  echo "FATAL: RL preflight failed with rc=${preflight_rc}" >&2
  exit "${preflight_rc}"
fi
rl_prepare_relax_entrypoint
rl_start_guards
set +e
rl_launch_training
rc=$?
if (( rc == 0 )); then
  rl_verify_training_completion
  completion_rc=$?
  if (( completion_rc != 0 )); then
    rc=${completion_rc}
  fi
fi
set -e
rl_record_finish "${rc}"
exit "${rc}"

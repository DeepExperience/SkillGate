# SkillGate

**SkillGate** is a research codebase for training LLM agents to *decide when to
read, which skill to trust, and how to act* when a library of textual skills
(procedural instructions) is available in long-horizon coding/terminal tasks.
It bundles a unified 4-benchmark agent environment, an SFT data pipeline, a
GRPO-based RL stack, and a frozen evaluation protocol.

The flagship method — **SkillGate** — is a token-local
credit-assignment scheme for skill selection:

> In long-horizon agentic RL, a trailing behavior reward attached to the final
> outcome almost never reaches the few tokens where the agent *chose* a skill:
> selection tokens are 2-3% of the trajectory, so outcome-level advantage is
> diluted 30-50x and the cheapest policy is to stop reading skills at all.
> SkillGate decouples the two credit streams: ordinary outcome GRPO drives the
> task-execution tokens, while a separate, token-local advantage — computed
> from clean oracle-vs-distractor selection evidence within each rollout group
> — is applied only on the skill-selection spans (the `<skill_reasoning>`
> block and the read calls). Reading the right skill gets credited at the
> decision point, not diffused over thousands of action tokens.

Method implementation (all paths are in this repository):

| Piece | File |
|---|---|
| **Clean-oracle utility + reward post-processing (the final method)** | `Relax/examples/agent_bench/selector_clean_oracle_action_credit.py` |
| Selection-span detection + token-local selector advantage | `Relax/examples/agent_bench/selector_action_credit.py` |
| Custom GRPO loss combining task + selector credit streams | `Relax/examples/agent_bench/selector_action_grpo_loss.py` |
| **Training profile of the final method** | `ops/workflows/rl_training/profiles/selector_clean_oracle_action_credit.sh` |
| Training profile of the earlier non-clean variant | `ops/workflows/rl_training/profiles/selector_action_credit.sh` |
| CPU smoke test of the credit math | `ops/workflows/rl_training/tools/smoke_selector_clean_oracle_action_credit.py` |

Key env knobs: `RELAX_SELECTOR_ACTION_CREDIT=1`,
`RELAX_SELECTOR_ACTION_LOSS_COEF` (see the profile for the full set).

**Which variant is the paper's method.** The final SkillGate results use the
*clean-oracle* utility: a read action earns positive utility only when the whole
trajectory contains exactly one attributed skill read *and* that read is the
oracle. An earlier variant credited the first oracle read even when the agent
kept reading other skills; it is retained as `selector_action_credit` and reported
as an ablation, not as the method.

## Repository map

    GeneralAgent/                 task runners (unified 9-tool OpenClaw-style interface),
                                  SFT collection, data conversion
    Relax/                        RL framework (vendored) + examples/agent_bench (all methods)
    Megatron-LM/, sglang/, slime/ pinned training/serving dependencies (see THIRD_PARTY.md)
    ops/
      workflows/
        sft_data_collection/      trajectory collection entrypoints
        sft_training/             LLaMA-Factory LoRA + merge entrypoints
        rl_data_prep/             deterministic RL parquet builders
        rl_training/              run_rl.sh + 5 maintained profiles
        rl_eval/                  frozen eval70 protocol + owner-aware wrapper
      launch/ monitor/ cleanup/   infrastructure bring-up, watchdogs, targeted cleaners
      recipes/catalog.toml        operator-facing recipe catalog (`./skillrl`)
    datasets/                     benchmark payloads + canonical RL parquet (asset bundle)
    skill_libraries/              merged skill library + frozen RL skill slates (asset bundle)
    env/                          environment reconstruction (3 Python envs)
    docs/OPERATIONS_GUIDE.md      complete operations runbook (Chinese)
    docs/EXPERIMENT_AND_INFRA_HISTORY.md
                                  experiment history: what failed, why, and the
                                  evidence chain that led to SkillGate (Chinese)
    THIRD_PARTY.md                vendored-component attribution and licenses

Naming note: the operator CLI (`./skillrl`) and the `SKILLRL_*` environment
variables keep this workspace's historical short name for "skill-RL
pipeline"; the method and repository name is **SkillGate**.

## Quickstart

### 0. Clone + assets

Code lives in Git; heavyweight inputs are delivered as a side asset bundle
(verified by `assets/migrated-assets.json`) plus public HF weights
(`assets/external-assets.toml`):

    models/Qwen3.5-9B            # base model  (huggingface.co/Qwen)
    models/Qwen3.5-27B           # teacher for SFT collection (optional but recommended)
    datasets/                    # benchmark payloads + RL parquet   (asset bundle)
    skill_libraries/             # merged skills + frozen RL slates  (asset bundle)

Frozen skill slates (the oracle/misleading/relevant/irrelevant inputs of the
mixed-skill experiments) are **frozen deliverables**: derived variants can be
rebuilt with `ops/workflows/rl_data_prep/make_hybrid_v8body_0704desc_slate.py`,
but the original misleading-body generation is not re-runnable from this repo.

### 1. Environments

Three Python environments (never mix their PYTHONPATHs):
`slime` (serving/eval/collection), `relax` (RL training),
`GeneralAgent/.venvs/llamafactory` (SFT training). Reconstruction order,
pins, and patches: `env/README.md`.

### 2. Sanity

    cp .env.example .env         # fill endpoints/keys; never commit real values
    ./skillrl doctor             # asset + wiring check, tells you what is missing
    ./skillrl recipes            # all runnable recipes with missing-input hints
    ./skillrl verify             # no-GPU static checks

If the checkout path differs from the recorded root: `./skillrl relocate`.

### 3. SFT stage (data collection → LoRA → merge)

The RL profiles start from a 9B SFT model that this repo lets you reproduce
end to end (no SFT weights are shipped):

    # (a) collect trajectories (student + teacher endpoints; any
    #     OpenAI-compatible provider works for the teacher):
    bash ops/launch/run_qwen35_sglang_server.sh          # student endpoint
    RUN_ID=$(date -u +%Y%m%d_%H%M)_sft bash ops/workflows/sft_data_collection/run_sft_pipeline.sh

    # (b) or skip collection: the exact final 1708-record dataset is in the
    #     asset bundle, and `tools/rebuild_final_sft.py` re-derives it
    #     byte-identically from intermediate assets (no GPU needed).

    # (c) train + merge:
    ./skillrl show sft.final-9b
    ./skillrl run sft.final-9b --execute

### 4. RL stage

    bash ops/workflows/rl_training/run_rl.sh selector_action_credit --dry-run
    ./skillrl run rl.selector-action-credit              # safe-mode validation
    ./skillrl run rl.selector-action-credit --execute    # real training

Maintained profiles (all share the same 4-benchmark GRPO base recipe):

| Profile | Role |
|---|---|
| `no_skill` | no-skill GRPO baseline |
| `mixed_task_reward` | 16-skill mixed slate, outcome-only reward (control) |
| `mixed_separated` | outcome-stratified separated behavior advantage (ablation) |
| `hybrid_slate` | paired regret + stratified behavior advantage (ablation) |
| `selector_action_credit` | non-clean token-local selector credit (ablation) |
| `selector_clean_oracle_masked_task_only` | task mask only, selector coefficient exactly 0 (ablation) |
| **`selector_clean_oracle_action_credit`** | **SkillGate — clean-oracle token-local selector credit (main method)** |

Two supervised selection baselines share the same direct-SFT9B init and are built
by `GeneralAgent/rl_data_prep/build_selector_bc_dpo_data.py`, then trained through
`ops/workflows/sft_training/run_skillgate_selection_baseline.sh`:

| Baseline | What it learns |
|---|---|
| Gold Selector BC | teacher-forces the first-turn `read(gold)` on the training tasks; no negative candidates, no outcome signal |
| SelSkill-style DPO | `read(gold) > read(misleading)` preference pairs (each gold paired with 5 misleading); a documented adaptation of [SelSkill](https://arxiv.org/abs/2606.00510) to a 16-way slate, not a reproduction of its entropy-branching pipeline |

Evaluation beyond the frozen eval70 protocol: `ops/workflows/rl_eval/build_eval_claw_147_slate.py`
builds the independent, corrected 147-task Claw slate, and
`ops/workflows/rl_eval/build_skillgate_paper_analysis.py` rebuilds every reported
table and figure from the raw evaluation artifacts.

Init-model note: the SkillGate profile defaults to a mid-training export of an
oracle-GRPO warmup run (`experiments/rl/runs/.../model/exports/...`, manifest
stub only — weights are not distributed). To run without it, point
`MODEL_DIR` at your own SFT merged model from stage 3; the method itself is
init-agnostic. Resume semantics (`EXPERIMENT_ID` / `RUN_NAME` / `LOAD_DIR` /
`EXPECTED_LATEST_CKPT`) are documented in `docs/OPERATIONS_GUIDE.md`.

### 5. Evaluation

The frozen 70-task held-out protocol (eval70: 4 repeats, fixed seed, fixed
prompt/tool schema) lives in `ops/workflows/rl_eval/specs/eval70_v1/`:

    bash ops/workflows/rl_eval/run_eval70_checkpoint_set.sh \
      --group <comparison-name> --skill-mode mixed \
      --snapshot skill_libraries/snapshots/rl/<snapshot> \
      --manifest skill_libraries/snapshots/rl/<snapshot>/manifest/slate_manifest_eval70.jsonl \
      --model <owner-experiment> <row-label> <hf-model-dir> \
      --dry-run

Comparison hygiene (non-negotiable): identical prompt profile, tool schema,
skill snapshot, repeats, seed, Docker mode, and one SGLang process per
comparison group; report task-level pass@k with errors counted in the
denominator.

## Experiment ownership layout

One scientific experiment owns its restart segments, model exports, and every
evaluation of its model:

    experiments/rl/runs/<experiment_id>/
      experiment.json
      segments/<run_name>/          # one per process lifetime / resume
      model/exports/<export_id>/    # atomically-renamed validated exports
      eval/<eval_id>/rows/<row_id>/

`EXPERIMENT_ID` stays stable across resumes; every restart gets a new
`RUN_NAME`; shared datasets and skill snapshots live outside any single run.
`experiments/rl/catalog.json` lists the retained decisive experiments.

## Infrastructure notes

RL rollouts run benchmark containers at high concurrency against a local
overlay2 dockerd. The hard-won operational rules (subreaper-wrapped dockerd,
host-network containers, disk-bomb reapers, teardown red-lines, resume
checklists) are condensed in `docs/OPERATIONS_GUIDE.md` — read it before the
first `--execute`. Never use broad `pkill -f ray` / `ray stop` /
bulk-container deletion on shared machines.

## Validation before contributing

    bash -n <changed-shell-files>
    python3 -m py_compile <changed-python-files>
    ./skillrl doctor && ./skillrl verify

Source belongs in package directories or `ops/`; outputs belong in
`experiments/`. Changes to `Relax/`, evaluator adapters, tool schemas, prompt
compatibility, or data conversion are high-risk and need a reason, a
compatibility note, and a validation record.

## Status, licensing, attribution

- This is the research codebase accompanying a paper in preparation
  (working title **SkillGate**; naming may change at submission time).
- Vendored third-party components and their licenses are listed in
  `THIRD_PARTY.md`. The `Relax/` directory vendors an internal RL framework —
  confirm licensing with the owning team before making this repository
  public.
- No model weights, no credentials, and no proprietary data are distributed
  through Git; heavyweight inputs travel in the verified asset bundle.

# Environment reconstruction

Three Python stacks power the pipeline. Build them separately (Python 3.12
recommended) and never mix their PYTHONPATHs:

| Env | Used for |
|---|---|
| `slime` (conda) | evaluation, SGLang serving, retrieval, SFT collection/export, checkpoint→HF conversion, most `./skillrl` recipes |
| `relax` (conda) | Relax/Megatron RL training (Ray, Transformer Engine, FlashAttention, actor/rollout stack) |
| `GeneralAgent/.venvs/llamafactory` (venv) | LLaMA-Factory SFT LoRA training + merge |

Point `.env` at the interpreters once built (see `.env.example`:
`SKILLRL_SLIME_PYTHON`, `SKILLRL_RELAX_PYTHON`, `SKILLRL_CONDA_ROOT`).

## 1. `slime` env (evaluation / serving / collection)

Install the CUDA/Torch base first (torch 2.9.x + a matching CUDA 12.x stack,
flash-attn, transformer-engine — `freezes/slime_env_pipfreeze_20260407.txt`
records the last known-good full pin set), then install the vendored sources
**in this order, from the repo root**:

```bash
python -m pip install -e sglang/python --no-deps
python -m pip install -e slime --no-deps
python -m pip install -e Megatron-LM --no-deps
python -m pip install -e datasets/claw-eval/src --no-deps
python -c "import sglang, slime, megatron.core, claw_eval"   # verify
```

Notes:
- The vendored `sglang/` and `Megatron-LM/` trees already carry the required
  local patches; do not re-clone upstream into these paths.
- Editable installs bind to absolute paths — if you move the checkout,
  reinstall all editables (and run `./skillrl relocate`).
- When installing `nvidia-modelopt`, always quote the spec
  (`pip install "nvidia-modelopt[torch]>=0.37.0"`); an unquoted `>=` silently
  creates a stray redirect file.

## 2. `relax` env (RL training)

```bash
python -m pip install -e Relax
python -m pip install -r Relax/requirements.txt
python -m pip install -e Relax/deps/sglang/python --no-deps
python -m pip install -e Relax/deps/Megatron-LM --no-deps
# Megatron-Bridge is vendored at Relax/deps/Megatron-Bridge if bridge-mode
# checkpoint export complains about missing megatron.bridge:
python -m pip install -e Relax/deps/Megatron-Bridge --no-deps
```

Notes:
- `Relax/deps/sglang` (not the top-level `sglang/`) is the rollout engine for
  RL; it carries a required `qwen3_5.py` patch. Never point the relax env at
  the top-level `sglang/`.
- Multi-node Ray setups must have identical editable installs of the same
  checkout on every node. Single-node recipes default to
  `${SKILLRL_CONDA_ROOT}/envs/relax/bin/python`.

## 3. LLaMA-Factory venv (SFT training)

```bash
python -m venv --system-site-packages GeneralAgent/.venvs/llamafactory
source GeneralAgent/.venvs/llamafactory/bin/activate
python -m pip install -e GeneralAgent/third_party/LLaMA-Factory
python -m pip install deepspeed "liger-kernel>=0.6.3"
```

Notes:
- `--system-site-packages` on top of the slime env reuses its
  Torch/Transformers/Datasets/Accelerate stack; the venv only adds
  LLaMA-Factory and trainer-side packages.
- `liger-kernel` must be a version providing
  `apply_liger_kernel_to_qwen3_5` (>= 0.6.3); the run script preflights this.
- The vendored LLaMA-Factory already contains the `qwen3_5_nothink` template
  and the torch-2.9 Conv3D loader guard
  (`LLAMAFACTORY_ALLOW_TORCH29_CONV3D=1`, set by the run script).
- Day-to-day entry: `source GeneralAgent/sft_training/activate_llamafactory.sh`.

## 4. Verify

```bash
./skillrl doctor
./skillrl verify
```

`freezes/` holds historical pip freezes for pin archaeology; the vendored
source trees in this repository are the authoritative implementations.

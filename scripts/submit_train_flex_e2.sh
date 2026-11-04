#!/usr/bin/env bash
# E2: Fine-tune Kinya-Flex on Kin pool + W&B logging.
#
# Prereqs:
#   wandb login
#   bash scripts/setup_deepkin.sh
#   python scripts/export_kin_train_datasets.py
#
#   bash scripts/submit_train_flex_e2.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
DEEPKIN_ROOT="${DEEPKIN_ROOT:-$PROJECT_ROOT/code/ac-ai-models/DeepKIN-AgAI}"
FLEX_DATA="${FLEX_DATA:-$PROJECT_ROOT/tts_data_cleaning/kin_flex_data}"
OUT_DIR="${OUT_DIR:-$BASELINES/finetune_e2_flex_kin}"
INIT_CKPT="${INIT_CKPT:-$BASELINES/kinya_flex_tts/kinya_flex_tts_base_trained.pt}"
SAVE_CKPT="${SAVE_CKPT:-$OUT_DIR/kinya_flex_tts_e2.pt}"
WANDB_PROJECT="${WANDB_PROJECT:-healthtts}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-E2-flex-kin}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-e2-flex-kin}"

# Base ckpt stores iter≈1e6; DeepKIN treats --num_iters as absolute end_iter.
# 1_000_000 + 25_000 ≈ 30 ep more fine-tune (do NOT use 25000 alone — job exits instantly).
NUM_ITERS="${NUM_ITERS:-1025000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM="${ACCUM:-2}"
WARMUP="${WARMUP:-1000}"
PEAK_LR="${PEAK_LR:-2e-4}"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Missing init ckpt: $INIT_CKPT"
  exit 1
fi
if [[ ! -f "$FLEX_DATA/tts_train_data.psv" ]]; then
  echo "Missing $FLEX_DATA/tts_train_data.psv"
  echo "Run: python scripts/export_kin_train_datasets.py"
  exit 1
fi
if [[ ! -d "$DEEPKIN_ROOT" ]]; then
  echo "Missing DeepKIN at $DEEPKIN_ROOT — run bash scripts/setup_deepkin.sh"
  exit 1
fi

mkdir -p "$OUT_DIR"
if [[ ! -f "$SAVE_CKPT" ]]; then
  echo "Seeding $SAVE_CKPT from $INIT_CKPT"
  cp -f "$INIT_CKPT" "$SAVE_CKPT"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --time=${SLURM_TIME}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH -J ${SLURM_JOB_NAME}
#SBATCH -o ${OUT_DIR}/train_%j.log

set -euo pipefail
source "\${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate healthtts

export DEEPKIN_ROOT="${DEEPKIN_ROOT}"
# Package is DeepKIN-AgAI/monotonic_align/ (needs core.cpython-311*.so beside __init__.py).
# Do NOT prepend .../monotonic_align — nested dir only has a py312 .so and shadows the package.
export PYTHONPATH="${DEEPKIN_ROOT}:\${PYTHONPATH:-}"
export FLEX_DATA="${FLEX_DATA}"
export SAVE_CKPT="${SAVE_CKPT}"
export NUM_ITERS="${NUM_ITERS}"
export BATCH_SIZE="${BATCH_SIZE}"
export ACCUM="${ACCUM}"
export WARMUP="${WARMUP}"
export PEAK_LR="${PEAK_LR}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME}"
export TMPDIR="/tmp/e2_flex_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR"

cd "${REPO_ROOT}"
pip install -q wandb progressbar2 Cython 2>/dev/null || true
# Build core*.so for *this* Python (3.11). Nested monotonic_align/*.so is py312 — ignore it.
if [[ -d "${DEEPKIN_ROOT}/monotonic_align" ]]; then
  (
    cd "${DEEPKIN_ROOT}/monotonic_align"
    python setup.py build_ext --inplace
    ls -lh core.cpython-*.so
  )
fi
export PYTHONPATH="${DEEPKIN_ROOT}:${PYTHONPATH:-}"
python -c "from monotonic_align.core import maximum_path_c; print('monotonic_align OK')" \
  || { echo "FATAL: monotonic_align.core missing — fix setup before train"; exit 1; }
nvidia-smi || true

python scripts/run_flex_finetune_e2.py \\
  --deepkin_root "${DEEPKIN_ROOT}" \\
  --tts_data_dir "${FLEX_DATA}" \\
  --model_save_path "${SAVE_CKPT}" \\
  --num_iters ${NUM_ITERS} \\
  --batch_size ${BATCH_SIZE} \\
  --accumulation_steps ${ACCUM} \\
  --warmup_iter ${WARMUP} \\
  --peak_lr ${PEAK_LR} \\
  --wandb_project "${WANDB_PROJECT}" \\
  --wandb_run_name "${WANDB_RUN_NAME}"

echo DONE_E2
ls -lh "${SAVE_CKPT}"* || true
EOF

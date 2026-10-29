#!/usr/bin/env bash
# E1: Fine-tune MMS-TTS-Kin on Kin pool only (~30 epochs) + W&B.
#
# Prereqs (once):
#   wandb login
#   bash scripts/setup_mms_finetune.sh
#   python scripts/export_kin_train_datasets.py
#
#   bash scripts/submit_train_mms_e1.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
FT_ROOT="${FT_ROOT:-$PROJECT_ROOT/code/finetune-hf-vits}"
OUT_DIR="${OUT_DIR:-$BASELINES/finetune_e1_mms_kin}"
LOG_DIR="${LOG_DIR:-$OUT_DIR}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/e1_mms_kin_finetune.json}"
WANDB_PROJECT="${WANDB_PROJECT:-healthtts}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-E1-mms-kin}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-8:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-e1-mms-kin}"

INIT_CKPT="${INIT_CKPT:-$BASELINES/mms_tts_kin_train}"
MMS_HF="${MMS_HF:-$PROJECT_ROOT/tts_data_cleaning/kin_mms_hf}"

mkdir -p "$OUT_DIR" "$HF_HOME" "$LOG_DIR"
if ! touch "$OUT_DIR/.write_test" 2>/dev/null; then
  echo "Cannot write to OUT_DIR=$OUT_DIR — /project may be full. df -h $OUT_DIR" >&2
  exit 1
fi
rm -f "$OUT_DIR/.write_test"

missing=0
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  missing=1
fi
if [[ ! -f "$INIT_CKPT/config.json" ]]; then
  echo "Missing trainable MMS Kin ckpt at $INIT_CKPT" >&2
  echo "Run: bash scripts/setup_mms_finetune.sh" >&2
  missing=1
fi
if [[ ! -f "$MMS_HF/dataset_dict.json" ]]; then
  echo "Missing MMS HF dataset at $MMS_HF" >&2
  echo "Run: python scripts/export_kin_train_datasets.py" >&2
  missing=1
fi
if [[ ! -f "$FT_ROOT/run_vits_finetuning.py" ]]; then
  echo "Missing finetune-hf-vits at $FT_ROOT" >&2
  echo "Run: bash scripts/setup_mms_finetune.sh" >&2
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

JOB_SCRIPT="$OUT_DIR/e1_mms_kin.sbatch"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --time=${SLURM_TIME}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH -J ${SLURM_JOB_NAME}
#SBATCH -o ${LOG_DIR}/train_%j.log
#SBATCH -e ${LOG_DIR}/train_%j.err

set -euo pipefail
source "\${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate healthtts

export BASELINES="${BASELINES}"
export HF_HOME="${HF_HOME}"
export FT_ROOT="${FT_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_NAME="${WANDB_RUN_NAME}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME}"
# Map caches are ~14G — do NOT keep on /project (fills shared NFS).
# Compute-node /tmp is ~6TB SSD; rebuilt each job (~15min) but ckpts fit on community.
# Home is 25GB quota — never use it for caches.
export TMPDIR="/tmp/e1_mms_\${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="\$TMPDIR/hf_datasets"
export DATASETS_CACHE="\$HF_DATASETS_CACHE"
export WANDB_DIR="\$TMPDIR/wandb"
export WANDB_CACHE_DIR="\$TMPDIR/wandb_cache"
export WANDB_CONFIG_DIR="\$TMPDIR/wandb_config"
export WANDB_CONSOLE=off
mkdir -p "\$TMPDIR" "\$HF_DATASETS_CACHE" "\$HUGGINGFACE_HUB_CACHE" "${OUT_DIR}" \
  "\$WANDB_DIR" "\$WANDB_CACHE_DIR" "\$WANDB_CONFIG_DIR"
# Drop any leftover community map cache from older runs (safe; slows next preprocess only).
rm -rf "${BASELINES}/e1_map_cache" 2>/dev/null || true
df -h /tmp /project/community ~ 2>/dev/null | tail -8 || true
echo "Preprocess cache (node-local): \$HF_DATASETS_CACHE"
echo "WANDB local dir: \$WANDB_DIR (console off)"

cd "${REPO_ROOT}"
pip install -q wandb 2>/dev/null || true
nvidia-smi || true

python scripts/run_mms_finetune_e1.py --config "${CONFIG}" --finetune_root "${FT_ROOT}"
echo DONE_E1
EOF

echo "Submitting: partition=${SLURM_PARTITION} time=${SLURM_TIME} mem=${SLURM_MEM} cpus=${SLURM_CPUS} gpus=${SLURM_GPUS}"
echo "Job script:  $JOB_SCRIPT"
echo "Logs/ckpts:  $OUT_DIR"
df -h "$PROJECT_ROOT" | tail -1
sbatch "$JOB_SCRIPT"

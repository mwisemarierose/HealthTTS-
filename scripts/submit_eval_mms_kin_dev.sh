#!/usr/bin/env bash
# Eval E1 (or B0) MMS on Kin training-pool DEV split (not FLEURS).
#
#   # E1 fine-tune (default)
#   bash scripts/submit_eval_mms_kin_dev.sh
#
#   # smoke
#   LIMIT=20 bash scripts/submit_eval_mms_kin_dev.sh
#
#   # zero-shot B0 for comparison
#   MODEL_DIR=/project/community/rmwisene/tts_baselines/mms_tts_kin \
#   OUT_ROOT=/project/community/rmwisene/tts_baselines/eval_kin_dev/mms_b0 \
#     bash scripts/submit_eval_mms_kin_dev.sh
#
# Then score:
#   SYNTH_DIR=.../eval_kin_dev/mms_e1/dev bash scripts/submit_score_tts_fleurs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"

MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_ROOT/tts_data_cleaning/kinyarwanda_tts_mbaza_afrispeak/manifests}"
SPLIT="${SPLIT:-dev}"
# After E1, save_pretrained writes HF weights into this dir (not only checkpoint-*).
MODEL_DIR="${MODEL_DIR:-$BASELINES/finetune_e1_mms_kin}"
OUT_ROOT="${OUT_ROOT:-$BASELINES/eval_kin_dev/mms_e1}"
LIMIT="${LIMIT:-0}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-mms-kin-dev-eval}"

if [[ ! -f "$MANIFEST_DIR/${SPLIT}.csv" && ! -f "$MANIFEST_DIR/${SPLIT}.tsv" ]]; then
  echo "Missing $MANIFEST_DIR/${SPLIT}.csv|tsv" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing model at $MODEL_DIR (need config.json from save_pretrained)" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$HF_HOME"

PY_EXTRA=""
if [[ "$LIMIT" != "0" ]]; then
  PY_EXTRA="--limit ${LIMIT}"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --time=${SLURM_TIME}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH -J ${SLURM_JOB_NAME}
#SBATCH -o ${OUT_ROOT}/eval_%j.log

set -euo pipefail
source "\${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate healthtts

export BASELINES="${BASELINES}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"

cd "${REPO_ROOT}"
nvidia-smi || true
echo "MODEL_DIR=${MODEL_DIR}"
echo "MANIFEST_DIR=${MANIFEST_DIR} SPLIT=${SPLIT}"
python scripts/eval_mms_fleurs.py \\
  --manifest_dir "${MANIFEST_DIR}" \\
  --split "${SPLIT}" \\
  --model_dir "${MODEL_DIR}" \\
  --out_dir "${OUT_ROOT}" \\
  --device cuda ${PY_EXTRA}
echo DONE_SYNTH
EOF

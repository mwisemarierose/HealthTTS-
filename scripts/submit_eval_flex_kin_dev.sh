#!/usr/bin/env bash
# Eval Flex on Kin training-pool DEV (same foundation as MMS E1 kin-dev).
#
#   # E2 fine-tune (default)
#   bash scripts/submit_eval_flex_kin_dev.sh
#
#   # smoke
#   LIMIT=20 bash scripts/submit_eval_flex_kin_dev.sh
#
#   # B1 baseline
#   CKPT=/project/community/rmwisene/tts_baselines/kinya_flex_tts/kinya_flex_tts_base_trained.pt \
#   OUT_ROOT=/project/community/rmwisene/tts_baselines/eval_kin_dev/flex_b1 \
#     bash scripts/submit_eval_flex_kin_dev.sh
#
# Then score (needs CTC .nemo):
#   SYNTH_DIR=.../eval_kin_dev/flex_e2/dev RETIME=0 \
#     SLURM_JOB_NAME=score-flex-e2 bash scripts/submit_score_tts_fleurs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
DEEPKIN_ROOT="${DEEPKIN_ROOT:-$PROJECT_ROOT/code/ac-ai-models/DeepKIN-AgAI}"

MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_ROOT/tts_data_cleaning/kinyarwanda_tts_mbaza_afrispeak/manifests}"
SPLIT="${SPLIT:-dev}"
CKPT="${CKPT:-$BASELINES/finetune_e2_flex_kin/kinya_flex_tts_e2.pt}"
OUT_ROOT="${OUT_ROOT:-$BASELINES/eval_kin_dev/flex_e2}"
SPEAKER="${SPEAKER:--1}"   # -1 = per-utt from Kin CSV
LIMIT="${LIMIT:-0}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-flex-kin-dev-eval}"

if [[ ! -f "$MANIFEST_DIR/${SPLIT}.csv" && ! -f "$MANIFEST_DIR/${SPLIT}.tsv" ]]; then
  echo "Missing $MANIFEST_DIR/${SPLIT}.csv|tsv" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi
if [[ ! -d "$DEEPKIN_ROOT" ]]; then
  echo "Missing DeepKIN at $DEEPKIN_ROOT — run bash scripts/setup_deepkin.sh" >&2
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
export DEEPKIN_ROOT="${DEEPKIN_ROOT}"
export PYTHONPATH="${DEEPKIN_ROOT}:\${PYTHONPATH:-}"
export TMPDIR="/tmp/flex_kin_dev_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR"

cd "${REPO_ROOT}"
nvidia-smi || true
echo "CKPT=${CKPT}"
echo "MANIFEST_DIR=${MANIFEST_DIR} SPLIT=${SPLIT} SPEAKER=${SPEAKER}"
python scripts/eval_flex_fleurs.py \\
  --manifest_dir "${MANIFEST_DIR}" \\
  --split "${SPLIT}" \\
  --checkpoint "${CKPT}" \\
  --out_dir "${OUT_ROOT}" \\
  --speaker ${SPEAKER} \\
  --deepkin_root "${DEEPKIN_ROOT}" \\
  --device cuda ${PY_EXTRA}
echo DONE_SYNTH
EOF

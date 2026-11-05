#!/usr/bin/env bash
# E3 MMS bilingual → synth Kidaw’ida DEV + TEST (report separately).
#
#   bash scripts/submit_eval_mms_e3_dav.sh
#   LIMIT=20 bash scripts/submit_eval_mms_e3_dav.sh   # smoke
#
# Score after DONE:
#   for sp in dev test; do
#     SYNTH_DIR=.../eval_kidawida/mms_e3/$sp RETIME=0 \
#       SLURM_JOB_NAME=score-e3-dav-$sp bash scripts/submit_score_tts_fleurs.sh
#   done

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"

MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_ROOT/tts_data_cleaning/kidawida_tts_prepared/manifests}"
MODEL_DIR="${MODEL_DIR:-$BASELINES/finetune_e3_mms_bilingual_balanced}"
OUT_ROOT="${OUT_ROOT:-$BASELINES/eval_kidawida/mms_e3}"
# Dav speakers in E3 start at id 3 (see build_bilingual_mms_dataset.py)
SPEAKER_ID="${SPEAKER_ID:-3}"
SPLITS="${SPLITS:-dev,test}"
LIMIT="${LIMIT:-0}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-mms-e3-dav-eval}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing E3 HF model at $MODEL_DIR (need config.json from training export)" >&2
  echo "If only checkpoint-* exists, wait for save_pretrained or copy from last ckpt." >&2
  exit 1
fi
for sp in ${SPLITS//,/ }; do
  if [[ ! -f "$MANIFEST_DIR/${sp}.csv" && ! -f "$MANIFEST_DIR/${sp}.tsv" ]]; then
    echo "Missing $MANIFEST_DIR/${sp}.csv|tsv" >&2
    exit 1
  fi
done

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
echo "MANIFEST_DIR=${MANIFEST_DIR} SPLITS=${SPLITS} SPEAKER_ID=${SPEAKER_ID}"

IFS=',' read -r -a SPLIT_ARR <<< "${SPLITS}"
for sp in "\${SPLIT_ARR[@]}"; do
  sp=\$(echo "\$sp" | xargs)
  echo "===== E3 MMS Dav synth split=\$sp ====="
  python scripts/eval_mms_fleurs.py \\
    --manifest_dir "${MANIFEST_DIR}" \\
    --split "\$sp" \\
    --model_dir "${MODEL_DIR}" \\
    --out_dir "${OUT_ROOT}" \\
    --speaker_id ${SPEAKER_ID} \\
    --device cuda ${PY_EXTRA}
done
echo DONE_SYNTH_E3_DAV
EOF

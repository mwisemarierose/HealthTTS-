#!/usr/bin/env bash
# E5 MMS bilingual FULL Dav → synth Kidaw’ida DEV + TEST.
#
#   bash scripts/submit_eval_mms_e5_dav.sh
#
# Score:
#   for sp in dev test; do
#     SYNTH_DIR=.../eval_kidawida/mms_e5/$sp RETIME=0 \
#       SLURM_JOB_NAME=score-e5-dav-$sp bash scripts/submit_score_tts_fleurs.sh
#   done

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"

MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_ROOT/tts_data_cleaning/kidawida_tts_prepared/manifests}"
MODEL_DIR="${MODEL_DIR:-$BASELINES/finetune_e5_mms_bilingual_full}"
OUT_ROOT="${OUT_ROOT:-$BASELINES/eval_kidawida/mms_e5}"
SPEAKER_ID="${SPEAKER_ID:-3}"
SPLITS="${SPLITS:-dev,test}"
LIMIT="${LIMIT:-0}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-mms-e5-dav-eval}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing E5 HF model at $MODEL_DIR (need config.json)" >&2
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
[[ "$LIMIT" != "0" ]] && PY_EXTRA="--limit ${LIMIT}"

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
export HF_HOME="${HF_HOME}" HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub" TRANSFORMERS_CACHE="${HF_HOME}"
cd "${REPO_ROOT}"
nvidia-smi || true
echo "MODEL_DIR=${MODEL_DIR} SPLITS=${SPLITS}"
IFS=',' read -r -a SPLIT_ARR <<< "${SPLITS}"
for sp in "\${SPLIT_ARR[@]}"; do
  sp=\$(echo "\$sp" | xargs)
  echo "===== E5 MMS Dav synth split=\$sp ====="
  python scripts/eval_mms_fleurs.py \\
    --manifest_dir "${MANIFEST_DIR}" --split "\$sp" \\
    --model_dir "${MODEL_DIR}" --out_dir "${OUT_ROOT}" \\
    --speaker_id ${SPEAKER_ID} --device cuda ${PY_EXTRA}
done
echo DONE_SYNTH_E5_DAV
EOF

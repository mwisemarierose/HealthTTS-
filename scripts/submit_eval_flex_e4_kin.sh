#!/usr/bin/env bash
# E4 Flex bilingual → synth Kin pool DEV + TEST (report separately).
#
#   bash scripts/submit_eval_flex_e4_kin.sh
#
# Score:
#   for sp in dev test; do
#     SYNTH_DIR=.../eval_kin/flex_e4/$sp RETIME=0 \
#       SLURM_JOB_NAME=score-e4-kin-$sp bash scripts/submit_score_tts_fleurs.sh
#   done

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
DEEPKIN_ROOT="${DEEPKIN_ROOT:-$PROJECT_ROOT/code/ac-ai-models/DeepKIN-AgAI}"

MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_ROOT/tts_data_cleaning/kinyarwanda_tts_mbaza_afrispeak/manifests}"
CKPT="${CKPT:-$BASELINES/finetune_e4_flex_bilingual_balanced/kinya_flex_tts_e4.pt}"
OUT_ROOT="${OUT_ROOT:-$BASELINES/eval_kin/flex_e4}"
SPEAKER="${SPEAKER:--1}"   # per-utt Kin speakers
SPLITS="${SPLITS:-dev,test}"
LIMIT="${LIMIT:-0}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-flex-e4-kin-eval}"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi
if [[ ! -d "$DEEPKIN_ROOT" ]]; then
  echo "Missing DeepKIN at $DEEPKIN_ROOT" >&2
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
export DEEPKIN_ROOT="${DEEPKIN_ROOT}"
export PYTHONPATH="${DEEPKIN_ROOT}:\${PYTHONPATH:-}"
export TMPDIR="/tmp/flex_e4_kin_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR"

cd "${REPO_ROOT}"
nvidia-smi || true
echo "CKPT=${CKPT} SPEAKER=${SPEAKER} SPLITS=${SPLITS}"

IFS=',' read -r -a SPLIT_ARR <<< "${SPLITS}"
for sp in "\${SPLIT_ARR[@]}"; do
  sp=\$(echo "\$sp" | xargs)
  echo "===== E4 Flex Kin synth split=\$sp ====="
  python scripts/eval_flex_fleurs.py \\
    --manifest_dir "${MANIFEST_DIR}" \\
    --split "\$sp" \\
    --checkpoint "${CKPT}" \\
    --out_dir "${OUT_ROOT}" \\
    --speaker ${SPEAKER} \\
    --deepkin_root "${DEEPKIN_ROOT}" \\
    --device cuda ${PY_EXTRA}
done
echo DONE_SYNTH_E4_KIN
EOF

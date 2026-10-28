#!/bin/bash
# B1: Kinya-Flex zero-shot on Kidaw’ida ASR test (GPU).
#
#   python scripts/prepare_kidawida_eval_manifest.py   # once
#   bash scripts/setup_deepkin.sh                     # once
#   bash scripts/submit_eval_flex_kidawida.sh
#   SPLITS=test,dev SPEAKER=0 bash scripts/submit_eval_flex_kidawida.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
DEEPKIN_ROOT="${DEEPKIN_ROOT:-$PROJECT_ROOT/code/ac-ai-models/DeepKIN-AgAI}"
MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_ROOT/datasets/kidawida_asr_drive/processed/kidawida/eval}"
OUT_ROOT="${OUT_ROOT:-$BASELINES/eval_kidawida/kinya_flex_tts}"
SPLITS="${SPLITS:-${SPLIT:-test}}"
SPEAKER="${SPEAKER:-0}"
LIMIT="${LIMIT:-0}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-flex-dav-zeroshot}"

CKPT="$BASELINES/kinya_flex_tts/kinya_flex_tts_base_trained.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing $CKPT — run bash scripts/download_baselines.sh"
  exit 1
fi
if [[ ! -d "$DEEPKIN_ROOT" ]]; then
  echo "Missing DeepKIN at $DEEPKIN_ROOT — run bash scripts/setup_deepkin.sh"
  exit 1
fi
if [[ ! -f "$MANIFEST_DIR/test.tsv" ]]; then
  echo "Missing $MANIFEST_DIR/test.tsv"
  echo "Run: python scripts/prepare_kidawida_eval_manifest.py"
  exit 1
fi

mkdir -p "$OUT_ROOT" "$HF_HOME"

PY_EXTRA=""
if [[ "$LIMIT" != "0" ]]; then
  PY_EXTRA="--limit ${LIMIT}"
fi

SPLITS_CSV="$SPLITS"

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

cd "${REPO_ROOT}"
nvidia-smi || true

IFS=',' read -r -a SPLIT_ARR <<< "${SPLITS_CSV}"
for sp in "\${SPLIT_ARR[@]}"; do
  sp=\$(echo "\$sp" | xargs)
  echo "===== Flex Kidaw’ida zero-shot split=\$sp speaker=${SPEAKER} ====="
  python scripts/eval_flex_fleurs.py \\
    --manifest_dir "${MANIFEST_DIR}" \\
    --split "\$sp" \\
    --out_dir "${OUT_ROOT}" \\
    --speaker ${SPEAKER} \\
    --checkpoint "${CKPT}" \\
    --deepkin_root "${DEEPKIN_ROOT}" \\
    --device cuda ${PY_EXTRA}
done

echo DONE_ALL_SPLITS
EOF

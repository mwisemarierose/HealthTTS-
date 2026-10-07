#!/bin/bash
# Kinya-Flex synthesis on cleaned FLEURS (GPU).
#
#   bash scripts/setup_deepkin.sh          # once
#   SPLIT=test bash scripts/submit_eval_flex_fleurs.sh
#   SPLIT=dev  bash scripts/submit_eval_flex_fleurs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
DEEPKIN_ROOT="${DEEPKIN_ROOT:-$PROJECT_ROOT/code/ac-ai-models/DeepKIN-AgAI}"
OUT_ROOT="${OUT_ROOT:-$BASELINES/eval_fleurs/kinya_flex_tts}"
SPLIT="${SPLIT:-test}"
SPEAKER="${SPEAKER:-0}"
LIMIT="${LIMIT:-0}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-3:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-flex-fleurs-eval}"

CKPT="$BASELINES/kinya_flex_tts/kinya_flex_tts_base_trained.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing $CKPT — run bash scripts/download_baselines.sh"
  exit 1
fi
if [[ ! -d "$DEEPKIN_ROOT" ]]; then
  echo "Missing DeepKIN at $DEEPKIN_ROOT — run bash scripts/setup_deepkin.sh"
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
export TMPDIR="/tmp/flex_fleurs_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR"

cd "${REPO_ROOT}"
nvidia-smi || true
python scripts/eval_flex_fleurs.py \\
  --split "${SPLIT}" \\
  --speaker ${SPEAKER} \\
  --checkpoint "${CKPT}" \\
  --deepkin_root "${DEEPKIN_ROOT}" \\
  --device cuda ${PY_EXTRA}
echo DONE
EOF

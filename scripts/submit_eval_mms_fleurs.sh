#!/bin/bash
# Full MMS-TTS Kin baseline synthesis on cleaned FLEURS test (GPU).
#
#   cd ~/HealthTTS
#   bash scripts/submit_eval_mms_fleurs.sh
#
# Optional overrides:
#   SPLIT=dev LIMIT=0 SLURM_TIME=2:00:00 bash scripts/submit_eval_mms_fleurs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
OUT_ROOT="${OUT_ROOT:-$BASELINES/eval_fleurs/mms_tts_kin}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-0}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-2:00:00}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-mms-fleurs-eval}"

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
export TMPDIR="/tmp/mms_fleurs_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR" "\$HUGGINGFACE_HUB_CACHE"

cd "${REPO_ROOT}"
nvidia-smi || true
python scripts/eval_mms_fleurs.py --split "${SPLIT}" --device cuda ${PY_EXTRA}
echo DONE
EOF

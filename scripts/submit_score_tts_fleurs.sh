#!/bin/bash
# Score MMS FLEURS synths: WER/CER + latency/RTF + UTMOS (GPU).
#
#   bash scripts/submit_score_tts_fleurs.sh
#
# Existing synths have no timing → uses --retime by default.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
BASELINES="${BASELINES:-$PROJECT_ROOT/tts_baselines}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
SYNTH_DIR="${SYNTH_DIR:-$BASELINES/eval_fleurs/mms_tts_kin/test}"
ASR_MODEL="${ASR_MODEL:-mbazaNLP/Whisper-Small-Kinyarwanda}"

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-tts-score-fleurs}"

mkdir -p "$SYNTH_DIR" "$HF_HOME"

sbatch <<EOF
#!/bin/bash
#SBATCH -p ${SLURM_PARTITION}
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --time=${SLURM_TIME}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH -J ${SLURM_JOB_NAME}
#SBATCH -o ${SYNTH_DIR}/score_%j.log

set -euo pipefail
source "\${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate healthtts

export BASELINES="${BASELINES}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "\$HUGGINGFACE_HUB_CACHE" "\$TORCH_HOME"

cd "${REPO_ROOT}"
pip install -q jiwer 2>/dev/null || true
nvidia-smi || true

python scripts/score_tts_fleurs.py \\
  --synth_dir "${SYNTH_DIR}" \\
  --asr_model "${ASR_MODEL}" \\
  --tts_model_dir "${BASELINES}/mms_tts_kin" \\
  --retime \\
  --device cuda

echo DONE
cat "${SYNTH_DIR}/metrics.json"
EOF

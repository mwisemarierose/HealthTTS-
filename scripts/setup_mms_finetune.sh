#!/usr/bin/env bash
# Clone ylacombe/finetune-hf-vits and convert MMS-TTS-Kin discriminator (once).
#
#   conda activate healthtts
#   bash scripts/setup_mms_finetune.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
CODE_ROOT="${CODE_ROOT:-$PROJECT_ROOT/code}"
FT_ROOT="${FT_ROOT:-$CODE_ROOT/finetune-hf-vits}"
OUT_CKPT="${OUT_CKPT:-$PROJECT_ROOT/tts_baselines/mms_tts_kin_train}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

mkdir -p "$CODE_ROOT" "$(dirname "$OUT_CKPT")"

if [[ ! -d "$FT_ROOT/.git" ]]; then
  echo "Cloning ylacombe/finetune-hf-vits → $FT_ROOT"
  git clone --depth 1 https://github.com/ylacombe/finetune-hf-vits.git "$FT_ROOT"
else
  echo "Already cloned: $FT_ROOT"
fi

cd "$FT_ROOT"
pip install -q -r requirements.txt
pip install -q Cython

# monotonic alignment (required)
if [[ -d monotonic_align ]]; then
  echo "Building monotonic_align …"
  (
    cd monotonic_align
    mkdir -p monotonic_align
    python setup.py build_ext --inplace
  )
fi

if [[ ! -f "$OUT_CKPT/config.json" ]]; then
  echo "Converting MMS-TTS Kin discriminator → $OUT_CKPT"
  python convert_original_discriminator_checkpoint.py \
    --language_code kin \
    --pytorch_dump_folder_path "$OUT_CKPT"
else
  echo "Trainable Kin checkpoint already present: $OUT_CKPT"
fi

echo
echo "OK"
echo "  finetune-hf-vits: $FT_ROOT"
echo "  train init ckpt:  $OUT_CKPT"
echo "Next:"
echo "  python scripts/export_kin_train_datasets.py"
echo "  bash scripts/submit_train_mms_e1.sh"

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

# ylacombe/finetune-hf-vits is pinned to ~4.35; newer transformers VitsConfig
# raises AttributeError on missing pad_token_id during discriminator convert.
# datasets 2.14.x also needs NumPy 1.x (np.array(..., copy=False) breaks on NumPy 2).
echo "Pinning transformers/datasets/accelerate/numpy for MMS finetune …"
pip install -q --upgrade \
  "numpy==1.26.4" \
  "transformers==4.35.1" \
  "datasets[audio]==2.14.7" \
  "accelerate==0.24.1"
pip install -q -r requirements.txt
pip install -q Cython wandb
# Re-assert pins in case requirements.txt floated them.
pip install -q \
  "numpy==1.26.4" \
  "transformers==4.35.1" \
  "datasets[audio]==2.14.7" \
  "accelerate==0.24.1"

# Ensure local VitsConfig always exposes pad_token_id (MMS pad id = 0).
python - <<'PY'
from pathlib import Path

p = Path("utils/configuration_vits.py")
text = p.read_text(encoding="utf-8")
if "self.pad_token_id" in text:
    print("VitsConfig already sets pad_token_id")
else:
    needle = "        self.hop_length = hop_length\n"
    insert = (
        "        self.hop_length = hop_length\n"
        "        # MMS / VITS pad token (missing on some transformers builds)\n"
        "        self.pad_token_id = kwargs.pop(\"pad_token_id\", 0)\n"
    )
    if needle not in text:
        raise SystemExit(f"Could not patch {p}: hop_length assignment not found")
    p.write_text(text.replace(needle, insert, 1), encoding="utf-8")
    print(f"Patched pad_token_id into {p}")
PY

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
echo "  transformers:     $(python -c 'import transformers; print(transformers.__version__)')"
echo "Next:"
echo "  python scripts/export_kin_train_datasets.py   # if not done"
echo "  bash scripts/submit_train_mms_e1.sh"

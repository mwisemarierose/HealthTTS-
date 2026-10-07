#!/usr/bin/env bash
# Install DeepKIN-AgAI (needed for Kinya-Flex TTS) on Orchard project storage.
#
#   bash scripts/setup_deepkin.sh
#   conda activate healthtts
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/community/rmwisene}"
DEEPKIN_ROOT="${DEEPKIN_ROOT:-$PROJECT_ROOT/code/ac-ai-models}"
DEEPKIN_DIR="$DEEPKIN_ROOT/DeepKIN-AgAI"

mkdir -p "$(dirname "$DEEPKIN_ROOT")"

if [[ ! -d "$DEEPKIN_DIR" ]]; then
  echo "Cloning c4ir-rw/ac-ai-models → $DEEPKIN_ROOT"
  git clone --depth 1 https://github.com/c4ir-rw/ac-ai-models.git "$DEEPKIN_ROOT"
else
  echo "Already cloned: $DEEPKIN_DIR"
fi

cd "$DEEPKIN_DIR"
pip install -r requirements.txt
pip install -e .

python - <<'PY'
from deepkin.models.flex_tts import FlexKinyaTTS
from deepkin.data.kinya_norm import text_to_sequence
print("DeepKIN Flex TTS import OK")
PY

echo "OK — DeepKIN at $DEEPKIN_DIR"

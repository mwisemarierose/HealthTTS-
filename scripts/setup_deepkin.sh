#!/usr/bin/env bash
# Minimal DeepKIN install for Kinya-Flex TTS inference ONLY.
# Do NOT pip install DeepKIN's full requirements.txt (flash_attn / torch pin hell).
#
#   conda activate healthtts
#   bash scripts/setup_deepkin.sh
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

# Editable install only (setup.py has empty install_requires)
pip install -e .

# Light deps for Flex inference (NOT full requirements.txt / flash_attn)
pip install -q \
  "typed-argument-parser" \
  einops \
  Cython

# Build monotonic_align if present (some TTS paths need it)
if [[ -d monotonic_align ]]; then
  echo "Building monotonic_align extension (best-effort)…"
  (cd monotonic_align && python setup.py build_ext --inplace) || \
    echo "WARNING: monotonic_align build failed — Flex infer may still work"
fi

export PYTHONPATH="$DEEPKIN_DIR:${PYTHONPATH:-}"
python - <<'PY'
from deepkin.models.flex_tts import FlexKinyaTTS
from deepkin.data.kinya_norm import text_to_sequence
from deepkin.modules.tts_commons import intersperse
print("DeepKIN Flex TTS import OK")
PY

echo
echo "OK — DeepKIN at $DEEPKIN_DIR"
echo "Use: export DEEPKIN_ROOT=$DEEPKIN_DIR"
echo "     export PYTHONPATH=\$DEEPKIN_ROOT:\$PYTHONPATH"

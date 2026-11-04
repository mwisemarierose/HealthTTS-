#!/usr/bin/env bash
# Minimal DeepKIN install for Kinya-Flex TTS (infer + train).
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

# Light deps for Flex (NOT full requirements.txt / flash_attn)
pip install -q \
  "typed-argument-parser" \
  einops \
  Cython \
  progressbar2 \
  numpy

# Build monotonic_align for *this* interpreter (healthtts = 3.11).
# Repo ships a nested monotonic_align/core.cpython-312*.so — wrong for 3.11.
# setup.py cythonizes top-level core.pyx → core.cpython-311*.so beside __init__.py.
if [[ ! -d monotonic_align ]]; then
  echo "ERROR: no monotonic_align/ under $DEEPKIN_DIR" >&2
  exit 1
fi
echo "Building monotonic_align for $(python -V) …"
(
  cd monotonic_align
  python setup.py build_ext --inplace
  ls -lh core.cpython-*.so
)

export PYTHONPATH="$DEEPKIN_DIR:${PYTHONPATH:-}"
python - <<'PY'
from deepkin.models.flex_tts import FlexKinyaTTS
from deepkin.data.kinya_norm import text_to_sequence
from deepkin.modules.tts_commons import intersperse
from monotonic_align.core import maximum_path_c
print("DeepKIN Flex TTS + monotonic_align import OK")
PY

echo
echo "OK — DeepKIN at $DEEPKIN_DIR"
echo "Use: export DEEPKIN_ROOT=$DEEPKIN_DIR"
echo "     export PYTHONPATH=\$DEEPKIN_ROOT:\$PYTHONPATH"

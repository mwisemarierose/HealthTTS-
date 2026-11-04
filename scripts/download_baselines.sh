#!/usr/bin/env bash
# Download Kin TTS baselines to Orchard project storage (not home / not git).
#
#   conda activate healthtts
#   bash scripts/download_baselines.sh
set -euo pipefail

BASELINES="${BASELINES:-/project/community/rmwisene/tts_baselines}"
mkdir -p "$BASELINES/mms_tts_kin" "$BASELINES/kinya_flex_tts"

# Prefer `hf`; fall back to huggingface-cli; install CLI if missing.
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Installing huggingface_hub[cli] ..."
  pip install -q -U "huggingface_hub[cli]"
fi

hf_dl() {
  local repo="$1"
  local dest="$2"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --local-dir "$dest"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" --local-dir "$dest"
  else
    echo "hf / huggingface-cli still missing after install" >&2
    exit 1
  fi
}

echo "==> facebook/mms-tts-kin → $BASELINES/mms_tts_kin"
hf_dl facebook/mms-tts-kin "$BASELINES/mms_tts_kin"

echo "==> wixdivin/kinya-flex-tts → $BASELINES/kinya_flex_tts"
hf_dl wixdivin/kinya-flex-tts "$BASELINES/kinya_flex_tts"

echo
echo "Done."
du -sh "$BASELINES"/mms_tts_kin "$BASELINES"/kinya_flex_tts
ls -lh "$BASELINES"/mms_tts_kin | head
ls -lh "$BASELINES"/kinya_flex_tts | head

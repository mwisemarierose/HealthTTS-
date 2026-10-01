#!/usr/bin/env bash
# Download Kin TTS baselines to Orchard project storage (not home / not git).
#
#   conda activate healthtts
#   bash scripts/download_baselines.sh
set -euo pipefail

BASELINES="${BASELINES:-/project/community/rmwisene/tts_baselines}"
mkdir -p "$BASELINES/mms_tts_kin" "$BASELINES/kinya_flex_tts"

echo "==> facebook/mms-tts-kin → $BASELINES/mms_tts_kin"
hf download facebook/mms-tts-kin \
  --local-dir "$BASELINES/mms_tts_kin"

echo "==> wixdivin/kinya-flex-tts → $BASELINES/kinya_flex_tts"
hf download wixdivin/kinya-flex-tts \
  --local-dir "$BASELINES/kinya_flex_tts"

echo
echo "Done."
du -sh "$BASELINES"/mms_tts_kin "$BASELINES"/kinya_flex_tts
ls -lh "$BASELINES"/mms_tts_kin | head
ls -lh "$BASELINES"/kinya_flex_tts | head

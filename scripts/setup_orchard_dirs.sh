#!/usr/bin/env bash
# Create dataset directories on Orchard project storage (not in this git repo).
# Usage:
#   bash scripts/setup_orchard_dirs.sh
#
# FLEURS is not created here — not needed for current Kin/Dav train + Kin-dev eval.
set -euo pipefail

DATA="${DATA:-/project/community/rmwisene/datasets}"

mkdir -p \
  "$DATA/kinyarwanda_tts" \
  "$DATA/afrispeak_kinyarwanda_female" \
  "$DATA/afrispeak_kinyarwanda_male" \
  "$DATA/kidawida_cv27"

echo "Created dataset dirs under: $DATA"
ls -la "$DATA" | head -50

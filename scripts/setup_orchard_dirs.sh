#!/usr/bin/env bash
# Create dataset directories on Orchard project storage (not in this git repo).
# Usage:
#   bash scripts/setup_orchard_dirs.sh
set -euo pipefail

DATA="${DATA:-/project/community/rmwisene/datasets}"

mkdir -p \
  "$DATA/kinyarwanda_tts" \
  "$DATA/kidawida_cv27" \
  "$DATA/fleurs_kinyarwanda"

echo "Created dataset dirs under: $DATA"
ls -la "$DATA" | head -50

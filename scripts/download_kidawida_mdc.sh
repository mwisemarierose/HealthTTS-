#!/usr/bin/env bash
# Download Kidaw'ida Common Voice from Mozilla Data Collective → project datasets/
# Usage:
#   export MDC_API_KEY='...'
#   bash scripts/download_kidawida_mdc.sh
set -euo pipefail

DATA="${DATA:-/project/community/rmwisene/datasets}"
OUT="$DATA/kidawida_cv27"
DATASET_ID="${MDC_DATASET_ID:-cmu5w2bh700aso107ffpduqpp}"
ARCHIVE="${MDC_ARCHIVE_NAME:-common-voice-scripted-speech-27-0-kidaw-69655f89.tar.gz}"

if [[ -z "${MDC_API_KEY:-}" ]]; then
  echo "Set MDC_API_KEY first (Mozilla Data Collective API token)." >&2
  exit 1
fi

command -v jq >/dev/null || { echo "jq is required"; exit 1; }

mkdir -p "$OUT"
cd "$OUT"

echo "Requesting presigned URL…"
RESPONSE=$(curl -sS -X POST \
  "https://mozilladatacollective.com/api/datasets/${DATASET_ID}/download" \
  -H "Authorization: Bearer ${MDC_API_KEY}" \
  -H "Content-Type: application/json")

DOWNLOAD_URL=$(echo "$RESPONSE" | jq -r '.downloadUrl')
if [[ -z "$DOWNLOAD_URL" || "$DOWNLOAD_URL" == "null" ]]; then
  echo "Failed to get downloadUrl. Response:" >&2
  echo "$RESPONSE" >&2
  exit 1
fi

echo "Downloading → $OUT/$ARCHIVE"
curl -L --fail -o "$ARCHIVE" "$DOWNLOAD_URL"

echo "Extracting…"
tar -xzf "$ARCHIVE"

echo "Done."
ls -lh
du -sh .

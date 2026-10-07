#!/usr/bin/env bash
# Download CTC ASR (.nemo) from Google Drive to project storage.
# File: https://drive.google.com/file/d/1gx4olc2XdlUNkNlFqXcOq1F4IbcVHyGI/view
#
#   bash scripts/download_ctc_asr.sh
set -euo pipefail

FILE_ID="${GDRIVE_CTC_ID:-1gx4olc2XdlUNkNlFqXcOq1F4IbcVHyGI}"
OUT_DIR="${CTC_DIR:-/project/community/rmwisene/asr}"
OUT_FILE="${CTC_NEMO:-$OUT_DIR/combined-ctc-15-ep-nocl.nemo}"

mkdir -p "$OUT_DIR"

if [[ -f "$OUT_FILE" ]]; then
  echo "Already present: $OUT_FILE"
  ls -lh "$OUT_FILE"
  exit 0
fi

echo "Downloading CTC .nemo → $OUT_FILE"

if command -v gdown >/dev/null 2>&1; then
  gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUT_FILE"
elif command -v rclone >/dev/null 2>&1; then
  echo "gdown not found; if you use rclone Drive remotes, copy manually, e.g.:"
  echo "  rclone copy <remote>:.../combined-ctc-15-ep-nocl.nemo $OUT_DIR/"
  exit 1
else
  pip install -q gdown
  gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUT_FILE"
fi

ls -lh "$OUT_FILE"
echo "Set: export CTC_NEMO=$OUT_FILE"

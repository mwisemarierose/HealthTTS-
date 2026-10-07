#!/usr/bin/env bash
# Download CTC ASR (.nemo) from Google Drive via rclone → project storage.
#
# Drive file:
#   https://drive.google.com/file/d/1gx4olc2XdlUNkNlFqXcOq1F4IbcVHyGI/view
#
# Usage (set remote path to where the .nemo lives on Drive):
#   export RCLONE_CTC_SRC='Gdrive-Okeyo:ASR Summer 2026/combined-ctc-15-ep-nocl.nemo'
#   bash scripts/download_ctc_asr.sh
#
# Or copy by Drive file id (if your rclone remote supports it):
#   rclone backend copyid <remote>: 1gx4olc2XdlUNkNlFqXcOq1F4IbcVHyGI /project/.../asr/
set -euo pipefail

OUT_DIR="${CTC_DIR:-/project/community/rmwisene/asr}"
OUT_FILE="${CTC_NEMO:-$OUT_DIR/combined-ctc-15-ep-nocl.nemo}"
# Override this to your actual Drive path/filename
RCLONE_CTC_SRC="${RCLONE_CTC_SRC:-}"
GDRIVE_CTC_ID="${GDRIVE_CTC_ID:-1gx4olc2XdlUNkNlFqXcOq1F4IbcVHyGI}"
RCLONE_REMOTE="${RCLONE_REMOTE:-Gdrive-Okeyo}"

mkdir -p "$OUT_DIR"

if [[ -f "$OUT_FILE" ]]; then
  echo "Already present: $OUT_FILE"
  ls -lh "$OUT_FILE"
  exit 0
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found on PATH" >&2
  exit 1
fi

echo "Destination: $OUT_FILE"

if [[ -n "$RCLONE_CTC_SRC" ]]; then
  echo "rclone copyfrom: $RCLONE_CTC_SRC"
  rclone copy "$RCLONE_CTC_SRC" "$OUT_DIR/" --progress
  # If source basename differs, rename
  src_base="$(basename "$RCLONE_CTC_SRC")"
  if [[ "$src_base" != "$(basename "$OUT_FILE")" && -f "$OUT_DIR/$src_base" ]]; then
    mv "$OUT_DIR/$src_base" "$OUT_FILE"
  fi
else
  echo "RCLONE_CTC_SRC not set — trying copyid on remote '$RCLONE_REMOTE'"
  echo "File id: $GDRIVE_CTC_ID"
  rclone backend copyid "$RCLONE_REMOTE:" "$GDRIVE_CTC_ID" "$OUT_DIR/" --progress
  # copyid keeps Drive filename; normalize to expected name if needed
  if [[ ! -f "$OUT_FILE" ]]; then
    found="$(find "$OUT_DIR" -maxdepth 1 -name '*.nemo' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [[ -n "${found:-}" && -f "$found" ]]; then
      mv "$found" "$OUT_FILE"
    fi
  fi
fi

if [[ ! -f "$OUT_FILE" ]]; then
  echo "Download finished but $OUT_FILE not found." >&2
  echo "List remotes: rclone listremotes"
  echo "Find file:    rclone ls \"$RCLONE_REMOTE:\" | grep -i nemo"
  echo "Then: export RCLONE_CTC_SRC='<remote>:<path/to/file.nemo>'"
  ls -lah "$OUT_DIR" || true
  exit 1
fi

ls -lh "$OUT_FILE"
echo "OK — export CTC_NEMO=$OUT_FILE"

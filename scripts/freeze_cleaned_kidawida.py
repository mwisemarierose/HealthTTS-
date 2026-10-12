#!/usr/bin/env python3
"""Freeze the cleaned Kidaw’ida corpus pointer + extract flagged rows for review.

Does not modify kidawida_cv27. Does not correct q/x/ä.

  python scripts/freeze_cleaned_kidawida.py

Writes under /project/community/rmwisene/tts_data_cleaning/:
  FROZEN.txt
  flagged_for_native_review.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

CLEAN_DIR = Path("/project/community/rmwisene/tts_data_cleaning")
CLEANED = CLEAN_DIR / "kidawida_cleaned_transcripts.csv"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned_csv", type=Path, default=CLEANED)
    ap.add_argument("--out_dir", type=Path, default=CLEAN_DIR)
    args = ap.parse_args()

    if not args.cleaned_csv.is_file():
        raise SystemExit(f"Missing {args.cleaned_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    flagged_path = args.out_dir / "flagged_for_native_review.csv"
    freeze_path = args.out_dir / "FROZEN.txt"

    n = 0
    n_flagged = 0
    flagged_rows: list[dict] = []
    with args.cleaned_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            n += 1
            if (row.get("flagged") or "").strip().lower() in {"yes", "true", "1"}:
                n_flagged += 1
                flagged_rows.append(row)

    with flagged_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flagged_rows)

    digest = file_sha256(args.cleaned_csv)
    freeze_path.write_text(
        "\n".join(
            [
                "FROZEN cleaned Kidaw’ida corpus",
                "==============================",
                f"frozen_at_utc: {datetime.now(timezone.utc).isoformat()}",
                f"cleaned_csv: {args.cleaned_csv}",
                f"sha256: {digest}",
                f"n_sentences: {n}",
                f"n_flagged_for_native_review: {n_flagged}",
                f"flagged_file: {flagged_path}",
                "",
                "Policy:",
                "- Use cleaned_text for tokenizer/frontend coverage and later training prep.",
                "- Keep the flagged sentences; do not delete or auto-correct q, x, ä yet.",
                "- Native-speaker review of flagged rows is deferred.",
                "- Do not overwrite datasets/kidawida_cv27.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Frozen: {args.cleaned_csv}")
    print(f"  sha256: {digest}")
    print(f"  sentences: {n}")
    print(f"  flagged kept for review: {n_flagged}")
    print(f"Wrote {freeze_path}")
    print(f"Wrote {flagged_path}")


if __name__ == "__main__":
    main()

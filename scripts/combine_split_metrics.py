#!/usr/bin/env python3
"""Pack existing test/dev metrics.json into one file (no recompute).

  python scripts/combine_split_metrics.py \\
    --eval_root $BASELINES/eval_fleurs/mms_tts_kin

Writes:
  <eval_root>/metrics_test_dev.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval_root",
        required=True,
        help="e.g. .../eval_fleurs/mms_tts_kin (must contain test/ and dev/)",
    )
    p.add_argument(
        "--splits",
        default="test,dev",
        help="Comma-separated split dirs (default: test,dev)",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output path (default: <eval_root>/metrics_test_dev.json)",
    )
    args = p.parse_args()

    root = Path(args.eval_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    packed: dict = {
        "eval_root": str(root),
        "splits": {},
    }
    for sp in splits:
        path = root / sp / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        packed["splits"][sp] = json.loads(path.read_text(encoding="utf-8"))

    out = Path(args.out) if args.out else root / "metrics_test_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(packed, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(packed, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

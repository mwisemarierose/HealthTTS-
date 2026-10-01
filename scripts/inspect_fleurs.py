#!/usr/bin/env python3
"""Inspect FLEURS Kinyarwanda layout on Orchard (eval dataset).

  conda activate healthtts
  python scripts/inspect_fleurs.py
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/project/community/rmwisene/datasets/fleurs_kinyarwanda",
    )
    args = p.parse_args()
    root = Path(args.root)
    print(f"root: {root}  exists={root.is_dir()}")
    if not root.is_dir():
        return

    print("\n== top-level ==")
    for x in sorted(root.iterdir()):
        kind = "dir" if x.is_dir() else "file"
        size = ""
        if x.is_file():
            size = f"  {x.stat().st_size / 1e6:.1f}MB"
        print(f"  [{kind}] {x.name}{size}")

    for name in ("train.tsv", "dev.tsv", "test.tsv"):
        tsv = root / name
        print(f"\n== {name} ==")
        if not tsv.is_file():
            print("  MISSING")
            continue
        lines = tsv.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"  lines: {len(lines)}")
        for row in lines[:3]:
            print(f"  {row[:200]}")

    print("\n== audio archives / dirs ==")
    for pattern in ("**/*.tar.xz", "**/clips", "**/audio"):
        hits = list(root.glob(pattern))[:20]
        for h in hits:
            print(f"  {h.relative_to(root)}")


if __name__ == "__main__":
    main()

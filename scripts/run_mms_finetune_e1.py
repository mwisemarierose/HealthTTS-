#!/usr/bin/env python3
"""Launch E1 MMS Kin fine-tune via ylacombe/finetune-hf-vits.

Patches datasets.load_dataset so local DatasetDict paths from
export_kin_train_datasets.py work with run_vits_finetuning.py.

Also redirects Dataset.map/filter cache_file_name to $HF_DATASETS_CACHE
(or $E1_SCRATCH) so preprocess Arrow caches do not grow inside
kin_mms_hf on /project. Does NOT copy the full dataset (home quota).

  python scripts/run_mms_finetune_e1.py
  python scripts/run_mms_finetune_e1.py --config configs/e1_mms_kin_finetune.json
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


def patch_load_dataset_for_disk() -> None:
    import datasets
    from datasets import DatasetDict, load_from_disk

    orig = datasets.load_dataset

    def load_dataset(path, name=None, split=None, **kwargs):
        p = Path(str(path))
        if p.is_dir() and (
            (p / "dataset_dict.json").is_file() or (p / "state.json").is_file()
        ):
            ds = load_from_disk(str(p))
            if isinstance(ds, DatasetDict):
                if split is None:
                    return ds
                if split in ds:
                    return ds[split]
                aliases = {
                    "validation": ["validation", "eval", "dev"],
                    "eval": ["validation", "eval", "dev"],
                    "train": ["train"],
                }
                for cand in aliases.get(str(split), [str(split)]):
                    if cand in ds:
                        return ds[cand]
                raise KeyError(f"Split {split!r} not in {list(ds.keys())} under {p}")
            return ds
        return orig(path, name, split=split, **kwargs)

    datasets.load_dataset = load_dataset  # type: ignore[assignment]


def patch_map_filter_cache_dir(cache_root: Path) -> None:
    """Force map/filter Arrow outputs under cache_root (not next to load_from_disk)."""
    from datasets import Dataset

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    counter = {"i": 0}

    def _next_prefix() -> str:
        counter["i"] += 1
        return str(cache_root / f"e1_ds_{counter['i']:03d}")

    def _wrap(orig):
        def wrapped(self, *args, **kwargs):
            if kwargs.get("cache_file_name") is None:
                kwargs["cache_file_name"] = _next_prefix()
            return orig(self, *args, **kwargs)

        return wrapped

    Dataset.map = _wrap(Dataset.map)  # type: ignore[method-assign]
    Dataset.filter = _wrap(Dataset.filter)  # type: ignore[method-assign]
    print(f"datasets map/filter caches → {cache_root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "e1_mms_kin_finetune.json",
    )
    ap.add_argument(
        "--finetune_root",
        type=Path,
        default=Path(
            os.environ.get(
                "FT_ROOT",
                "/project/community/rmwisene/code/finetune-hf-vits",
            )
        ),
    )
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ds_path = Path(cfg["dataset_name"])
    if not (ds_path / "dataset_dict.json").is_file():
        raise SystemExit(
            f"Missing MMS HF dataset at {ds_path}\n"
            "Run: python scripts/export_kin_train_datasets.py"
        )
    model_path = Path(cfg["model_name_or_path"])
    if not (model_path / "config.json").is_file():
        raise SystemExit(
            f"Missing trainable MMS Kin ckpt at {model_path}\n"
            "Run: bash scripts/setup_mms_finetune.sh"
        )
    ft = args.finetune_root
    train_script = ft / "run_vits_finetuning.py"
    if not train_script.is_file():
        raise SystemExit(f"Missing {train_script}\nRun: bash scripts/setup_mms_finetune.sh")

    cache_root = Path(
        os.environ.get("HF_DATASETS_CACHE")
        or os.environ.get("E1_SCRATCH")
        or tempfile.gettempdir()
    )
    patch_load_dataset_for_disk()
    patch_map_filter_cache_dir(cache_root)

    sys.path.insert(0, str(ft))
    os.chdir(ft)
    sys.argv = [str(train_script), str(args.config.resolve())]
    print(f"Launching {train_script} with {args.config}")
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()

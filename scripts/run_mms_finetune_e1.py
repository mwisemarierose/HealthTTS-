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


def print_run_banner(cfg: dict, resume_step: int | None) -> None:
    epochs = int(cfg.get("num_train_epochs", 30))
    # From prior runs: ~50370 steps / 30 epochs ≈ 1679 steps/epoch
    steps_per_epoch = int(os.environ.get("E1_STEPS_PER_EPOCH", "1679"))
    total_steps = steps_per_epoch * epochs
    step = resume_step or 0
    epoch_f = step / steps_per_epoch if steps_per_epoch else 0.0
    print("=" * 72)
    print("E1_STATUS: PHASE=launch")
    print(f"E1_STATUS: run={cfg.get('run_name')}  project={cfg.get('project_name')}")
    print(f"E1_STATUS: output_dir={cfg.get('output_dir')}")
    if resume_step is not None:
        print(f"E1_STATUS: resume_step={resume_step}  ≈epoch {epoch_f:.2f}/{epochs}")
        print("E1_STATUS: next=preprocess_or_load_cache → then continue TRAINING")
    else:
        print(f"E1_STATUS: start_step=0  epochs={epochs}  (~{total_steps} steps)")
        print("E1_STATUS: next=preprocess → TRAINING")
    print(
        "E1_STATUS: phases you will see: "
        "preprocess | train_epoch_start | train_step | "
        "validation_eval (held-out batches) | save_checkpoint"
    )
    print(f"E1_STATUS: epoch≈ step/{steps_per_epoch}   (eval every {cfg.get('eval_steps')} steps)")
    print("=" * 72, flush=True)


def patch_vits_status_logging(ft_root: Path) -> None:
    """Make run_vits_finetuning.py print clear E1_STATUS lines with epoch/step/phase."""
    path = ft_root / "run_vits_finetuning.py"
    text = path.read_text(encoding="utf-8")
    if "E1_STATUS: PHASE=train_epoch_start" in text:
        print(f"status logging already patched: {path}")
        return

    reps_once = [
        (
            "for epoch in range(first_epoch, training_args.num_train_epochs):\n"
            "        # keep track of train losses\n",
            "for epoch in range(first_epoch, training_args.num_train_epochs):\n"
            "        logger.info(\n"
            '            f"E1_STATUS: PHASE=train_epoch_start | '
            "epoch={epoch+1}/{int(training_args.num_train_epochs)} | "
            'step={global_step}/{training_args.max_steps}"\n'
            "        )\n"
            "        # keep track of train losses\n",
        ),
        (
            'logger.info(f"Saved state to {save_path}")',
            'logger.info(f"Saved state to {save_path}")\n'
            "                    logger.info(\n"
            '                        f"E1_STATUS: PHASE=save_checkpoint | '
            "epoch≈{(global_step / max(num_update_steps_per_epoch, 1)):.2f}/"
            "{int(training_args.num_train_epochs)} | "
            'step={global_step} | path={save_path}"\n'
            "                    )",
        ),
        (
            'logger.info("***** Running training *****")',
            'logger.info("***** Running training *****")\n'
            '    logger.info("E1_STATUS: PHASE=train_loop_starting")',
        ),
    ]
    new = text
    for old, repl in reps_once:
        if old not in new:
            raise SystemExit(f"status patch failed, missing snippet in {path}:\n{old[:120]}...")
        new = new.replace(old, repl, 1)

    # These strings appear in mid-train and final validation — replace all.
    reps_all = [
        (
            'logger.info("Running validation... ")',
            'logger.info(f"E1_STATUS: PHASE=validation_eval | epoch={epoch+1}/{int(training_args.num_train_epochs)} | step={global_step}/{training_args.max_steps} | action=held-out_synth+metrics")',
        ),
        (
            'logger.info("Running final validation... ")',
            'logger.info(f"E1_STATUS: PHASE=validation_eval_final | epoch={epoch}/{int(training_args.num_train_epochs)} | step={global_step}/{training_args.max_steps} | action=held-out_synth+metrics")',
        ),
        (
            "VALIDATION - batch {step}, process{accelerator.process_index}, waveform {(batch['waveform'].shape)}, tokens {(batch['input_ids'].shape)}... ",
            "E1_STATUS: PHASE=validation_batch | epoch={epoch+1}/{int(training_args.num_train_epochs)} | step={global_step} | eval_batch={step} | waveform={batch['waveform'].shape} tokens={batch['input_ids'].shape}",
        ),
        (
            "VALIDATION - batch {step}, process{accelerator.process_index}, PADDING AND GATHER... ",
            "E1_STATUS: PHASE=validation_batch | epoch={epoch+1}/{int(training_args.num_train_epochs)} | step={global_step} | eval_batch={step} | action=pad_and_gather",
        ),
        (
            'logger.info("Validation inference done, now evaluating... ")',
            'logger.info(f"E1_STATUS: PHASE=validation_score | epoch={epoch+1}/{int(training_args.num_train_epochs)} | step={global_step} | action=plot_and_log_metrics")',
        ),
        (
            'logger.info("Validation finished... ")',
            'logger.info(f"E1_STATUS: PHASE=validation_done | epoch={epoch+1}/{int(training_args.num_train_epochs)} | step={global_step} | next=resume_training")',
        ),
    ]
    for old, repl in reps_all:
        if old not in new:
            raise SystemExit(f"status patch failed, missing snippet in {path}:\n{old[:160]}...")
        new = new.replace(old, repl)

    path.write_text(new, encoding="utf-8")
    print(f"Patched clear E1_STATUS logging in {path}")


def patch_finetune_plot_matplotlib(ft_root: Path) -> None:
    """Matplotlib 3.8+ removed FigureCanvasAgg.tostring_rgb — used at eval logging."""
    plot_py = ft_root / "utils" / "plot.py"
    if not plot_py.is_file():
        return
    text = plot_py.read_text(encoding="utf-8")
    if "buffer_rgba" in text:
        print(f"plot.py already matplotlib-safe: {plot_py}")
        return
    old = "data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)\n    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))"
    new = (
        "buf = np.asarray(fig.canvas.buffer_rgba())\n"
        "    data = buf[:, :, :3]"
    )
    if old not in text:
        raise SystemExit(f"Could not patch tostring_rgb in {plot_py}")
    plot_py.write_text(text.replace(old, new), encoding="utf-8")
    print(f"Patched matplotlib tostring_rgb → buffer_rgba in {plot_py}")


def patch_map_filter_cache_dir(cache_root: Path) -> None:
    """Force map/filter Arrow outputs under cache_root (not next to load_from_disk)."""
    from datasets import Dataset

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    counter = {"i": 0}

    def _next_cache() -> str:
        counter["i"] += 1
        # datasets num_proc>1 does cache_file_name.rindex(".") — must include a suffix.
        return str(cache_root / f"e1_ds_{counter['i']:03d}.arrow")

    def _wrap(orig):
        def wrapped(self, *args, **kwargs):
            if kwargs.get("cache_file_name") is None:
                kwargs["cache_file_name"] = _next_cache()
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
    patch_finetune_plot_matplotlib(ft)
    patch_vits_status_logging(ft)
    patch_load_dataset_for_disk()
    patch_map_filter_cache_dir(cache_root)

    # Resume if a prior checkpoint exists (e.g. crash at first eval).
    out_dir = Path(cfg["output_dir"])
    ckpts = sorted(
        (p for p in out_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    runtime_cfg = args.config
    resume_step = None
    if ckpts:
        resume_step = int(ckpts[-1].name.split("-")[-1])
        cfg["resume_from_checkpoint"] = str(ckpts[-1])
        runtime_cfg = Path(
            os.environ.get("TMPDIR") or tempfile.gettempdir()
        ) / "e1_mms_runtime_resume.json"
        runtime_cfg.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"Resuming from {ckpts[-1]}")

    print_run_banner(cfg, resume_step)
    print("E1_STATUS: PHASE=preprocess_or_load_cache | action=filter+map_audio_features")

    sys.path.insert(0, str(ft))
    os.chdir(ft)
    sys.argv = [str(train_script), str(runtime_cfg.resolve())]
    print(f"Launching {train_script} with {runtime_cfg}", flush=True)
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()

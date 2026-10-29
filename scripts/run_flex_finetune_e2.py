#!/usr/bin/env python3
"""Launch E2 Flex fine-tune with Weights & Biases logging.

DeepKIN has no native W&B hooks — we init a run and stream parsed
train/valid loss lines from the trainer stdout into wandb.log.

  python scripts/run_flex_finetune_e2.py
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

VALID_RE = re.compile(
    r"VALID LOSS:\s*([0-9.eE+-]+).*?BEST VALID LOSS:\s*([0-9.eE+-]+)",
    re.IGNORECASE,
)
ITER_RE = re.compile(r"Validation Iter:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
# generic "loss" float dumps from train logs
TRAIN_LOSS_RE = re.compile(
    r"(?:train\s+)?loss[^0-9\-]*([0-9]+\.[0-9]+(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--deepkin_root",
        type=Path,
        default=Path(
            os.environ.get(
                "DEEPKIN_ROOT",
                "/project/community/rmwisene/code/ac-ai-models/DeepKIN-AgAI",
            )
        ),
    )
    ap.add_argument(
        "--tts_data_dir",
        default=os.environ.get(
            "FLEX_DATA",
            "/project/community/rmwisene/tts_data_cleaning/kin_flex_data",
        ),
    )
    ap.add_argument(
        "--model_save_path",
        default=os.environ.get(
            "SAVE_CKPT",
            "/project/community/rmwisene/tts_baselines/finetune_e2_flex_kin/kinya_flex_tts_e2.pt",
        ),
    )
    ap.add_argument("--num_iters", type=int, default=int(os.environ.get("NUM_ITERS", "25000")))
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    ap.add_argument("--accumulation_steps", type=int, default=int(os.environ.get("ACCUM", "2")))
    ap.add_argument("--warmup_iter", type=int, default=int(os.environ.get("WARMUP", "1000")))
    ap.add_argument("--peak_lr", type=float, default=float(os.environ.get("PEAK_LR", "2e-4")))
    ap.add_argument(
        "--wandb_project",
        default=os.environ.get("WANDB_PROJECT", "healthtts"),
    )
    ap.add_argument(
        "--wandb_run_name",
        default=os.environ.get("WANDB_RUN_NAME", "E2-flex-kin"),
    )
    args = ap.parse_args()

    try:
        import wandb
    except ImportError as e:
        raise SystemExit("pip install wandb") from e

    cfg = {
        "experiment": "E2",
        "model": "kinya-flex",
        "num_iters": args.num_iters,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "warmup_iter": args.warmup_iter,
        "peak_lr": args.peak_lr,
        "tts_data_dir": args.tts_data_dir,
        "model_save_path": args.model_save_path,
    }
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=cfg,
        tags=["E2", "flex", "kin-only"],
    )

    cmd = [
        sys.executable,
        "deepkin/train/flex_trainer.py",
        '--model_variant=flex_tts:base',
        "--gpus=1",
        "--enable_amp=False",
        "--use_ddp=False",
        "--use_mtl_optimizer=False",
        "--num_losses=8",
        f"--warmup_iter={args.warmup_iter}",
        f"--peak_lr={args.peak_lr}",
        f"--num_iters={args.num_iters}",
        f"--batch_size={args.batch_size}",
        f"--accumulation_steps={args.accumulation_steps}",
        "--dataloader_num_workers=4",
        "--dataloader_persistent_workers=True",
        "--dataloader_pin_memory=True",
        "--use_iterable_dataset=False",
        "--train_log_steps=20",
        "--checkpoint_steps=1000",
        "--validation_steps=2000",
        "--load_saved_model=True",
        f"--tts_data_dir={args.tts_data_dir}",
        "--tts_train_data_file=tts_train_data.psv",
        f"--model_save_path={args.model_save_path}",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.deepkin_root}:{env.get('PYTHONPATH', '')}"

    print("Launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(args.deepkin_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    step = 0
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m_iter = ITER_RE.search(line)
        m_val = VALID_RE.search(line)
        if m_iter and m_val:
            step = int(m_iter.group(1))
            wandb.log(
                {
                    "valid/loss": float(m_val.group(1)),
                    "valid/best_loss": float(m_val.group(2)),
                    "train/iter": step,
                },
                step=step,
            )
            continue
        # opportunistic train loss scrape
        if "loss" in line.lower() and "valid" not in line.lower():
            m = TRAIN_LOSS_RE.search(line)
            if m:
                step += 1
                try:
                    wandb.log({"train/loss_scraped": float(m.group(1))}, step=step)
                except ValueError:
                    pass

    rc = proc.wait()
    wandb.summary["exit_code"] = rc
    run.finish()
    if rc != 0:
        raise SystemExit(rc)
    print("DONE_E2")


if __name__ == "__main__":
    main()

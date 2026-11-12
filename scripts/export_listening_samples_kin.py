#!/usr/bin/env python3
"""Export a small labeled Kin listening set (~10 wavs per system).

Clear names (no E1/E2 codes). Same utterance ids across systems when possible.

  python scripts/export_listening_samples_kin.py --n 10

Output:
  …/listening_eval/kinyarwanda_labeled/
    MMS_pretrained/
    MMS_Kin_only/
    MMS_bilingual_balanced/
    Flex_Kin_only/
    Flex_bilingual_balanced/
    manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

# Display name → synth dir (Kin pool DEV)
SYSTEMS = [
    (
        "MMS_pretrained",
        Path("/project/community/rmwisene/tts_baselines/eval_kin_dev/mms_b0/dev"),
    ),
    (
        "MMS_Kin_only",
        Path("/project/community/rmwisene/tts_baselines/eval_kin_dev/mms_e1/dev"),
    ),
    (
        "MMS_bilingual_balanced",
        Path("/project/community/rmwisene/tts_baselines/eval_kin/mms_e3/dev"),
    ),
    (
        "Flex_Kin_only",
        Path("/project/community/rmwisene/tts_baselines/eval_kin_dev/flex_e2/dev"),
    ),
    (
        "Flex_bilingual_balanced",
        Path("/project/community/rmwisene/tts_baselines/eval_kin/flex_e4/dev"),
    ),
]


def load_index(synth_dir: Path) -> dict[str, dict]:
    meta = synth_dir / "scores.tsv"
    if not meta.is_file():
        meta = synth_dir / "synthesis.tsv"
    if not meta.is_file():
        return {}
    with meta.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for r in rows:
        uid = (r.get("id") or r.get("utt_id") or "").strip()
        if not uid:
            continue
        hyp = (r.get("hyp_audio") or "").strip() or str(synth_dir / f"{uid}.wav")
        if not Path(hyp).is_file():
            hyp2 = synth_dir / f"{uid}.wav"
            if hyp2.is_file():
                hyp = str(hyp2)
            else:
                continue
        r = dict(r)
        r["_wav"] = hyp
        r["_uid"] = uid
        out[uid] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "/project/community/rmwisene/tts_baselines/listening_eval/kinyarwanda_labeled"
        ),
    )
    args = ap.parse_args()

    indexes: dict[str, dict[str, dict]] = {}
    for name, d in SYSTEMS:
        if not d.is_dir():
            print(f"SKIP (missing dir): {name} → {d}")
            continue
        idx = load_index(d)
        if not idx:
            # fallback: wav stems only
            for wav in sorted(d.glob("*.wav")):
                idx[wav.stem] = {"_wav": str(wav), "_uid": wav.stem, "text": ""}
        print(f"FOUND {name}: {len(idx)} clips @ {d}")
        indexes[name] = idx

    if not indexes:
        raise SystemExit("No systems available")

    # Prefer ids present in as many systems as possible
    id_sets = [set(idx) for idx in indexes.values()]
    common = set.intersection(*id_sets) if id_sets else set()
    rng = random.Random(args.seed)
    if len(common) >= args.n:
        chosen = sorted(common)
        rng.shuffle(chosen)
        chosen = chosen[: args.n]
        print(f"Using {len(chosen)} shared utterance ids across {len(indexes)} systems")
    else:
        # take from the largest available system, then copy overlaps
        best = max(indexes.items(), key=lambda kv: len(kv[1]))
        pool = list(best[1].keys())
        rng.shuffle(pool)
        chosen = pool[: args.n]
        print(
            f"WARNING: only {len(common)} fully shared ids; "
            f"sampling {len(chosen)} from {best[0]} (some systems may skip)"
        )

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    rows: list[dict] = []
    for name, idx in indexes.items():
        sys_dir = args.out_dir / name
        sys_dir.mkdir(parents=True)
        k = 0
        for uid in chosen:
            if uid not in idx:
                continue
            k += 1
            src = Path(idx[uid]["_wav"])
            # clear filename: System__01__uttid.wav
            dst = sys_dir / f"{name}__{k:02d}__{uid}.wav"
            shutil.copy2(src, dst)
            rows.append(
                {
                    "system": name,
                    "sample_num": k,
                    "utt_id": uid,
                    "text": (idx[uid].get("text") or "").strip(),
                    "wav": str(dst),
                }
            )
        print(f"  wrote {k} → {sys_dir}")

    man = args.out_dir / "manifest.csv"
    with man.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "sample_num", "utt_id", "text", "wav"])
        w.writeheader()
        w.writerows(rows)

    readme = args.out_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Kinyarwanda labeled listening samples (naturalness / UTMOS check)",
                "",
                "Folders = systems (clear names):",
                "  MMS_pretrained          — B0, no fine-tune",
                "  MMS_Kin_only            — fine-tuned on Kin only",
                "  MMS_bilingual_balanced  — Kin + balanced Dav",
                "  Flex_Kin_only           — fine-tuned on Kin only",
                "  Flex_bilingual_balanced — Kin + balanced Dav",
                "",
                "Same sample_num / utt_id = same sentence across folders (when available).",
                "Ask listeners to rate naturalness 1–5 and/or pick which folder sounds better.",
                "",
                f"n_per_system_target={args.n}  seed={args.seed}",
                f"manifest={man}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nDone → {args.out_dir}")
    print(f"manifest → {man}")


if __name__ == "__main__":
    main()

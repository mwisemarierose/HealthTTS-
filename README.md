# HealthTTS

Standalone TTS project (separate from HealthASR). Bilingual TTS for **Kinyarwanda** + **Kidaw’ida**.

This repo is **code only**. Datasets are not included — create them on shared storage when you set up a machine (e.g. Orchard).

## Repo layout

```text
HealthTTS/
├── configs/
├── training/
├── eval/
├── baselines/
├── scripts/
└── README.md
```

## Clone (Orchard)

```bash
cd ~
git clone https://github.com/mwisemarierose/HealthTTS-.git HealthTTS
cd HealthTTS
```

## Create dataset folders (required before download)

Data lives under project storage, **not** in this repo / not under home:

```bash
export DATA=/project/community/rmwisene/datasets
mkdir -p "$DATA"/{kinyarwanda_tts,kidawida_cv27,fleurs_kinyarwanda}
# or: bash scripts/setup_orchard_dirs.sh
```

Expected data layout after downloads:

```text
/project/community/rmwisene/datasets/
├── kinyarwanda_tts/       # train/adapt  (HF: mbazaNLP/kinyarwanda-tts-dataset)
├── kidawida_cv27/         # train/adapt  (Mozilla Data Collective)
└── fleurs_kinyarwanda/    # baseline eval only (HF: mbazaNLP/fleurs-kinyarwanda)
```

## Data roles (fixed)

| Dataset | Role | Notes |
|---------|------|--------|
| `kinyarwanda_tts` | **Train / adapt** (Kin) | No official splits — optional fixed held-out (e.g. 90/10, seed=42) for training checks only |
| `kidawida_cv27` | **Train / adapt** (Dav) | Use CV `train` / `dev` / `test` |
| `fleurs_kinyarwanda` | **Baseline evaluation only** (Kin) | Do **not** train or adapt on this set |

Nothing changes: **FLEURS stays baseline evaluation** for Kinyarwanda.

## Environment

```bash
conda create -n healthtts python=3.11 -y
conda activate healthtts
pip install -U "huggingface_hub[cli]" datasets
hf auth login
```

## Download datasets

### 1) Kinyarwanda TTS (Hugging Face)

```bash
conda activate healthtts
export DATA=/project/community/rmwisene/datasets

hf download mbazaNLP/kinyarwanda-tts-dataset \
  --repo-type dataset \
  --local-dir "$DATA/kinyarwanda_tts"
```

### 2) Kidaw’ida — Mozilla Data Collective (not Hugging Face)

```bash
export MDC_API_KEY='YOUR_API_KEY'   # never commit
export DATA=/project/community/rmwisene/datasets
mkdir -p "$DATA/kidawida_cv27"
cd "$DATA/kidawida_cv27"

RESPONSE=$(curl -sS -X POST \
  "https://mozilladatacollective.com/api/datasets/cmu5w2bh700aso107ffpduqpp/download" \
  -H "Authorization: Bearer ${MDC_API_KEY}" \
  -H "Content-Type: application/json")

DOWNLOAD_URL=$(echo "$RESPONSE" | jq -r '.downloadUrl')
curl -L --fail -o common-voice-scripted-speech-27-0-kidaw-69655f89.tar.gz "$DOWNLOAD_URL"
tar -xzf common-voice-scripted-speech-27-0-kidaw-69655f89.tar.gz
# optional: rm -f common-voice-scripted-speech-27-0-kidaw-69655f89.tar.gz
```

Or: `MDC_API_KEY=... bash scripts/download_kidawida_mdc.sh`

### 3) FLEURS Kinyarwanda — eval only (Hugging Face)

```bash
hf download mbazaNLP/fleurs-kinyarwanda \
  --repo-type dataset \
  --local-dir "$DATA/fleurs_kinyarwanda"
```

### Check

```bash
du -sh "$DATA"/kinyarwanda_tts "$DATA"/kidawida_cv27 "$DATA"/fleurs_kinyarwanda
ls "$DATA"
```

## Notes

- Kidaw’ida is from **Mozilla Data Collective**, not Hugging Face.
- **`fleurs_kinyarwanda` = baseline evaluation only** — never use it for training/adaptation.
- Never store large audio under `~/` (home quota); never commit API keys.

## Load baselines (Kin)

Checkpoints go on **project storage** (not in this git repo):

```text
/project/community/rmwisene/tts_baselines/
├── mms_tts_kin/          # facebook/mms-tts-kin
└── kinya_flex_tts/       # wixdivin/kinya-flex-tts  (kinya_flex_tts_base_trained.pt)
```

| Baseline | Hub | Role |
|----------|-----|------|
| MMS-TTS Kin | [facebook/mms-tts-kin](https://huggingface.co/facebook/mms-tts-kin) | Kin VITS baseline (Transformers) — eval on FLEURS |
| Kinya-Flex-TTS | [wixdivin/kinya-flex-tts](https://huggingface.co/wixdivin/kinya-flex-tts) | Kin Flex/VITS2 baseline (DeepKIN `.pt`) — eval on FLEURS |

### Download on Orchard

```bash
conda activate healthtts
cd ~/HealthTTS
pip install -r requirements.txt

export BASELINES=/project/community/rmwisene/tts_baselines
bash scripts/download_baselines.sh
```

### Smoke-test MMS-TTS Kin

```bash
export BASELINES=/project/community/rmwisene/tts_baselines
python scripts/smoke_mms_tts_kin.py \
  --text "Muraho. Amakuru yawe?" \
  --out /tmp/mms_tts_kin_smoke.wav
```

### Kinya-Flex-TTS note

Weights download with the script (`kinya_flex_tts_base_trained.pt`). Runtime inference uses **DeepKIN** (`FlexKinyaTTS`), not Transformers — wire that up after the weights are local. Speakers: `0` Female1, `1` Female2, `2` Male (24 kHz).

Dav has no public dedicated baseline yet; later adaptation can init from MMS Kin + `kidawida_cv27`.

## Clean FLEURS for baseline evaluation

Eval data only (`test` + `dev`). **Train split is never written into eval manifests.**

```bash
conda activate healthtts
cd ~/HealthTTS
# git pull   # after you push from Mac

# 1) see what's on disk (TSV headers, audio tars)
python scripts/inspect_fleurs.py

# 2) extract archives + write cleaned manifests
python scripts/clean_fleurs_eval.py --config configs/fleurs_eval.yaml
```

Outputs (on project):

```text
/project/community/rmwisene/datasets/fleurs_kinyarwanda/processed/eval/
├── test.tsv          # primary baseline eval
├── dev.tsv           # optional
└── summary.json      # kept / dropped counts
```

Cleaning rules: Unicode NFC, whitespace collapse, drop empty text, missing audio, and extreme durations (default 0.3–30s).

## Run MMS baseline on cleaned FLEURS (eval — not training)

FLEURS stays **evaluation only**. This synthesizes from cleaned `test.tsv` texts:

```bash
conda activate healthtts
export BASELINES=/project/community/rmwisene/tts_baselines
export HF_HOME=/project/community/rmwisene/.cache/huggingface

# prefer a GPU node for full test; login CPU ok for --limit 20
python scripts/eval_mms_fleurs.py --split test --limit 20
# full:
# python scripts/eval_mms_fleurs.py --split test
```

Outputs under `$BASELINES/eval_fleurs/mms_tts_kin/test/`.

**Adaptation / training** (later) uses `kinyarwanda_tts` + `kidawida_cv27`, not FLEURS.

## Agreed evaluation metrics

See `configs/metrics.yaml`.

1. **WER & CER (ASR loopback)** — Text → TTS → audio → ASR → text  
2. **Latency & RTF** — synth wall time and `synth_time / audio_duration`  
3. **Audio quality** — UTMOS (automatic MOS / naturalness)

### Score existing MMS FLEURS synths (GPU)

```bash
cd ~/HealthTTS && git pull
conda activate healthtts
pip install jiwer

export BASELINES=/project/community/rmwisene/tts_baselines
export HF_HOME=/project/community/rmwisene/.cache/huggingface

bash scripts/submit_score_tts_fleurs.sh
```

Results:

```text
$BASELINES/eval_fleurs/mms_tts_kin/test/metrics.json
$BASELINES/eval_fleurs/mms_tts_kin/test/scores.tsv
```

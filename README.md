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
git clone https://semanticservices.ghe.com/Marie-Rose-MWISENEZA/HealthTTS.git HealthTTS
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
└── fleurs_kinyarwanda/    # eval only    (HF: mbazaNLP/fleurs-kinyarwanda)
```

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
- Do **not** train on `fleurs_kinyarwanda` — baseline evaluation only.
- Never store large audio under `~/` (home quota); never commit API keys.

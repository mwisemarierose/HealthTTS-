#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import io, json, time, tempfile, os
import pandas as pd
import soundfile as sf
import requests
from huggingface_hub import HfApi, HfFileSystem, hf_hub_url

TARGET = 27.0
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
CHECKPOINT_DIR = Path("tts_hour_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "AfriSpeak Female": ("DigitalUmuganda/afrispeak_kinyarwanda_female_tts_dataset", "audio"),
    "AfriSpeak Male": ("DigitalUmuganda/afrispeak_kinyarwanda_male_tts_dataset", "audio"),
    "Bateesa": ("Bateesa/rw-tts-dataset", "parquet"),
}

AUDIO_WORKERS = 8
MAX_RETRIES = 5

def safe(name):
    return name.lower().replace(" ", "_")

def remote_duration(fs, path):
    err = None
    for attempt in range(MAX_RETRIES):
        try:
            with fs.open(path, "rb") as f:
                data = f.read()
            with sf.SoundFile(io.BytesIO(data)) as a:
                return len(a) / float(a.samplerate)
        except Exception as e:
            err = e
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(err)

def check_audio_repo(name, repo):
    api, fs = HfApi(), HfFileSystem()
    files = api.list_repo_files(repo, repo_type="dataset")
    audio_files = sorted(f for f in files if f.lower().endswith(AUDIO_EXTS))
    print(f"\n[{name}] audio files: {len(audio_files):,}")

    ckpt = CHECKPOINT_DIR / f"{safe(name)}.json"
    done = {}
    if ckpt.exists():
        try:
            done = {k: float(v) for k, v in json.loads(ckpt.read_text()).get("durations", {}).items()}
            print(f"[{name}] resuming from {len(done):,} measured files")
        except Exception:
            done = {}

    remaining = [f for f in audio_files if f not in done]
    failed = []

    def task(filename):
        p = f"datasets/{repo}/{filename}"
        return filename, remote_duration(fs, p)

    with ThreadPoolExecutor(max_workers=AUDIO_WORKERS) as ex:
        futures = {ex.submit(task, f): f for f in remaining}
        count = 0
        for fut in as_completed(futures):
            filename = futures[fut]
            try:
                f, secs = fut.result()
                done[f] = secs
            except Exception as e:
                failed.append(filename)
                print(f"[{name}] failed {filename}: {e}")
            count += 1

            if count % 100 == 0:
                hours = sum(done.values()) / 3600
                ckpt.write_text(json.dumps({"durations": done, "failed": failed}))
                print(f"[{name}] {len(done):,}/{len(audio_files):,} | {hours:.2f} h")

    ckpt.write_text(json.dumps({"durations": done, "failed": failed}))
    hours = sum(done.values()) / 3600

    print(f"[{name}] RESULT: {hours:.3f} h")
    return {"dataset": name, "clips": len(done), "failed": len(failed), "hours": hours}

def embedded_audio_duration(v):
    if isinstance(v, dict):
        if v.get("bytes") is not None:
            with sf.SoundFile(io.BytesIO(v["bytes"])) as a:
                return len(a) / float(a.samplerate)
        if v.get("path") and os.path.exists(v["path"]):
            i = sf.info(v["path"])
            return i.frames / float(i.samplerate)
    return None

def check_bateesa(name, repo):
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    pq = [f for f in files if f.endswith(".parquet") and "train" in Path(f).name.lower()]
    if not pq:
        pq = [f for f in files if f.endswith(".parquet")]
    if not pq:
        raise RuntimeError("No parquet file found")

    total = 0.0
    clips = 0
    failed = 0

    for filename in pq:
        print(f"\n[{name}] reading {filename}")
        url = hf_hub_url(repo, filename, repo_type="dataset")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as t:
            tmp = Path(t.name)

        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

            df = pd.read_parquet(tmp)
            print(f"[{name}] rows: {len(df):,}; columns: {list(df.columns)}")

            dur_col = next((c for c in ["duration","duration_s","duration_sec","duration_seconds","seconds"] if c in df.columns), None)
            if dur_col:
                vals = pd.to_numeric(df[dur_col], errors="coerce").dropna()
                total += vals.sum() / 1000 if len(vals) and vals.median() > 1000 else vals.sum()
                clips += len(vals)
            elif "audio" in df.columns:
                for i, v in enumerate(df["audio"], 1):
                    try:
                        d = embedded_audio_duration(v)
                        if d is None:
                            failed += 1
                        else:
                            total += d
                            clips += 1
                    except Exception:
                        failed += 1
                    if i % 500 == 0:
                        print(f"[{name}] {i:,}/{len(df):,} | {total/3600:.2f} h")
            else:
                raise RuntimeError("No duration or audio column")
        finally:
            tmp.unlink(missing_ok=True)

    hours = total / 3600
    print(f"[{name}] RESULT: {hours:.3f} h")
    return {"dataset": name, "clips": clips, "failed": failed, "hours": hours}

def run_one(name, repo, kind):
    return check_audio_repo(name, repo) if kind == "audio" else check_bateesa(name, repo)

def main():
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run_one, name, repo, kind): name for name, (repo, kind) in DATASETS.items()}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[{name}] ERROR: {e}")

    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    if not results:
        return

    df = pd.DataFrame(results).sort_values("dataset")
    print(df.to_string(index=False, formatters={"hours": lambda x: f"{x:.3f}"}))

    rows = df.to_dict("records")
    print("\nPAIRWISE TOTALS")
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            h = rows[i]["hours"] + rows[j]["hours"]
            status = "MEETS 27 h" if h >= TARGET else f"short by {TARGET-h:.3f} h"
            print(f"{rows[i]['dataset']} + {rows[j]['dataset']}: {h:.3f} h -> {status}")

    total = sum(r["hours"] for r in rows)
    status = f"MEETS target by {total-TARGET:.3f} h" if total >= TARGET else f"short by {TARGET-total:.3f} h"
    print(f"\nALL THREE: {total:.3f} h -> {status}")

if __name__ == "__main__":
    main()

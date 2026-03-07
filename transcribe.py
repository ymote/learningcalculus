#!/usr/bin/env python3
"""Transcribe all calculus lecture audio using local ASR endpoint."""

import base64
import json
import subprocess
import tempfile
import requests
from pathlib import Path

API = "http://localhost:8080/v1/audio/transcriptions"
AUDIO_DIR = Path("audio")
OUTPUT_DIR = Path("transcripts")
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SECS = 240  # 4 minutes per chunk


def get_duration(wav_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(wav_path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def transcribe_chunk(wav_path, language="zh"):
    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = requests.post(API, json={
        "file": audio_b64,
        "model": "qwen3-asr",
        "language": language,
    }, timeout=300)

    if resp.status_code != 200:
        print(f"    ERROR {resp.status_code}: {resp.text[:200]}")
        return ""

    return resp.json().get("text", "")


audio_files = sorted(AUDIO_DIR.glob("*.wav"))
print(f"Found {len(audio_files)} audio files\n")

for af in audio_files:
    out_path = OUTPUT_DIR / f"{af.stem}.txt"
    if out_path.exists():
        print(f"Skip (exists): {af.name}")
        continue

    duration = get_duration(af)
    num_chunks = max(1, int(duration // CHUNK_SECS) + (1 if duration % CHUNK_SECS > 0 else 0))
    print(f"Transcribing: {af.name} ({duration:.0f}s, {num_chunks} chunk(s))")

    all_text = []

    for i in range(num_chunks):
        start = i * CHUNK_SECS
        chunk_dur = min(CHUNK_SECS, duration - start)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(af),
                "-ss", str(start), "-t", str(chunk_dur),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                tmp.name
            ], capture_output=True, check=True)

            print(f"  Chunk {i+1}/{num_chunks} [{start:.0f}s - {start+chunk_dur:.0f}s] ...")
            text = transcribe_chunk(tmp.name)

        if text:
            all_text.append(text)
            print(f"    {text[:100]}...")

    full_text = "\n".join(all_text)
    out_path.write_text(full_text)
    print(f"  -> Saved to {out_path}\n")

print("All done!")

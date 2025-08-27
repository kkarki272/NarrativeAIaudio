# generate_audio.py — XTTS voice reference + safe long-form merge
import os
import re
import subprocess
from pathlib import Path
from TTS.api import TTS

# ---------- XTTS unpickling safelist (required for some xtts_v2 checkpoints) ----------
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

# XttsArgs can live in two places depending on TTS version
try:
    from TTS.tts.layers.xtts.gpt import XttsArgs
except Exception:
    from TTS.tts.models.xtts import XttsArgs  # fallback path on some builds

from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor  # <-- correct path for TTS 0.22.0

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs,
    SpeakerManager,
    AudioProcessor,
])
# --------------------------------------------------------------------------------------

# -------- Settings --------
INPUT_FILE = "story.txt"
OUTPUT_DIR = "output"
FINAL_WAV = os.path.join(OUTPUT_DIR, "final_narration.wav")
CHUNKS_DIR_NAME = "chunks"

# Use your reference voice here (20–60s of clean speech)
REFERENCE_VOICE = "voice_ref.wav"    # put this file next to this script
LANGUAGE = "en"
SPEED = 1.0                          # normal speed (try 0.95–0.85 for slower)

# Fallback speaker (used only if REFERENCE_VOICE not found)
FALLBACK_MODEL = "tts_models/en/vctk/vits"
FALLBACK_SPEAKER = "p225"            # try p227/p315 for male, p231/p248/p273 for female

# Chunking + audio normalization
MAX_CHARS = 4000
MAX_SENTENCES = 15
TARGET_SR, TARGET_CH, TARGET_CODEC = "44100", "1", "pcm_s16le"
KEEP_TEMP = False
# --------------------------

def split_sentences(text: str):
    """Basic sentence splitter that ensures each sentence ends with punctuation."""
    text = re.sub(r"\s+", " ", text.strip())
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s if re.search(r"[.!?]$", s) else s + "." for s in sents if s]

def make_chunks(sents, max_chars=MAX_CHARS, max_sents=MAX_SENTENCES):
    """Pack sentences into safe-sized chunks for TTS."""
    chunks, cur, n = [], [], 0
    for s in sents:
        if cur and (n + len(s) > max_chars or len(cur) >= max_sents):
            chunks.append(" ".join(cur))
            cur, n = [s], len(s)
        else:
            cur.append(s)
            n += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def ffmpeg_norm(src, dst):
    """Normalize sample rate/channels/bit-depth to a single consistent format."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", TARGET_SR, "-ac", TARGET_CH, "-c:a", TARGET_CODEC, dst],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=True,
    )

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base, INPUT_FILE)
    out_dir = os.path.join(base, OUTPUT_DIR)
    chunks_dir = os.path.join(out_dir, CHUNKS_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    # Load text
    text = Path(input_path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("story.txt is empty.")
    chunks = make_chunks(split_sentences(text))
    print(f"[TTS] chunks={len(chunks)} | speed={SPEED}")

    # Choose model path
    ref_path = os.path.join(base, REFERENCE_VOICE)
    use_xtts = os.path.exists(ref_path)
    if use_xtts:
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print(f"[TTS] Using XTTS v2 with reference: {REFERENCE_VOICE}")
    else:
        model_name = FALLBACK_MODEL
        print(f"[TTS] Reference not found → using VITS fallback: {FALLBACK_MODEL} | speaker={FALLBACK_SPEAKER}")

    # Load model
    tts = TTS(model_name)

    # Synthesize per chunk
    raw_wavs = []
    for i, chunk in enumerate(chunks, 1):
        out_wav = os.path.join(chunks_dir, f"chunk_{i:03d}.wav")
        print(f"[TTS] synth {i}/{len(chunks)} ({len(chunk)} chars)")
        if use_xtts:
            # XTTS: voice cloning with reference wav + language + speed
            tts.tts_to_file(
                text=chunk,
                language=LANGUAGE,
                speaker_wav=ref_path,
                speed=SPEED,
                file_path=out_wav,
            )
        else:
            # VITS fallback (no reference wav; use speaker ID)
            tts.tts_to_file(
                text=chunk,
                speaker=FALLBACK_SPEAKER,
                speed=SPEED,
                file_path=out_wav,
            )

        if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
            raise RuntimeError(f"Chunk {i} failed or is empty.")
        raw_wavs.append(out_wav)

    # Normalize + merge
    final = FINAL_WAV
    if len(raw_wavs) == 1:
        norm = raw_wavs[0].replace(".wav", "_norm.wav")
        ffmpeg_norm(raw_wavs[0], norm)
        if os.path.exists(final):
            os.remove(final)
        os.replace(norm, final)
    else:
        concat_txt = os.path.join(out_dir, "file_list.txt")
        norm_paths = []
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in raw_wavs:
                n = p.replace(".wav", "_norm.wav")
                ffmpeg_norm(p, n)
                f.write(f"file '{os.path.abspath(n)}'\n")
                norm_paths.append(n)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", final],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )
        os.remove(concat_txt)
        if not KEEP_TEMP:
            for n in norm_paths:
                if os.path.exists(n):
                    os.remove(n)

    # Cleanup chunk wavs
    if not KEEP_TEMP:
        for p in raw_wavs:
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(chunks_dir) and not os.listdir(chunks_dir):
            os.rmdir(chunks_dir)

    assert os.path.exists(final) and os.path.getsize(final) > 0, "Final file missing/empty."
    print(f"✅ Final: {final} ({os.path.getsize(final)} bytes)")

if __name__ == "__main__":
    main()

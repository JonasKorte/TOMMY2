"""
One-time model download script. Run this once while connected to the internet.
After completion the app runs 100% offline.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import MODELS_DIR, LLM_MODEL_PATH, STT_MODEL_DIR

LLM_DIR = MODELS_DIR / "llm"
STT_DIR = Path(STT_MODEL_DIR)

LLM_DIR.mkdir(parents=True, exist_ok=True)
STT_DIR.mkdir(parents=True, exist_ok=True)


def download_llm():
    from huggingface_hub import hf_hub_download
    if LLM_MODEL_PATH.exists():
        print(f"[LLM] Already downloaded: {LLM_MODEL_PATH.name}")
        return
    print("[LLM] Downloading Mistral-7B-Instruct-v0.3 Q4_K_M (~4.1 GB)...")
    hf_hub_download(
        repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        local_dir=str(LLM_DIR),
    )
    print("[LLM] Done.")


def download_stt():
    from huggingface_hub import snapshot_download
    marker = STT_DIR / "model.bin"
    if marker.exists():
        print("[STT] Already downloaded: faster-whisper-medium")
        return
    print("[STT] Downloading faster-whisper-medium (~1.5 GB)...")
    snapshot_download(
        repo_id="Systran/faster-whisper-medium",
        local_dir=str(STT_DIR),
    )
    print("[STT] Done.")


def download_tts():
    from TTS.api import TTS
    print("[TTS] Downloading Dutch voice model (nl/mai/tacotron2-DDC)...")
    TTS("tts_models/nl/mai/tacotron2-DDC", progress_bar=True)
    print("[TTS] Downloading English voice model (en/ljspeech/tacotron2-DDC)...")
    TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)
    print("[TTS] Done.")


if __name__ == "__main__":
    print("=" * 55)
    print("TOMMY2 — Model Download")
    print("This runs once. Keep internet connected until done.")
    print("=" * 55)
    download_llm()
    download_stt()
    download_tts()
    print("\nAll models downloaded. You can go offline now.")

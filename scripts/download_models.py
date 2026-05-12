"""
One-time model download script. Run this once while connected to the internet.
After completion the app runs 100% offline.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import (
    LLM_MODEL_PATH,
    MODELS_DIR,
    STT_MODEL_DIR,
    TTS_EN_ENGINE,
    TTS_EN_MODEL,
    TTS_NL_ENGINE,
    TTS_NL_MODEL,
    TTS_PIPER_DIR,
    TTS_PIPER_EN_VOICE,
    TTS_PIPER_NL_VOICE,
)

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


def _download_coqui(model_id: str):
    from TTS.api import TTS
    print(f"[TTS] Downloading Coqui model ({model_id})...")
    TTS(model_id, progress_bar=True)


def _download_piper(voice_id: str):
    from backend.tts.piper import _ensure_voice
    print(f"[TTS] Downloading Piper voice ({voice_id})...")
    _ensure_voice(voice_id, TTS_PIPER_DIR / voice_id)


def download_tts():
    # Only fetch what the active engine config actually needs.
    for lang, engine, coqui_model, piper_voice in (
        ("nl", TTS_NL_ENGINE, TTS_NL_MODEL, TTS_PIPER_NL_VOICE),
        ("en", TTS_EN_ENGINE, TTS_EN_MODEL, TTS_PIPER_EN_VOICE),
    ):
        if engine == "piper":
            _download_piper(piper_voice)
        elif engine == "coqui":
            _download_coqui(coqui_model)
        else:
            print(f"[TTS] Unknown engine {engine!r} for {lang}; skipping.")
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

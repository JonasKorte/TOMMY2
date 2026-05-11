import os
from faster_whisper import WhisperModel
from pathlib import Path
from backend.config import STT_MODEL_DIR


class STTEngine:
    def __init__(self):
        if not Path(STT_MODEL_DIR).exists():
            raise FileNotFoundError(
                f"STT model not found at {STT_MODEL_DIR}\n"
                "Run: python scripts/download_models.py"
            )
        # ctranslate2 probes the CUDA driver at model-load time even when device="cpu".
        # On Windows this can deadlock if llama-cpp already holds a CUDA context or the
        # driver is in a bad state. Hiding all GPUs from ctranslate2 here is safe because
        # llama-cpp's context was established earlier (app.py line 22 < line 23) and
        # CUDA_VISIBLE_DEVICES only affects *new* context creation.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.model = WhisperModel(
            STT_MODEL_DIR,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )

    def transcribe(self, audio_path: str, language: str = "nl") -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text for seg in segments).strip()

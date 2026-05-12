import threading
import wave
from pathlib import Path

from backend.config import TTS_PIPER_DIR, TTS_PIPER_EN_VOICE, TTS_PIPER_NL_VOICE

PIPER_REPO = "rhasspy/piper-voices"


def _voice_repo_path(voice_id: str) -> str:
    # rhasspy/piper-voices lays voices out as <lang>/<locale>/<dataset>/<quality>/<voice_id>.onnx
    # e.g. nl_NL-mls-medium  ->  nl/nl_NL/mls/medium/nl_NL-mls-medium.onnx
    locale, dataset, quality = voice_id.split("-", 2)
    lang = locale.split("_", 1)[0]
    return f"{lang}/{locale}/{dataset}/{quality}/{voice_id}"


def _ensure_voice(voice_id: str, dest_dir: Path) -> Path:
    """Returns the local .onnx path, downloading both .onnx and .onnx.json on first use."""
    from huggingface_hub import hf_hub_download

    repo_path = _voice_repo_path(voice_id)
    onnx_name = f"{voice_id}.onnx"
    json_name = f"{voice_id}.onnx.json"
    local_onnx = dest_dir / onnx_name
    local_json = dest_dir / json_name

    if not local_onnx.exists() or not local_json.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=PIPER_REPO,
            filename=f"{repo_path}.onnx",
            local_dir=str(dest_dir),
        )
        hf_hub_download(
            repo_id=PIPER_REPO,
            filename=f"{repo_path}.onnx.json",
            local_dir=str(dest_dir),
        )
        # hf_hub_download preserves the repo subpath; flatten so the file sits at dest_dir/<name>.
        nested_onnx = dest_dir / f"{repo_path}.onnx"
        nested_json = dest_dir / f"{repo_path}.onnx.json"
        if nested_onnx.exists() and not local_onnx.exists():
            nested_onnx.replace(local_onnx)
        if nested_json.exists() and not local_json.exists():
            nested_json.replace(local_json)

    return local_onnx


class PiperProvider:
    def __init__(self):
        self._voices: dict = {}
        self._locks: dict[str, threading.Lock] = {}
        self._load_lock = threading.Lock()

    def _voice_for(self, lang: str) -> str:
        return TTS_PIPER_NL_VOICE if lang == "nl" else TTS_PIPER_EN_VOICE

    def _get_voice(self, lang: str):
        voice_id = self._voice_for(lang)
        if voice_id in self._voices:
            return self._voices[voice_id], self._locks[voice_id]
        with self._load_lock:
            if voice_id not in self._voices:
                from piper import PiperVoice

                onnx_path = _ensure_voice(voice_id, TTS_PIPER_DIR / voice_id)
                self._voices[voice_id] = PiperVoice.load(str(onnx_path))
                self._locks[voice_id] = threading.Lock()
        return self._voices[voice_id], self._locks[voice_id]

    def synthesize(self, text: str, lang: str, output_path: str) -> None:
        voice, lock = self._get_voice(lang)
        # PiperVoice wraps a single ONNX session that isn't safe to drive from
        # multiple threads at once. Serialize per-voice.
        with lock:
            with wave.open(output_path, "wb") as wav:
                voice.synthesize(text, wav)

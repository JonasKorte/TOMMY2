from backend.config import TTS_EN_ENGINE, TTS_NL_ENGINE
from backend.tts.base import TTSProvider
from backend.tts.coqui import CoquiProvider
from backend.tts.piper import PiperProvider

_PROVIDER_CLASSES = {
    "coqui": CoquiProvider,
    "piper": PiperProvider,
}


class TTSEngine:
    def __init__(self):
        self._providers: dict[str, TTSProvider] = {}

    def _engine_name(self, lang: str) -> str:
        name = TTS_NL_ENGINE if lang == "nl" else TTS_EN_ENGINE
        if name not in _PROVIDER_CLASSES:
            raise ValueError(
                f"Unknown TTS engine {name!r} for lang={lang!r}. "
                f"Valid options: {sorted(_PROVIDER_CLASSES)}"
            )
        return name

    def _get_provider(self, lang: str) -> TTSProvider:
        name = self._engine_name(lang)
        if name not in self._providers:
            self._providers[name] = _PROVIDER_CLASSES[name]()
        return self._providers[name]

    def synthesize(self, text: str, lang: str, output_path: str):
        self._get_provider(lang).synthesize(text, lang, output_path)

from backend.config import TTS_EN_MODEL, TTS_NL_MODEL


class CoquiProvider:
    def __init__(self):
        self._engines: dict = {}

    def _load(self, lang: str):
        from TTS.api import TTS

        model_name = TTS_NL_MODEL if lang == "nl" else TTS_EN_MODEL
        self._engines[lang] = TTS(model_name, progress_bar=False, gpu=False)

    def synthesize(self, text: str, lang: str, output_path: str) -> None:
        if lang not in self._engines:
            self._load(lang)
        self._engines[lang].tts_to_file(text=text, file_path=output_path)

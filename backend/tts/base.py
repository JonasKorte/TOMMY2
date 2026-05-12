from typing import Protocol


class TTSProvider(Protocol):
    def synthesize(self, text: str, lang: str, output_path: str) -> None:
        ...

SUPPORTED = {"nl", "en"}


class LanguageState:
    def __init__(self, default: str = "nl"):
        self.current = default

    def toggle(self):
        self.current = "en" if self.current == "nl" else "nl"

    def set(self, lang: str):
        if lang not in SUPPORTED:
            raise ValueError(f"Unsupported language '{lang}'. Choose from: {SUPPORTED}")
        self.current = lang

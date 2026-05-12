import os
from pathlib import Path

BASE = Path(__file__).parent.parent
MODELS_DIR = BASE / "models"
TEMP_DIR = BASE / "temp"
MEMORY_DIR = BASE / "memory"

LLM_MODEL_PATH = MODELS_DIR / "llm" / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
STT_MODEL_DIR = str(MODELS_DIR / "stt" / "faster-whisper-medium")
PERSONALITY_FILE = BASE / "personality" / "system_prompt.txt"
HISTORY_FILE = MEMORY_DIR / "history.jsonl"
PROFILE_FILE = MEMORY_DIR / "profile.json"

HISTORY_CONTEXT_TURNS = 20
PROFILE_UPDATE_EVERY = 4

# Coqui model IDs — used when the corresponding engine is set to "coqui".
TTS_NL_MODEL = "tts_models/nl/css10/vits"
TTS_EN_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

# Which engine drives each language. "piper" sounds far more natural in Dutch
# than the default Coqui CSS10 VITS, which slurs and has a strong accent.
# Override via env var to A/B without touching code.
TTS_NL_ENGINE = os.getenv("TTS_NL_ENGINE", "piper").lower()
TTS_EN_ENGINE = os.getenv("TTS_EN_ENGINE", "coqui").lower()

# Piper voice IDs follow the rhasspy/piper-voices repo convention:
#   <locale>-<dataset>-<quality>   e.g. nl_NL-mls-medium
# Good Dutch options:
#   nl_NL-mls-medium      — Netherlands NL, multi-speaker, best quality/size tradeoff
#   nl_NL-mls_5809-low    — Netherlands NL, single speaker, smallest/fastest
#   nl_BE-nathalie-medium — Flemish (Belgium)
#   nl_BE-rdh-medium      — Flemish (Belgium)
TTS_PIPER_NL_VOICE = os.getenv("TTS_PIPER_NL_VOICE", "nl_NL-mls-medium")
TTS_PIPER_EN_VOICE = os.getenv("TTS_PIPER_EN_VOICE", "en_US-lessac-medium")
TTS_PIPER_DIR = MODELS_DIR / "tts" / "piper"

LLM_CONTEXT_WINDOW = 4096
LLM_N_THREADS = 8
# Mistral-7B has 32 transformer layers + output. -1 offloads everything to GPU.
# Falls back to CPU automatically if llama-cpp-python was built without CUDA.
LLM_N_GPU_LAYERS = 0

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

TEMP_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)

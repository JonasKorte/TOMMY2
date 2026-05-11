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

TTS_NL_MODEL = "tts_models/nl/css10/vits"
TTS_EN_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

LLM_CONTEXT_WINDOW = 4096
LLM_N_THREADS = 8
# Mistral-7B has 32 transformer layers + output. -1 offloads everything to GPU.
# Falls back to CPU automatically if llama-cpp-python was built without CUDA.
LLM_N_GPU_LAYERS = 0

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

TEMP_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)

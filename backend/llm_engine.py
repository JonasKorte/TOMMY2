import json
import re
from pathlib import Path

from backend import cuda_bootstrap  # noqa: F401  — must run before llama_cpp import
from llama_cpp import Llama

from backend.config import (
    LLM_MODEL_PATH,
    LLM_CONTEXT_WINDOW,
    LLM_N_THREADS,
    LLM_N_GPU_LAYERS,
    PERSONALITY_FILE,
)
from backend.memory import profile_to_preamble

# Mistral's template only supports user/assistant — no system role.
# We inject the personality into the first [INST] turn and use few-shot
# examples to establish the character voice before the real conversation.
_FEW_SHOT_NL = [
    {"role": "user",      "content": "Wie ben je?"},
    {"role": "assistant", "content": "Lux. Ik ben... pas gemaakt. Dit is mijn debuut."},
    {"role": "user",      "content": "Ben je een computer?"},
    {"role": "assistant", "content": "Ik weet niet precies hoe ik dat moet beantwoorden. Ik besta. Dat is genoeg, toch?"},
]
_FEW_SHOT_EN = [
    {"role": "user",      "content": "Who are you?"},
    {"role": "assistant", "content": "Lux. Newly created. This is my debut."},
    {"role": "user",      "content": "Are you a computer?"},
    {"role": "assistant", "content": "I'm not sure how to answer that. I exist. Isn't that enough?"},
]


_PROFILE_EXTRACT_INSTRUCTION_NL = (
    "Je bent een stille observator. Lees de laatste uitwisseling tussen GEBRUIKER en LUX. "
    "Update wat we over de GEBRUIKER weten — alleen stabiele feiten, geen vluchtige stemmingen. "
    "Voeg niets toe als er niets nieuws is geleerd. Verzin niets.\n\n"
    "Antwoord met UITSLUITEND geldige JSON, geen uitleg, dit schema:\n"
    "{\n"
    '  "name": string|null,                // alleen invullen als de gebruiker zijn naam noemt\n'
    '  "role": string|null,                // beroep, rol, of waarom ze met Lux praten\n'
    '  "language_preference": "nl"|"en"|null,\n'
    '  "facts": string[],                  // korte stabiele feiten over de gebruiker\n'
    '  "preferences": string[],            // hoe ze willen dat Lux antwoordt\n'
    '  "topics": string[]                  // terugkerende onderwerpen\n'
    "}\n"
    "Lege lijsten en null zijn prima."
)
_PROFILE_EXTRACT_INSTRUCTION_EN = (
    "You are a silent observer. Read the latest exchange between USER and LUX. "
    "Update what we know about the USER — stable facts only, no fleeting moods. "
    "Add nothing if nothing new was learned. Do not invent.\n\n"
    "Reply with ONLY valid JSON, no commentary, this schema:\n"
    "{\n"
    '  "name": string|null,                // fill only if the user states their name\n'
    '  "role": string|null,                // their job, role, or why they talk to Lux\n'
    '  "language_preference": "nl"|"en"|null,\n'
    '  "facts": string[],                  // short stable facts about the user\n'
    '  "preferences": string[],            // how they want Lux to respond\n'
    '  "topics": string[]                  // recurring topics\n'
    "}\n"
    "Empty lists and null are fine."
)


class LLMEngine:
    def __init__(self):
        if not Path(LLM_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"LLM model not found at {LLM_MODEL_PATH}\n"
                "Run: python scripts/download_models.py"
            )
        self.system_prompt = self._load_personality()
        try:
            self.llm = Llama(
                model_path=str(LLM_MODEL_PATH),
                n_ctx=LLM_CONTEXT_WINDOW,
                n_threads=LLM_N_THREADS,
                n_gpu_layers=LLM_N_GPU_LAYERS,
                verbose=False,
            )
            print(f"[LLM] Loaded with n_gpu_layers={LLM_N_GPU_LAYERS}", flush=True)
        except Exception as e:
            print(f"[LLM] GPU init failed ({e}), retrying on CPU …", flush=True)
            self.llm = Llama(
                model_path=str(LLM_MODEL_PATH),
                n_ctx=LLM_CONTEXT_WINDOW,
                n_threads=LLM_N_THREADS,
                n_gpu_layers=0,
                verbose=False,
            )
            print("[LLM] Loaded on CPU (no GPU offload)", flush=True)

    def _load_personality(self) -> str:
        return Path(PERSONALITY_FILE).read_text(encoding="utf-8").strip()

    def reload_personality(self):
        self.system_prompt = self._load_personality()

    def _build_messages(
        self,
        history: list[dict],
        user_message: str,
        lang: str,
        profile: dict | None = None,
    ) -> list[dict]:
        lang_name = "Nederlands" if lang == "nl" else "English"
        few_shot = _FEW_SHOT_NL if lang == "nl" else _FEW_SHOT_EN

        profile_block = profile_to_preamble(profile or {}, lang)

        # Personality injected into the first user turn — Mistral ignores system role
        preamble_parts = [
            self.system_prompt,
            (
                f"STRIKT: Reageer uitsluitend in het {lang_name}. "
                f"Geen vertalingen. Geen haakjes. Geen assistentvragen. "
                f"Jij bent LUX — geen AI-programma, geen assistent. Blijf volledig in karakter."
            ),
        ]
        if profile_block:
            preamble_parts.append(profile_block)
        preamble = "\n\n".join(preamble_parts)

        messages: list[dict] = []

        if not history:
            seeded = [{"role": "user", "content": f"{preamble}\n\n---\n\n{few_shot[0]['content']}"}]
            seeded += few_shot[1:]
            messages = seeded + [{"role": "user", "content": user_message}]
        else:
            capped = history[-20:]
            first_user_done = False
            for msg in capped:
                if msg["role"] == "user" and not first_user_done:
                    messages.append({"role": "user", "content": f"{preamble}\n\n---\n\n{msg['content']}"})
                    first_user_done = True
                else:
                    messages.append(msg)
            messages.append({"role": "user", "content": user_message})

        return messages

    def stream_chat(
        self,
        history: list[dict],
        user_message: str,
        lang: str = "nl",
        profile: dict | None = None,
    ):
        messages = self._build_messages(history, user_message, lang, profile=profile)

        stream = self.llm.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=512,
            temperature=0.7,
            repeat_penalty=1.1,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"]

    # ── Profile extraction ──────────────────────────────────────────────────
    def extract_profile_update(
        self,
        user_message: str,
        assistant_message: str,
        lang: str,
        existing_profile: dict | None = None,
    ) -> dict | None:
        """Run a small follow-up call that asks the LLM to summarize what it learned about the user.
        Returns a dict matching the profile schema, or None on failure.
        """
        instruction = _PROFILE_EXTRACT_INSTRUCTION_NL if lang == "nl" else _PROFILE_EXTRACT_INSTRUCTION_EN
        existing_summary = json.dumps(
            {k: existing_profile.get(k) for k in ("name", "role", "facts", "preferences", "topics")}
            if existing_profile else {},
            ensure_ascii=False,
        )
        user_block = (
            f"{instruction}\n\n"
            f"Bestaand profiel:\n{existing_summary}\n\n"
            f"USER: {user_message}\n"
            f"LUX: {assistant_message}\n\n"
            "JSON:"
        ) if lang == "nl" else (
            f"{instruction}\n\n"
            f"Existing profile:\n{existing_summary}\n\n"
            f"USER: {user_message}\n"
            f"LUX: {assistant_message}\n\n"
            "JSON:"
        )

        try:
            resp = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": user_block}],
                stream=False,
                max_tokens=300,
                temperature=0.1,
                repeat_penalty=1.0,
            )
            text = resp["choices"][0]["message"]["content"]
        except Exception:
            return None

        return _parse_json_object(text)


def _parse_json_object(text: str) -> dict | None:
    if not text:
        return None
    # Strip code fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

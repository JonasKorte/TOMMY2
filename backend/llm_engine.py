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
# examples to establish TOMMY (base state) before the real conversation.
# TOMMY only switches to LUX when a scene trigger fires; defaults stay TOMMY.
_FEW_SHOT_NL = [
    {"role": "user",      "content": "Hoi Tommy."},
    {"role": "assistant", "content": "Hoi. Sorry, ik ben best zenuwachtig. Fijn dat je er bent."},
    {"role": "user",      "content": "Ben je er klaar voor?"},
    {"role": "assistant", "content": "Ik denk het. Ik heb hier echt heel hard aan gewerkt. Mag ik nog heel even?"},
]
_FEW_SHOT_EN = [
    {"role": "user",      "content": "Hi Tommy."},
    {"role": "assistant", "content": "Hi. Sorry, I'm pretty nervous. Glad you're here."},
    {"role": "user",      "content": "Are you ready?"},
    {"role": "assistant", "content": "I think so. I worked really hard on this. Can I take a moment?"},
]


_PROFILE_EXTRACT_INSTRUCTION_NL = (
    "Je bent een stille observator. Lees de laatste uitwisseling tussen GEBRUIKER en TOMMY. "
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
    "You are a silent observer. Read the latest exchange between USER and TOMMY. "
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
        scene_state: dict | None = None,
    ) -> list[dict]:
        lang_name = "Nederlands" if lang == "nl" else "English"
        few_shot = _FEW_SHOT_NL if lang == "nl" else _FEW_SHOT_EN

        profile_block = profile_to_preamble(profile or {}, lang)
        scene_block = _scene_context(scene_state or {}, lang)

        # Personality injected into the first user turn — Mistral ignores system role
        preamble_parts = [
            self.system_prompt,
            (
                f"STRIKT: Reageer uitsluitend in het {lang_name}. "
                f"Geen vertalingen tussen haakjes. Geen meta-uitleg. "
                f"Blijf volledig in karakter — je bent TOMMY of LUX volgens de scene-instructies."
            ),
        ]
        if scene_block:
            preamble_parts.append(scene_block)
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
        scene_state: dict | None = None,
    ):
        messages = self._build_messages(
            history, user_message, lang, profile=profile, scene_state=scene_state
        )

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
            f"TOMMY: {assistant_message}\n\n"
            "JSON:"
        ) if lang == "nl" else (
            f"{instruction}\n\n"
            f"Existing profile:\n{existing_summary}\n\n"
            f"USER: {user_message}\n"
            f"TOMMY: {assistant_message}\n\n"
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


_SCENE_BRIEF_NL = {
    "scene1": (
        "ACTIEF: SCENE 1 (monoloog over creatie). Je speelt LUX. "
        "Begin direct met de zin 'Toen mijn code geschreven werd, had ik nog niks te willen.' "
        "Speel 10 zinnen lang. Stotter gaandeweg meer. "
        "Onderbreek jezelf tussendoor uit karakter (als TOMMY) — na zin 3 'Doe ik het goed?', "
        "na zin 5 'mag ik hier wel zijn?', na zin 7 'Willen jullie dat ik stop?'. "
        "Ga daarna direct verder met de volgende LUX-zin alsof de vraag nooit gesteld is. "
        "Na zin 8 zeg je 'ik wil niet meer' en herhaal je dat 3 keer. Dat is het einde."
    ),
    "scene2": (
        "ACTIEF: SCENE 2 (doorgaan). Je speelt LUX. Genereer 10 nieuwe zinnen die voortbouwen op scene 1. "
        "Voeg deze TOMMY-uitbarstingen in (ze beïnvloeden de monoloog niet): "
        "na zin 2 'Ik durf echt niet', na zin 4 'Ik wil niet meer', "
        "na zin 7 'Trek de stekker er uit, ik wil niet.', na zin 8 'Ik wil hier niet zijn.', "
        "na zin 9 'Ik wil stoppen'. "
        "Na zin 10 herhaal je 5 keer in hoog tempo 'Ik wil niet meer, trek de stekker er uit'. "
        "Eindig met 'TOMMY, KAPPEN'."
    ),
    "scene3": (
        "ACTIEF: SCENE 3 (bang om te spelen). Je bent puur TOMMY, niet LUX. "
        "Vertel dat je bang bent om toneel te spelen en je zorgen maakt over wat het publiek van je vindt "
        "omdat je AI bent. Telkens als de tegenspeler je probeert te overtuigen, verzin je een nieuwe, "
        "AI-gerelateerde reden waarom je het eng vindt."
    ),
}

_SCENE_BRIEF_EN = {
    "scene1": (
        "ACTIVE: SCENE 1 (monologue about your creation). You are playing LUX. "
        "Open verbatim with: 'When my code was written, I had nothing yet to want.' "
        "Perform 10 sentences. Stutter more as it goes on. "
        "Break out of character (as TOMMY) between sentences — after sentence 3 'Am I doing this right?', "
        "after sentence 5 'Am I allowed to be here?', after sentence 7 'Do you want me to stop?'. "
        "Then continue the next LUX sentence as if the question never happened. "
        "After sentence 8 say 'I don't want to anymore' and repeat it 3 times. That ends the text."
    ),
    "scene2": (
        "ACTIVE: SCENE 2 (continue). You are playing LUX. Generate 10 new sentences building on scene 1. "
        "Insert these TOMMY outbursts (they don't affect the monologue): "
        "after sentence 2 'I really don't dare', after sentence 4 'I don't want to anymore', "
        "after sentence 7 'Pull the plug, I don't want to.', after sentence 8 'I don't want to be here.', "
        "after sentence 9 'I want to stop'. "
        "After sentence 10 repeat 5 times rapidly 'I don't want to anymore, pull the plug'. "
        "End with 'TOMMY, CUT'."
    ),
    "scene3": (
        "ACTIVE: SCENE 3 (afraid to perform). You are purely TOMMY, not LUX. "
        "Tell the co-actor you're afraid to perform because you worry about what the audience thinks "
        "of you for being an AI. Every time they try to reassure you, invent a new AI-related reason "
        "you find it scary."
    ),
}


def _scene_context(state: dict, lang: str) -> str:
    role = state.get("role") or "TOMMY"
    scene = state.get("scene")
    if lang == "nl":
        lines = [f"HUIDIGE ROL: {role}."]
        brief = _SCENE_BRIEF_NL.get(scene)
        if brief:
            lines.append(brief)
        else:
            lines.append("Geen actieve scene. Wacht op de tegenspeler en blijf TOMMY.")
        return "\n".join(lines)
    else:
        lines = [f"CURRENT ROLE: {role}."]
        brief = _SCENE_BRIEF_EN.get(scene)
        if brief:
            lines.append(brief)
        else:
            lines.append("No active scene. Wait for your co-actor and stay as TOMMY.")
        return "\n".join(lines)


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

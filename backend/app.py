import json
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_sock import Sock

from backend.config import FLASK_HOST, FLASK_PORT, HISTORY_CONTEXT_TURNS, PROFILE_UPDATE_EVERY, TEMP_DIR
from backend.language_state import LanguageState
from backend.llm_engine import LLMEngine
from backend.memory import Memory
from backend.scene_state import SceneState
from backend.stt_engine import STTEngine
from backend.tts_engine import TTSEngine

FRONTEND_DIR = str(Path(__file__).parent.parent / "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
sock = Sock(app)

state = LanguageState(default="nl")
llm = LLMEngine()
stt = STTEngine()
tts = TTSEngine()
memory = Memory()
scene = SceneState()


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "lang": state.current})


@app.route("/language", methods=["GET"])
def get_language():
    return jsonify({"lang": state.current})


@app.route("/language", methods=["POST"])
def set_language():
    data = request.get_json()
    lang = data.get("lang")
    try:
        state.set(lang)
        return jsonify({"lang": state.current})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({"history": memory.load_history(), "profile": memory.load_profile()})


@app.route("/history/clear", methods=["POST"])
def clear_history():
    memory.clear_history()
    return jsonify({"status": "ok"})


@app.route("/profile", methods=["GET"])
def get_profile():
    return jsonify(memory.load_profile())


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    suffix = Path(audio_file.filename).suffix if audio_file.filename else ".webm"
    tmp_path = TEMP_DIR / f"{uuid.uuid4()}{suffix}"

    try:
        audio_file.save(str(tmp_path))
        text = stt.transcribe(str(tmp_path), language=state.current)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)


@sock.route("/ws/chat")
def chat(ws):
    raw = ws.receive()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        ws.send(json.dumps({"error": "Invalid JSON"}))
        return

    user_message = data.get("message", "").strip()
    lang = data.get("lang", state.current)

    if not user_message:
        ws.send("[DONE]")
        return

    # Server owns history now — pull recent context from disk.
    history = memory.recent_history(HISTORY_CONTEXT_TURNS)
    profile = memory.load_profile()

    # Detect performance cues ("TOMMY KAPPEN", scene triggers, …) before the LLM
    # sees the message, so the role/scene injected into the prompt is up to date.
    trigger = scene.apply_trigger(user_message)
    scene_snapshot = scene.snapshot()
    if trigger:
        print(f"[scene] trigger={trigger} → {scene_snapshot}", flush=True)

    full_response_parts: list[str] = []
    try:
        for token in llm.stream_chat(
            history, user_message, lang=lang, profile=profile, scene_state=scene_snapshot
        ):
            full_response_parts.append(token)
            ws.send(token)
        ws.send("[DONE]")
    except Exception as e:
        ws.send(f"[ERROR] {e}")
        return

    assistant_message = "".join(full_response_parts).strip()
    if not assistant_message:
        return

    # Persist the turn synchronously, then update the learned profile in the background
    # so the websocket can close immediately.
    memory.append_turn(user_message, assistant_message)
    turn_count = (profile.get("turn_count") or 0) + 1
    if turn_count % PROFILE_UPDATE_EVERY == 0:
        threading.Thread(
            target=_update_profile_async,
            args=(user_message, assistant_message, lang, profile),
            daemon=True,
        ).start()
    else:
        # Still bump turn_count so the cadence is honored
        memory.merge_profile(None)


def _update_profile_async(user_message: str, assistant_message: str, lang: str, profile: dict) -> None:
    try:
        update = llm.extract_profile_update(user_message, assistant_message, lang, existing_profile=profile)
        memory.merge_profile(update)
    except Exception:
        # Memory updates are best-effort — never crash the chat path.
        pass


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json()
    text = data.get("text", "").strip()
    lang = data.get("lang", state.current)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    tmp_path = TEMP_DIR / f"{uuid.uuid4()}.wav"
    try:
        tts.synthesize(text, lang, str(tmp_path))
        return send_file(str(tmp_path), mimetype="audio/wav")
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


@app.route("/reload-personality", methods=["POST"])
def reload_personality():
    llm.reload_personality()
    return jsonify({"status": "ok"})


@app.route("/scene", methods=["GET"])
def get_scene():
    return jsonify(scene.snapshot())


@app.route("/scene", methods=["POST"])
def set_scene():
    data = request.get_json(silent=True) or {}
    if data.get("reset"):
        return jsonify(scene.reset())
    try:
        return jsonify(scene.set(role=data.get("role"), scene=data.get("scene")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)

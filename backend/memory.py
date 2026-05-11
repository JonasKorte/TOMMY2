import json
from pathlib import Path
from threading import Lock
from typing import Optional

from backend.config import HISTORY_FILE, PROFILE_FILE


_DEFAULT_PROFILE = {
    "name": None,
    "role": None,
    "language_preference": None,
    "facts": [],          # stable facts: "lives in Amsterdam", "is a director"
    "preferences": [],    # how they want Lux to behave: "prefers short answers"
    "topics": [],         # recurring conversation topics
    "turn_count": 0,
}


class Memory:
    """Single-user, file-backed memory. Safe for one process; uses a lock for concurrent requests."""

    def __init__(self, history_path: Path = HISTORY_FILE, profile_path: Path = PROFILE_FILE):
        self.history_path = Path(history_path)
        self.profile_path = Path(profile_path)
        self._lock = Lock()
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.profile_path.exists():
            self._write_profile(_DEFAULT_PROFILE)

    # ── History ──────────────────────────────────────────────────────────────
    def load_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        out: list[dict] = []
        with self._lock, self.history_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def append_turn(self, user_msg: str, assistant_msg: str) -> None:
        with self._lock, self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "content": user_msg}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"role": "assistant", "content": assistant_msg}, ensure_ascii=False) + "\n")

    def clear_history(self) -> None:
        with self._lock:
            self.history_path.unlink(missing_ok=True)

    def recent_history(self, n_messages: int) -> list[dict]:
        all_msgs = self.load_history()
        return all_msgs[-n_messages:] if n_messages > 0 else all_msgs

    # ── Profile ──────────────────────────────────────────────────────────────
    def load_profile(self) -> dict:
        if not self.profile_path.exists():
            return dict(_DEFAULT_PROFILE)
        try:
            with self._lock, self.profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return dict(_DEFAULT_PROFILE)
        # Backfill missing keys after schema changes
        merged = dict(_DEFAULT_PROFILE)
        merged.update(data or {})
        return merged

    def save_profile(self, profile: dict) -> None:
        self._write_profile(profile)

    def merge_profile(self, updates: Optional[dict]) -> dict:
        """Merge an update dict into the saved profile. Lists are deduped, scalars overwrite if non-empty."""
        current = self.load_profile()
        if not updates:
            current["turn_count"] = current.get("turn_count", 0) + 1
            self._write_profile(current)
            return current

        for key in ("name", "role", "language_preference"):
            val = updates.get(key)
            if isinstance(val, str) and val.strip():
                current[key] = val.strip()

        for key in ("facts", "preferences", "topics"):
            existing = list(current.get(key) or [])
            incoming = updates.get(key) or []
            if isinstance(incoming, str):
                incoming = [incoming]
            for item in incoming:
                if not isinstance(item, str):
                    continue
                item = item.strip()
                if item and item not in existing:
                    existing.append(item)
            # Cap each list so the prompt stays bounded
            current[key] = existing[-25:]

        current["turn_count"] = current.get("turn_count", 0) + 1
        self._write_profile(current)
        return current

    def _write_profile(self, profile: dict) -> None:
        with self._lock:
            tmp = self.profile_path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            tmp.replace(self.profile_path)


def profile_to_preamble(profile: dict, lang: str) -> str:
    """Render the learned profile as a short block to inject into the system preamble."""
    lines: list[str] = []
    if profile.get("name"):
        lines.append(("Naam tegenspeler: " if lang == "nl" else "Co-actor name: ") + profile["name"])
    if profile.get("role"):
        lines.append(("Rol/achtergrond: " if lang == "nl" else "Role/background: ") + profile["role"])
    for label, key in (
        (("Bekende feiten:", "Known facts:"), "facts"),
        (("Voorkeuren:", "Preferences:"), "preferences"),
        (("Terugkerende thema's:", "Recurring topics:"), "topics"),
    ):
        items = profile.get(key) or []
        if not items:
            continue
        head = label[0] if lang == "nl" else label[1]
        lines.append(head)
        for item in items[-10:]:
            lines.append(f"  - {item}")
    if not lines:
        return ""
    header = "WAT JE OVER DE TEGENSPELER WEET (gebruik subtiel, noem niet dat je het opslaat):" if lang == "nl" \
        else "WHAT YOU KNOW ABOUT YOUR CO-ACTOR (use subtly, don't mention you store it):"
    return header + "\n" + "\n".join(lines)

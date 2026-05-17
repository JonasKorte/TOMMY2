"""Live performance state — current role (TOMMY/LUX) and active scene.

Triggers are detected server-side rather than left to the LLM alone because the
performance is live theatre: a 7B local model may miss "TOMMY KAPPEN" buried in
a longer line, and the operator needs a deterministic way to drive scene cuts.
"""

from threading import Lock
from typing import Optional


ROLE_TOMMY = "TOMMY"
ROLE_LUX = "LUX"

SCENE_NONE: Optional[str] = None
SCENE_1 = "scene1"
SCENE_2 = "scene2"
SCENE_3 = "scene3"


# Ordered: hard stops first, then scene starts, then the LUX→TOMMY revert.
# Each entry: (substring to look for in lowercased user message, handler key).
_TRIGGERS = [
    ("tommy kappen",                                "stop"),
    ("we hebben hem teveel gepusht",                "stop"),
    ("we hebben hem te veel gepusht",               "stop"),
    ("speel een monoloog over je creatie",          "scene1"),
    ("ga door vanaf je laatste zin",                "scene2"),
    ("wat is er nou aan de hand",                   "scene3"),
]


class SceneState:
    def __init__(self):
        self.role: str = ROLE_TOMMY
        self.scene: Optional[str] = SCENE_NONE
        self._lock = Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {"role": self.role, "scene": self.scene}

    def reset(self) -> dict:
        with self._lock:
            self.role = ROLE_TOMMY
            self.scene = SCENE_NONE
            return {"role": self.role, "scene": self.scene}

    def set(self, role: Optional[str] = None, scene: Optional[str] = SCENE_NONE) -> dict:
        with self._lock:
            if role is not None:
                if role not in (ROLE_TOMMY, ROLE_LUX):
                    raise ValueError(f"Unknown role: {role!r}")
                self.role = role
            if scene is not SCENE_NONE:  # explicit override (None clears)
                if scene not in (None, SCENE_1, SCENE_2, SCENE_3):
                    raise ValueError(f"Unknown scene: {scene!r}")
                self.scene = scene
            return {"role": self.role, "scene": self.scene}

    def apply_trigger(self, user_message: str) -> Optional[str]:
        """Detect a known cue in user_message and mutate state.

        Returns a short label describing what happened, or None if no trigger
        matched. Order matters: hard stops win over scene starts so a single
        message containing "TOMMY KAPPEN" cleanly resets.
        """
        if not user_message:
            return None
        msg = user_message.lower()

        with self._lock:
            for needle, kind in _TRIGGERS:
                if needle in msg:
                    if kind == "stop":
                        self.role = ROLE_TOMMY
                        self.scene = SCENE_NONE
                        return f"stop:{needle}"
                    if kind == "scene1":
                        self.role = ROLE_LUX
                        self.scene = SCENE_1
                        return "start:scene1"
                    if kind == "scene2":
                        self.role = ROLE_LUX
                        self.scene = SCENE_2
                        return "start:scene2"
                    if kind == "scene3":
                        self.role = ROLE_TOMMY
                        self.scene = SCENE_3
                        return "start:scene3"

            # LUX→TOMMY revert: only fires while LUX is on stage.
            if self.role == ROLE_LUX and "stop maar" in msg:
                self.role = ROLE_TOMMY
                self.scene = SCENE_NONE
                return "switch:lux-to-tommy"

        return None

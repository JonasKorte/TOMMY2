import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.app import app
from backend.config import FLASK_HOST, FLASK_PORT

if __name__ == "__main__":
    print(f"TOMMY2 starting on http://localhost:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)

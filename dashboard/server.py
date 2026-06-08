"""dashboard/server.py — Web dashboard server for ai-trader.

Usage:
    python3 dashboard/server.py          # starts on http://localhost:8080
    python3 dashboard/server.py --port 9000
"""

import json
import sys
import webbrowser
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn
except ImportError:
    print("pip install fastapi uvicorn")
    sys.exit(1)

STATE_FILE = Path(__file__).parent.parent / "dashboard_state.json"
HTML_FILE  = Path(__file__).parent / "index.html"

app = FastAPI(title="AI-Trader Dashboard")


@app.get("/")
def root():
    return FileResponse(HTML_FILE)


@app.get("/api/state")
def get_state():
    try:
        if STATE_FILE.exists():
            return JSONResponse(json.loads(STATE_FILE.read_text()))
    except Exception:
        pass
    return JSONResponse({})


if __name__ == "__main__":
    port = 8080
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])

    print(f"AI-Trader dashboard → http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

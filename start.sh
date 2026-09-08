#!/bin/bash
# Starts FastAPI in the background on localhost only, waits until it actually
# answers, then runs Streamlit in the foreground as the process the platform
# keeps alive. Streamlit's port reads $PORT with a local fallback, learned
# directly from a real deploy failure on a sibling project: a hardcoded port
# only works on the one platform (Hugging Face Spaces) that always assigns
# that exact number, and breaks silently everywhere else that assigns its own.
set -e

export API_BASE_URL="http://127.0.0.1:8000"

uvicorn api.main:app --host 127.0.0.1 --port 8000 &

python3 -c "
import time, urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit('FastAPI did not become healthy in time')
"

exec streamlit run ui/app.py --server.address=0.0.0.0 --server.port="${PORT:-8501}"

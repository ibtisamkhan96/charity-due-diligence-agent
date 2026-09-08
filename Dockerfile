# Combined image for single-service platforms (Railway, Hugging Face Spaces):
# one container running both the FastAPI backend and the Streamlit UI together,
# since these platforms expose exactly one public port per service. FastAPI
# listens on 127.0.0.1 only (internal, reached by Streamlit inside the same
# container); Streamlit is the one process actually exposed on the public port.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ui.txt

COPY agent/ agent/
COPY api/ api/
COPY observability/ observability/
COPY ui/ ui/
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8501
CMD ["./start.sh"]

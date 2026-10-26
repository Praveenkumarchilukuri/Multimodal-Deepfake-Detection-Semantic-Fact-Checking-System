FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (OpenCV & audio support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY data/ data/
COPY src/ src/
COPY scripts/ scripts/

ENV PYTHONPATH=/app
EXPOSE 8000 8501

# Default runs FastAPI server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

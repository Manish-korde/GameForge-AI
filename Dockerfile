# =========================================================
# Single-Stage: Unified Python AI Backend & Static Server Runtime
# =========================================================
FROM python:3.10-slim AS runtime

# Set container environment variables
ENV PYTHONUNBUFFERED=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PYTHONPATH=/app/backend \
    PORT=8000

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch and Python packages (Layered to prevent 1024MB RAM OOM)
RUN pip install --no-cache-dir --upgrade pip

# Step 1: Install PyTorch CPU alone (cached independently across deploys)
RUN pip install --no-cache-dir --no-compile --prefer-binary torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install TensorFlow CPU + compatible NumPy alone (memory flushed after completion)
RUN pip install --no-cache-dir --no-compile --prefer-binary tensorflow-cpu==2.15.0 "numpy<2.0.0,>=1.23.5"

# Step 3: Install remaining lightweight packages
COPY backend/requirements_docker.txt ./backend/requirements_docker.txt
RUN pip install --no-cache-dir --no-compile --prefer-binary -r ./backend/requirements_docker.txt

# Copy backend source code, ML model checkpoints, and runtime directories
COPY backend/ ./backend/
COPY models/ ./models/
COPY evaluation/ ./evaluation/
COPY config/ ./config/
COPY documentation/ ./documentation/
COPY demo_offline_gallery/ ./demo_offline_gallery/

# Copy pre-compiled production React frontend bundle directly
COPY gui/dist/ ./gui/dist/

# Expose FastAPI server port
EXPOSE 8000

# Health check to ensure model server is active
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/status || exit 1

# Launch Uvicorn server with dynamic PORT environment variable support
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

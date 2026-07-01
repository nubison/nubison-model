FROM python:3.12.8-slim-bookworm

# System deps for the common ML stack so a model's requirements.txt installs cleanly:
# build tools (C/C++ ext without wheels), git (git+ requirements), ffmpeg/libsndfile1
# (audio/video), libsm6/libxext6 (opencv/X runtime libs). Kept in parity with Dockerfile.gpu.
RUN apt-get update && apt-get install -y \
    bash \
    git \
    build-essential \
    pkg-config \
    ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

SHELL ["bash", "-c"]
WORKDIR /app

LABEL org.opencontainers.image.title="nubison-model" \
      org.opencontainers.image.description="A container image for nubison-model."\
      org.opencontainers.image.source="https://github.com/nubison/nubison-model"

COPY start_server.sh /app/start_server.sh
RUN chmod +x /app/start_server.sh

ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_MODEL_URI=""
ENV MLFLOW_MODEL_NAME=""
ENV MLFLOW_MODEL_VERSION=""
ENV DEBUG=""
ENV PORT=3000
ENV NUM_WORKERS=2

EXPOSE ${PORT}

RUN pip install --no-cache-dir uv==0.5.10 "mlflow-skinny>=3.12.0,<3.14.0" pyyaml && \
    uv python install 3.9 3.10 3.11 3.12

ENTRYPOINT [ "/app/start_server.sh" ]

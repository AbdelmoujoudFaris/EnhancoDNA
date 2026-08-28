# EnhancoAI -- CPU-only default image.
# For CUDA deployment, install a CUDA-enabled torch wheel on top of this
# image (see docs/installation.md) rather than modifying this Dockerfile,
# so the default build stays portable.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="EnhancoAI"
LABEL org.opencontainers.image.description="Deep learning, MD and AI analysis of TF cooperativity and DNA allostery"
LABEL org.opencontainers.image.licenses="MIT"

# Minimal system libraries needed by PySide6 (offscreen) and matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libxkbcommon0 \
    libxcb-cursor0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e .

ENV QT_QPA_PLATFORM=offscreen \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["enhancoai"]
CMD ["--help"]

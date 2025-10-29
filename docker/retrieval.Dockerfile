FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/model_cache \
    TRANSFORMERS_CACHE=/model_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl ca-certificates \
    gdal-bin libgdal-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    CPATH=/usr/include/gdal

WORKDIR /app
COPY docker_mount/project/code/retrieval/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && pip install -r /app/requirements.txt

# Non-root user that matches typical UID 1000; safe in rootless Docker
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app
WORKDIR /app/retrieval
ENTRYPOINT ["bash","-lc"]

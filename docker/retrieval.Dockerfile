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

WORKDIR /workspace/app/retrieval
COPY app/retrieval/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Non-root user that matches typical UID 1000; safe in rootless Docker
RUN useradd -m -u 1000 app && mkdir -p /workspace && chown -R app:app /workspace
USER app
WORKDIR /workspace/app/retrieval
CMD ["bash","-lc","CUDA_VISIBLE_DEVICES=${SEARCH_GPU:-0} python3 -m uvicorn search_api:app --host 0.0.0.0 --port 8099"]

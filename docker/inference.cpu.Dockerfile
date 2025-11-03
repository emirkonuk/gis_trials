FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/models/.hf \
    TRANSFORMERS_CACHE=/workspace/models/.hf

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/app/inference
COPY app/inference/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install "torch==2.4.0" "torchvision==0.19.0" --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r /tmp/requirements.txt

COPY app/inference/ /workspace/app/inference/

EXPOSE 8081
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]

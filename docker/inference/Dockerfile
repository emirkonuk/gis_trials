# map_serving/docker/inference/Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/project/models/.hf

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# PyTorch CUDA 12.4 build
RUN pip3 install --upgrade pip && \
    pip3 install "torch==2.4.0+cu124" "torchvision==0.19.0+cu124" \
        --extra-index-url https://download.pytorch.org/whl/cu124

# VLM stack
RUN pip3 install "transformers>=4.45" accelerate pillow fastapi uvicorn

WORKDIR /app
COPY app.py /app/app.py

RUN ln -s /usr/bin/python3 /usr/bin/python

EXPOSE 8081
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]


# LiveAvatar — Lynk & Co Jordan Avatar Studio
# RunPod image for single 80GB GPU (A100/H100) clip-from-audio generation.
#
# Build (on a machine with Docker + buildx):
#   docker build -t YOUR_REGISTRY/liveavatar-runpod:latest -f deploy/runpod/Dockerfile .
#   docker push YOUR_REGISTRY/liveavatar-runpod:latest
#
# Model weights are NOT baked in (they are ~30GB+). They download once to a
# RunPod persistent network volume on first boot — see deploy/runpod/start.sh.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git ffmpeg build-essential ninja-build \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch (CUDA 12.8 build) — must be installed before flash-attn builds against it.
RUN pip install --upgrade pip && \
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# FlashAttention 2 (works on Ampere A100 and Hopper H100/H800).
# If you deploy exclusively on Hopper, FlashAttention 3 is faster — see README.
RUN pip install flash-attn==2.8.3 --no-build-isolation

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# huggingface CLI used by start.sh to fetch weights.
RUN pip install "huggingface_hub[cli]"

COPY . /app

RUN chmod +x /app/deploy/runpod/start.sh

EXPOSE 7860

CMD ["/app/deploy/runpod/start.sh"]

#!/bin/bash
# LiveAvatar — Lynk & Co Jordan Avatar Studio
# RunPod entrypoint: ensure weights exist on the persistent volume, then launch
# the branded single-GPU Gradio UI.
#
# Expected RunPod setup:
#   - Pod with ONE 80GB GPU (A100 80GB or H100 80GB).
#   - A persistent Network Volume mounted at /workspace (so weights survive restarts).
#   - Expose HTTP port 7860.
set -euo pipefail

# --- Configurable via RunPod environment variables ------------------------
# Where weights live. Point this at your persistent volume.
CKPT_ROOT="${CKPT_ROOT:-/workspace/ckpt}"
# Use the default HuggingFace endpoint. Set HF_ENDPOINT=https://hf-mirror.com only if in mainland China.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
# Cache HF downloads on the volume too, so they survive restarts.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
# Pre-compile on boot? Keep true for the event, but PRE-WARM before doors open
# (the first generation after boot is slow while it compiles).
export ENABLE_COMPILE="${ENABLE_COMPILE:-true}"
SERVER_PORT="${SERVER_PORT:-7860}"
SIZE="${SIZE:-704*384}"
# FP8 saves VRAM (needed on 48GB GPUs) but slightly lowers quality. On an 80GB
# GPU you don't need it — leave off for best quality. Set ENABLE_FP8=true to enable.
ENABLE_FP8="${ENABLE_FP8:-false}"
# --------------------------------------------------------------------------

BASE_DIR="$CKPT_ROOT/Wan2.2-S2V-14B"
LORA_DIR="$CKPT_ROOT/LiveAvatar"

mkdir -p "$CKPT_ROOT" "$HF_HOME"

echo "=========================================="
echo " Lynk & Co Jordan — Avatar Studio (RunPod)"
echo " CKPT_ROOT=$CKPT_ROOT"
echo " ENABLE_COMPILE=$ENABLE_COMPILE"
echo "=========================================="

# --- Download weights once (idempotent) -----------------------------------
if [ ! -f "$BASE_DIR/config.json" ]; then
    echo "[setup] Downloading base model Wan2.2-S2V-14B (~30GB, one time)..."
    huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir "$BASE_DIR"
else
    echo "[setup] Base model already present, skipping download."
fi

if [ ! -d "$LORA_DIR" ] || [ -z "$(ls -A "$LORA_DIR" 2>/dev/null)" ]; then
    echo "[setup] Downloading LiveAvatar LoRA..."
    huggingface-cli download Quark-Vision/Live-Avatar --local-dir "$LORA_DIR"
else
    echo "[setup] LoRA already present, skipping download."
fi

# --- Launch single-GPU Gradio UI ------------------------------------------
# RunPod exposes the GPU as device 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF

FP8_FLAG=""
if [ "$ENABLE_FP8" = "true" ]; then
    FP8_FLAG="--fp8"
    echo "[run] FP8 quantization ENABLED (lower VRAM, slightly lower quality)."
fi

echo "[run] Launching Gradio on 0.0.0.0:$SERVER_PORT ..."
exec torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    minimal_inference/gradio_app.py \
    --ulysses_size 1 \
    --task s2v-14B \
    --size "$SIZE" \
    --base_seed 420 \
    --training_config liveavatar/configs/s2v_causal_sft.yaml \
    --offload_model True \
    --convert_model_dtype \
    --infer_frames 48 \
    --load_lora \
    --lora_path_dmd "$LORA_DIR" \
    --sample_steps 4 \
    --sample_guide_scale 0 \
    --num_clip 100 \
    --num_gpus_dit 1 \
    --sample_solver euler \
    --single_gpu \
    --ckpt_dir "$BASE_DIR" \
    --server_port "$SERVER_PORT" \
    --server_name "0.0.0.0" \
    $FP8_FLAG

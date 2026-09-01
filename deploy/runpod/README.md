# Deploying Avatar Studio on RunPod (single-GPU booth setup)

This deploys the **Lynk & Co Jordan Avatar Studio** (branded LiveAvatar Gradio UI)
on a single 80GB GPU for **clip-from-audio** generation: a guest provides audio +
picks a reference image, and a short avatar video is generated in seconds.

> This is the **robust** booth configuration. It is far more reliable and ~10x
> cheaper than the experimental 5-GPU real-time streaming mode.

## What you need
- A RunPod account.
- A **single 80GB GPU** pod: A100 80GB or H100 80GB.
- A **persistent Network Volume** (≈100GB) mounted at `/workspace` so the ~30GB
  of model weights survive pod restarts (download only happens once).
- HTTP port **7860** exposed.

## Steps

### 1. Build and push the image
On a machine with Docker (the weights are NOT in the image — they download on
first boot):
```bash
docker build -t YOUR_REGISTRY/liveavatar-runpod:latest -f deploy/runpod/Dockerfile .
docker push YOUR_REGISTRY/liveavatar-runpod:latest
```
> The image build installs flash-attn from source and can take 20–40 minutes.

### 2. Create the RunPod pod
- **GPU:** 1× A100 80GB or H100 80GB.
- **Image:** `YOUR_REGISTRY/liveavatar-runpod:latest`.
- **Network Volume:** attach one, mount path `/workspace`.
- **Expose HTTP port:** `7860`.
- **Environment variables** (all optional, sensible defaults shown):
  | Var | Default | Notes |
  |-----|---------|-------|
  | `CKPT_ROOT` | `/workspace/ckpt` | Weights location (on the volume). |
  | `HF_HOME` | `/workspace/hf_cache` | HF cache (on the volume). |
  | `ENABLE_COMPILE` | `true` | Faster after warm-up; first run is slow. |
  | `ENABLE_FP8` | `false` | Leave off on 80GB for best quality. |
  | `SIZE` | `704*384` | Output area; aspect follows the input image. |
  | `SERVER_PORT` | `7860` | Gradio port. |

### 3. First boot
On first start, `start.sh` downloads the base model and LoRA to the volume
(~30GB, one time), then launches Gradio. Subsequent restarts skip the download.

Open the pod's port-7860 HTTP endpoint — you'll see the branded Avatar Studio UI.

## Live event checklist (important)
- [ ] **PRE-WARM before doors open.** With `ENABLE_COMPILE=true` the *first*
      generation after boot is slow (minutes) while the model compiles. Run one
      throwaway generation yourself first, never in front of a guest.
- [ ] **Keep the pod running** for the whole event — cold starts re-download
      nothing (thanks to the volume) but still reload the model into VRAM.
- [ ] **Have a fallback.** This is a research-grade model; if it errors, the
      Gradio status box shows the message. Consider a few pre-rendered clips as
      a backup demo.
- [ ] **Test your exact avatars + audio** ahead of time (see below).

## Putting your branded frontend in front (optional)
If you want a Cloudflare-hosted front door, host a static page on Cloudflare
Pages that embeds/links the RunPod port-7860 URL. The **model cannot run on
Cloudflare** — RunPod runs the model, Cloudflare just serves the UI shell.

## Things to verify on first run
- **LoRA path:** `start.sh` points `--lora_path_dmd` at the local downloaded
  directory (`$CKPT_ROOT/LiveAvatar`). The shipped scripts instead pass the
  HuggingFace id `Quark-Vision/Live-Avatar`. If loading fails, try setting it to
  that id, or to the specific `.safetensors` file inside the directory.
- **Custom example avatars:** drop your Lynk & Co images/audio into `examples/`
  and update the `EXAMPLE_IMAGES` / `EXAMPLE_AUDIOS` / `EXAMPLE_PROMPT` lists at
  the top of `minimal_inference/gradio_app.py`.

## Not yet tested
These files have not been run on a live GPU from this workstation (no GPU/Docker
here). Build the image and do a full dry run on a RunPod pod before the event.

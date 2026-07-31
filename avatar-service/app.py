"""Avatar service for the Lynk & Co Jordan experience.

Clean API contract that the rest of the system depends on:

    POST /generate   (multipart/form-data)
        audio: the spoken audio to drive the avatar (any common format)
        image: optional reference-image identifier/name
      -> returns video/mp4

Two modes (env MODE):
    mock    (default) -> returns a placeholder clip instantly. Lets you build and
                         test the ENTIRE local experience on any machine (no GPU).
    runpod            -> forwards to the real LiveAvatar inference endpoint on a
                         GPU host. Stubbed until Phase 2 — this is the ONLY thing
                         that changes when you go live.

Swapping mock -> real is a single env flag; nothing else in the stack changes.
"""
import os
import subprocess
import tempfile

import httpx
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

MODE = os.environ.get("MODE", "mock")
RUNPOD_URL = os.environ.get("RUNPOD_URL", "")  # used only in runpod mode

app = FastAPI(title="Lynk & Co Avatar Service")

_PLACEHOLDER = os.path.join(tempfile.gettempdir(), "lynk_avatar_placeholder.mp4")


def _ensure_placeholder() -> str:
    """Render a small branded placeholder clip once, with ffmpeg."""
    if os.path.exists(_PLACEHOLDER):
        return _PLACEHOLDER
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    drawtext = (
        f"drawtext=fontfile={font}:text='LYNK & CO — Avatar (mock)':"
        "fontcolor=0x00E676:fontsize=30:x=(w-text_w)/2:y=(h-text_h)/2"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x0A0A0A:s=512x512:d=3:r=25",
        "-vf", drawtext,
        "-pix_fmt", "yuv420p",
        _PLACEHOLDER,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return _PLACEHOLDER


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE}


@app.post("/generate")
async def generate(audio: UploadFile = File(...), image: str = Form("default")):
    if MODE == "mock":
        # Audio/image are accepted (to exercise the real upload path) but ignored.
        path = _ensure_placeholder()
        return FileResponse(path, media_type="video/mp4", filename="avatar.mp4")

    if MODE == "runpod":
        if not RUNPOD_URL:
            return JSONResponse(
                {"error": "RUNPOD_URL not set"}, status_code=500
            )
        # PHASE 2: forward audio + reference image to the real LiveAvatar endpoint
        # running on the GPU host and stream back the generated MP4.
        # This is the single integration point to finish when the GPU is ready.
        audio_bytes = await audio.read()
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(
                f"{RUNPOD_URL.rstrip('/')}/generate",
                files={"audio": (audio.filename, audio_bytes, audio.content_type)},
                data={"image": image},
            )
        return JSONResponse(
            {"error": "runpod mode is stubbed", "upstream_status": r.status_code},
            status_code=501,
        )

    return JSONResponse({"error": f"unknown MODE={MODE}"}, status_code=500)

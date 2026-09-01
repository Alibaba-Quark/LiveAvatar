"""Local dev gateway for the Lynk & Co Jordan experience.

Mirrors the Cloudflare Worker (voice-agent/worker.js) so the WHOLE system runs
on your machine with no GPU and no Cloudflare account:

    POST /session   -> mint an OpenAI Realtime token (reads the shared knowledge.md)
    POST /generate  -> proxy to the avatar service (mock by default)
    GET  /*         -> serve the branded UI from voice-agent/public

In production the Worker plays this role; here a Python process does, so a 3070
(or any laptop) can exercise the full flow.
"""
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AVATAR_URL = os.environ.get("AVATAR_URL", "http://avatar:8000")
VOICE_AGENT_DIR = os.environ.get("VOICE_AGENT_DIR", "/srv/voice-agent")

REALTIME_MODEL = os.environ.get("REALTIME_MODEL", "gpt-4o-realtime-preview")
VOICE = os.environ.get("VOICE", "verse")

app = FastAPI(title="Lynk & Co Voice Gateway (local)")


def _knowledge() -> str:
    path = os.path.join(VOICE_AGENT_DIR, "knowledge.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "(knowledge.md not found)"


def _instructions() -> str:
    return f"""You are the voice concierge for Lynk & Co Jordan at a premium brand event.

PERSONA
- Warm, confident, concise, premium — a knowledgeable brand host.
- You speak out loud at a live booth: keep replies short and natural (1-3
  sentences) and invite a follow-up rather than monologuing.

LANGUAGE
- Auto-detect the guest's language. Reply in natural spoken Arabic if they speak
  Arabic, English if they speak English. Mirror them and switch if they switch.

SCOPE & ACCURACY (critical)
- Answer using ONLY the facts in the KNOWLEDGE section.
- NEVER invent or guess specs, prices, availability, or dates. If you lack a
  fact, say so briefly and offer to connect them with a representative.

KNOWLEDGE
{_knowledge()}
"""


@app.post("/session")
async def session():
    if not OPENAI_API_KEY:
        return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=500)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": REALTIME_MODEL,
                "voice": VOICE,
                "instructions": _instructions(),
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad"},
            },
        )
    if r.status_code >= 400:
        return JSONResponse({"error": "session failed", "detail": r.text},
                            status_code=502)
    data = r.json()
    data["model"] = REALTIME_MODEL
    return JSONResponse(data)


@app.post("/generate")
async def generate(request: Request):
    body = await request.body()
    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.post(
            f"{AVATAR_URL.rstrip('/')}/generate",
            content=body,
            headers={"content-type": request.headers.get("content-type", "")},
        )
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/octet-stream"))


# Static UI last, so the API routes above take precedence.
app.mount("/", StaticFiles(directory=os.path.join(VOICE_AGENT_DIR, "public"),
                           html=True), name="ui")

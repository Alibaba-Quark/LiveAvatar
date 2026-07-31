# Local-first stack — Lynk & Co Jordan experience

Run the **whole project on your own machine** (no GPU needed), then make hosting
the very last step. The voice agent is fully real; the avatar is mocked locally
and becomes the real GPU model with a single env flag at the end.

```
Browser  →  http://localhost:8080
  ├─ /session   → gateway → OpenAI Realtime token        [REAL]
  ├─ live voice ⇄ OpenAI Realtime (Arabic/English)       [REAL]
  └─ /generate  → gateway → avatar service (MOCK)        [swap → RunPod last]
```

## Requirements
- **Docker Desktop** (Windows/Mac/Linux). That's it — your RTX 3070 is not used;
  the avatar model does not run locally (it needs 48–80GB VRAM), so it's mocked.
- An **OpenAI API key** with Realtime access.

## Run
```bash
# from the repo root
export OPENAI_API_KEY=sk-...        # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
docker compose -f local-dev/docker-compose.yml up --build
```
Open **http://localhost:8080**, tap the orb, allow the mic, and talk (Arabic or
English). Tick **"Show avatar"** to see the mock avatar panel: each spoken reply
is captured and sent to the avatar service, which returns a placeholder clip —
proving the full voice→avatar pipeline end to end.

> Mic access needs `localhost` or HTTPS. `http://localhost:8080` is fine.

## What's real vs mock locally
| Part | Local | Notes |
|---|---|---|
| Branded UI | ✅ real | served by the gateway |
| Voice (STT+LLM+TTS) | ✅ real | OpenAI Realtime, bilingual |
| Audio → /generate handoff | ✅ real | exercises the real upload path |
| Avatar video | 🟡 mock | placeholder clip; real model lives on GPU |

## Going live (the last step)
1. Deploy the avatar model on RunPod (see `deploy/runpod/`) and expose an HTTP
   `/generate` endpoint (finish the Phase-2 stub in `avatar-service/app.py`).
2. Point the avatar service at it: set `AVATAR_MODE=runpod` and
   `RUNPOD_URL=https://...` (env in docker-compose).
3. For the public front door, deploy `voice-agent/` to Cloudflare
   (`npx wrangler deploy`) and set its `AVATAR_URL` var to the RunPod endpoint.

Nothing in the browser or the contract changes between mock and real.

## Edit before any event
`voice-agent/knowledge.md` — the single source of truth for what the agent may
say. Both the local gateway and the Cloudflare Worker read it. The agent will
not invent facts, so fill it with official Jordan-market data.

## Caveats / not yet tested here
- Built but **not run** from this workstation (no Docker/mic/OpenAI key here).
  Do a real `docker compose up` and a mic test before relying on it.
- `MediaRecorder` per-turn capture works in Chrome/Edge; verify in your booth
  browser.
- Mock audio is WebM/Opus. The real RunPod path will need to transcode to WAV
  for LiveAvatar — that belongs in the Phase-2 `runpod` branch of the avatar
  service.

# Lynk & Co Jordan — Voice Concierge

A bilingual (Arabic + English, auto-detected) realtime voice agent for the
Lynk & Co Jordan booth. Guests tap a circle and talk; the agent answers
questions about the vehicles. Built on the **OpenAI Realtime API** over WebRTC,
with a **Cloudflare Worker** minting short-lived session tokens so the API key
stays server-side.

This is **Phase 1**. The assistant's spoken audio is captured in the browser
(`assistantStream` in `app.js`) so a later **Phase 2** can feed it to LiveAvatar
and render a talking-head video — without re-architecting anything.

## Files
```
voice-agent/
├── public/
│   ├── index.html   # branded UI (black + Lynk & Co green)
│   ├── styles.css    # theme
│   └── app.js        # WebRTC client → OpenAI Realtime
├── worker.js         # /session token minting + persona + KNOWLEDGE base
├── wrangler.toml     # Cloudflare config
└── README.md
```

## What you MUST edit before the event
Open `worker.js` and replace the **KNOWLEDGE** placeholders with official
Jordan-market data (models, prices, features, showroom info). The agent is
instructed to **only** state facts found there and to **never invent**
specs/prices — so anything you leave blank, it will politely decline and offer a
representative. This is deliberate, to avoid wrong info at a brand event.

Optional tweaks in `worker.js`:
- `REALTIME_MODEL` — bump as OpenAI ships newer realtime models.
- `VOICE` — try voices for the best Arabic/English sound (see comment).

## Run locally
```bash
cd voice-agent
npm install -g wrangler          # if you don't have it
npx wrangler secret put OPENAI_API_KEY   # paste your key (stored encrypted)
npx wrangler dev                 # serves UI + /session at http://localhost:8787
```
Open the URL, tap the circle, allow the mic, and talk.

> Browsers require **HTTPS (or localhost)** for microphone access. `wrangler dev`
> on localhost is fine; any real deployment must be HTTPS (Cloudflare gives you
> that automatically).

## Deploy to Cloudflare
```bash
cd voice-agent
npx wrangler secret put OPENAI_API_KEY   # set the key on the deployed worker
npx wrangler deploy
```
You'll get a `*.workers.dev` URL (or attach your own domain). That single URL
serves both the branded page and the `/session` endpoint.

## How it works
```
Browser ──mic audio──►  OpenAI Realtime  (speech-to-text + LLM + text-to-speech)
   ▲   ◄──voice reply──        ▲
   └── POST /session ──► Cloudflare Worker ──► OpenAI (mints ephemeral token,
                                                 injects persona + knowledge)
```

## Costs
OpenAI Realtime API is billed per minute of audio in/out (no GPU, no servers to
run). The Cloudflare Worker is effectively free at booth volumes. Budget by
expected conversation minutes.

## Caveats / test before the event
- **Arabic voice quality:** OpenAI realtime voices are strong but English-leaning.
  Test the Arabic output at the booth; if it's not premium enough, Phase 1.5 can
  swap to an assembled stack with a dedicated Arabic TTS (e.g. ElevenLabs).
- **Noisy booth:** server VAD turn-detection may trip on crowd noise. If so,
  consider switching `app.js` to push-to-talk (hold the orb to speak).
- **Not load-tested here.** Do a full dry run on real hardware/network before
  the event.

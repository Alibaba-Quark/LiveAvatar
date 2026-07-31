// LYNK & CO JORDAN — Voice Concierge (Cloudflare Worker, PRODUCTION hosting)
//
// For LOCAL development use local-dev/ (docker-compose) instead — it mirrors this
// Worker with a Python gateway so the whole system runs on your machine.
//
// Jobs:
//   1. POST /session  -> mint a short-lived OpenAI Realtime token (API key stays
//                        server-side). Persona + knowledge.md injected.
//   2. POST /generate -> proxy to the avatar service (env.AVATAR_URL) once you
//                        host one. Until then this returns 503.
//   3. everything else -> serve the static branded UI from /public.
//
// Secrets:  npx wrangler secret put OPENAI_API_KEY
// Vars:     set AVATAR_URL (your RunPod avatar endpoint) when Phase 2 is live.

// knowledge.md is bundled as text (see the [[rules]] Text entry in wrangler.toml).
import KNOWLEDGE from "./knowledge.md";

const REALTIME_MODEL = "gpt-4o-realtime-preview"; // bump as newer realtime models ship
const VOICE = "verse"; // try voices for best Arabic + English sound

function buildInstructions() {
  return `You are the voice concierge for Lynk & Co Jordan at a premium brand event.

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
  fact, say so briefly and offer to connect them with a representative or point
  them to the showroom.
- Avoid comparative claims about competitors you cannot support from KNOWLEDGE.

KNOWLEDGE
${KNOWLEDGE}
`;
}

async function mintSession(env) {
  const resp = await fetch("https://api.openai.com/v1/realtime/sessions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: REALTIME_MODEL,
      voice: VOICE,
      instructions: buildInstructions(),
      input_audio_transcription: { model: "whisper-1" },
      turn_detection: { type: "server_vad" },
    }),
  });

  if (!resp.ok) {
    const detail = await resp.text();
    return new Response(`Failed to create session: ${resp.status} ${detail}`, { status: 502 });
  }

  const session = await resp.json();
  session.model = REALTIME_MODEL; // client needs this for the WebRTC URL
  return new Response(JSON.stringify(session), {
    headers: { "Content-Type": "application/json" },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/session" && request.method === "POST") {
      if (!env.OPENAI_API_KEY) {
        return new Response("OPENAI_API_KEY not configured", { status: 500 });
      }
      return mintSession(env);
    }

    if (url.pathname === "/generate" && request.method === "POST") {
      if (!env.AVATAR_URL) {
        return new Response("Avatar service not configured (set AVATAR_URL)", { status: 503 });
      }
      return fetch(`${env.AVATAR_URL.replace(/\/$/, "")}/generate`, {
        method: "POST",
        body: request.body,
        headers: request.headers,
      });
    }

    return env.ASSETS.fetch(request);
  },
};

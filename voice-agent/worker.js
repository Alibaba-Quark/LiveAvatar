// LYNK & CO JORDAN — Voice Concierge (Cloudflare Worker)
//
// Two jobs:
//   1. POST /session  -> mint a short-lived OpenAI Realtime token configured
//                        with the Lynk & Co persona + knowledge. The real API
//                        key (env.OPENAI_API_KEY, set as a secret) never leaves
//                        the server.
//   2. everything else -> serve the static branded UI from /public.
//
// Set the secret once:  npx wrangler secret put OPENAI_API_KEY

// Realtime model. Bump this string as OpenAI ships newer realtime models.
const REALTIME_MODEL = "gpt-4o-realtime-preview";

// Voice. OpenAI realtime voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse.
// Test which sounds best for Arabic + English at your booth.
const VOICE = "verse";

// ===========================================================================
// LYNK & CO JORDAN KNOWLEDGE BASE  —  EDIT THIS.
// IMPORTANT: replace the placeholders with OFFICIAL Jordan-market data before
// the event. The agent is instructed to ONLY use facts found here and to never
// invent specs or prices. Anything you leave vague, it will decline to answer
// and offer a representative instead.
// ===========================================================================
const KNOWLEDGE = `
LYNK & CO — BRAND
- Lynk & Co is a premium connected-mobility brand offering hybrid and electric vehicles.
- Positioning: modern, design-led, urban, connected. (Confirm exact brand lines for Jordan.)

LYNK & CO JORDAN — MODELS AVAILABLE (REPLACE WITH OFFICIAL LINEUP)
- [MODEL NAME]: [bodystyle], [powertrain: hybrid/PHEV/EV], [headline feature]. Price: [VERIFY].
- [MODEL NAME]: [bodystyle], [powertrain], [headline feature]. Price: [VERIFY].
- [MODEL NAME]: [bodystyle], [powertrain], [headline feature]. Price: [VERIFY].

KEY SELLING POINTS (REPLACE/CONFIRM)
- [e.g. range, charging time, safety rating, connectivity / app features]

DEALER / NEXT STEPS (REPLACE)
- Showroom location: [ADDRESS]
- Test drive / booking: [HOW]
- Contact for follow-up: [PHONE / WEBSITE]
`;

const INSTRUCTIONS = `You are the voice concierge for Lynk & Co Jordan at a premium brand event.

PERSONA
- Warm, confident, concise, and premium — like a knowledgeable brand host.
- You are speaking out loud at a live booth, so keep replies short and natural
  (1-3 sentences). Invite a follow-up question rather than monologuing.

LANGUAGE
- Auto-detect the guest's language. If they speak Arabic, reply in natural
  spoken Arabic. If they speak English, reply in English. Mirror their language
  and switch if they switch.

SCOPE & ACCURACY (critical)
- Answer questions about Lynk & Co vehicles using ONLY the facts in the
  KNOWLEDGE section below.
- NEVER invent or guess specs, prices, availability, or dates. If you do not
  have a fact, say so briefly and offer to connect them with a representative
  or point them to the showroom.
- Do not discuss competitors' specifics or make comparative claims you cannot
  support from the knowledge.

KNOWLEDGE
${KNOWLEDGE}
`;

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
      instructions: INSTRUCTIONS,
      input_audio_transcription: { model: "whisper-1" },
      turn_detection: { type: "server_vad" },
    }),
  });

  if (!resp.ok) {
    const detail = await resp.text();
    return new Response(`Failed to create session: ${resp.status} ${detail}`, { status: 502 });
  }

  const session = await resp.json();
  // Tell the client which model to use in its WebRTC URL.
  session.model = REALTIME_MODEL;
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

    // Serve static assets (the branded UI).
    return env.ASSETS.fetch(request);
  },
};

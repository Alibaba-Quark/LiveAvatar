// LYNK & CO JORDAN — Voice Concierge (browser client)
// Connects to the OpenAI Realtime API over WebRTC. The ephemeral session
// token is minted by our Cloudflare Worker at /session, so the real API key
// never reaches the browser.

const micBtn = document.getElementById("micBtn");
const orbLabel = document.getElementById("orbLabel");
const statusLine = document.getElementById("statusLine");
const transcript = document.getElementById("transcript");
const assistantAudio = document.getElementById("assistantAudio");

let pc = null;          // RTCPeerConnection
let dc = null;          // data channel for events
let micStream = null;
let connected = false;

// --- Phase 2 hook -----------------------------------------------------------
// The assistant's spoken audio arrives as a remote MediaStream track. Capturing
// it here is what will later drive the LiveAvatar talking-head video. We keep a
// reference so a future avatar layer can tap it without re-architecting.
let assistantStream = null;
// ---------------------------------------------------------------------------

function setState(state, label) {
  micBtn.classList.remove("connecting", "live", "speaking");
  if (state) micBtn.classList.add(state);
  if (label) orbLabel.textContent = label;
}

function addBubble(role, text) {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  el.textContent = text;
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
  return el;
}

async function start() {
  if (connected) {
    stop();
    return;
  }
  try {
    setState("connecting", "Connecting…");
    statusLine.textContent = "Starting the concierge…";

    // 1. Get an ephemeral session token from our Worker.
    const tokenResp = await fetch("/session", { method: "POST" });
    if (!tokenResp.ok) throw new Error(`session token failed: ${tokenResp.status}`);
    const session = await tokenResp.json();
    const EPHEMERAL_KEY = session.client_secret.value;
    const MODEL = session.model; // Worker tells us which realtime model it created.

    // 2. Set up WebRTC.
    pc = new RTCPeerConnection();

    pc.ontrack = (e) => {
      assistantStream = e.streams[0];
      assistantAudio.srcObject = assistantStream;
    };

    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    pc.addTrack(micStream.getTracks()[0], micStream);

    dc = pc.createDataChannel("oai-events");
    dc.addEventListener("message", onServerEvent);

    // 3. SDP offer/answer handshake with OpenAI.
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const sdpResp = await fetch(`https://api.openai.com/v1/realtime?model=${encodeURIComponent(MODEL)}`, {
      method: "POST",
      body: offer.sdp,
      headers: {
        Authorization: `Bearer ${EPHEMERAL_KEY}`,
        "Content-Type": "application/sdp",
      },
    });
    if (!sdpResp.ok) throw new Error(`realtime connect failed: ${sdpResp.status}`);

    await pc.setRemoteDescription({ type: "answer", sdp: await sdpResp.text() });

    connected = true;
    setState("live", "Listening…");
    statusLine.textContent = "Go ahead — ask in Arabic or English.";
  } catch (err) {
    console.error(err);
    statusLine.textContent = `Error: ${err.message}`;
    setState(null, "Tap to talk");
    cleanup();
  }
}

let currentAssistantBubble = null;

function onServerEvent(e) {
  let evt;
  try { evt = JSON.parse(e.data); } catch { return; }

  switch (evt.type) {
    // Guest speech (transcribed input).
    case "conversation.item.input_audio_transcription.completed":
      if (evt.transcript) addBubble("user", evt.transcript.trim());
      break;

    // Assistant text, streamed alongside the audio.
    case "response.audio_transcript.delta":
      if (!currentAssistantBubble) currentAssistantBubble = addBubble("assistant", "");
      currentAssistantBubble.textContent += evt.delta;
      transcript.scrollTop = transcript.scrollHeight;
      break;
    case "response.audio_transcript.done":
      currentAssistantBubble = null;
      break;

    // Speaking-state visuals.
    case "output_audio_buffer.started":
    case "response.audio.delta":
      setState("speaking", "Speaking…");
      break;
    case "response.audio.done":
    case "output_audio_buffer.stopped":
    case "response.done":
      if (connected) setState("live", "Listening…");
      break;

    case "error":
      console.error("Realtime error:", evt.error);
      statusLine.textContent = `Error: ${evt.error?.message || "unknown"}`;
      break;
  }
}

function cleanup() {
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  if (dc) { try { dc.close(); } catch {} dc = null; }
  if (pc) { try { pc.close(); } catch {} pc = null; }
  assistantStream = null;
  connected = false;
}

function stop() {
  cleanup();
  setState(null, "Tap to talk");
  statusLine.textContent = "Conversation ended. Tap to start again.";
}

micBtn.addEventListener("click", start);

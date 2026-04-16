import { PipecatClient } from "@pipecat-ai/client-js";
import { WebSocketTransport } from "@pipecat-ai/websocket-transport";

const statusEl   = document.getElementById("status");
const userTextEl = document.getElementById("userText");
const botTextEl  = document.getElementById("botText");
const startBtn   = document.getElementById("startBtn");
const stopBtn    = document.getElementById("stopBtn");

function setStatus(text) {
  statusEl.textContent = text;
}

const client = new PipecatClient({
  transport: new WebSocketTransport(),
  enableMic: true,
  enableCam: false,
  callbacks: {
    onConnected: () => {
      setStatus("Connected — speak now");
      startBtn.disabled = true;
      stopBtn.disabled  = false;
    },

    onDisconnected: () => {
      setStatus("Disconnected");
      startBtn.disabled = false;
      stopBtn.disabled  = true;
      userTextEl.textContent = "";
      botTextEl.textContent  = "";
    },

    onBotReady: () => {
      setStatus("Bot ready — speak now");
    },

    onUserTranscript: (data) => {
      if (data?.final) {
        userTextEl.textContent = data?.text || "";
      }
    },

    onBotTranscript: (data) => {
      botTextEl.textContent = data?.text || "";
    },

    onTransportStateChanged: (state) => {
      setStatus(`State: ${state}`);
    },

    onError: (err) => {
      console.error(err);
      setStatus(`Error: ${err?.message || String(err)}`);
    },
  },
});

startBtn.addEventListener("click", async () => {
  setStatus("Connecting...");
  try {
    await client.connect({ wsUrl: "ws://localhost:8765" });
  } catch (err) {
    console.error(err);
    setStatus(`Failed: ${err?.message || String(err)}`);
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    await client.disconnect();
  } catch (err) {
    console.error(err);
  }
});

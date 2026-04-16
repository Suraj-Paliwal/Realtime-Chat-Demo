# Jmoshi — Realtime Voice Companion

A realtime voice chatbot built with **Pipecat 1.0** that listens, thinks, and speaks. Uses Deepgram for speech recognition, OpenAI GPT-4o-mini as the brain, and Kokoro for warm CPU-based voice synthesis. No GPU required.

## How it works

```
Browser mic
    │  (protobuf over WebSocket)
    ▼
Pipecat Pipeline (bot.py)
    ├─ Silero VAD          — detects when you start/stop speaking
    ├─ Deepgram Nova-3     — transcribes speech to text (~150ms)
    ├─ OpenAI GPT-4o-mini  — generates a response
    ├─ Kokoro TTS          — converts text to audio (CPU, free)
    └─ WebSocket output    — streams audio back to browser

REST API (server.py)
    ├─ GET  /api/sessions/{user_id}        — session history
    ├─ GET  /api/memory/{user_id}          — memory graph facts
    └─ POST /api/memory/{user_id}/fact     — add a memory fact
```

## Project structure

```
jmoshi/
├── bot.py           — Pipecat voice pipeline (WebSocket server on :8765)
├── brain.py         — system prompt, emotion detection, session summariser
├── graph.py         — SQLite memory graph (stores facts about the user)
├── server.py        — FastAPI caregiver dashboard API (:8000)
├── index.html       — simple standalone HTML client (no build needed)
├── web/             — full Vite + Pipecat JS SDK client (recommended)
│   ├── index.html
│   ├── main.js
│   └── package.json
├── .env.example     — copy this to .env and fill in your keys
└── requirements.txt
```

## Quick start

### 1. Clone and install Python deps

```bash
git clone https://github.com/YOUR_USERNAME/jmoshi.git
cd jmoshi
pip install -r requirements.txt
```

### 2. Get API keys

| Service | Sign up | Free tier |
|---------|---------|-----------|
| Deepgram | https://console.deepgram.com | $200 free credit |
| OpenAI | https://platform.openai.com | pay-as-you-go |

### 3. Configure

```bash
cp .env.example .env
# Open .env and paste your DEEPGRAM_API_KEY and OPENAI_API_KEY
```

### 4. Install frontend deps

```bash
cd web
npm install
cd ..
```

### 5. Start the voice bot (Terminal 1)

```bash
python bot.py
# WebSocket server running on ws://localhost:8765
```

### 6. Start the frontend (Terminal 2)

```bash
cd web
npm run dev
# App running at http://localhost:5173
```

### 7. Open your browser

Go to **http://localhost:5173**, click **Start**, allow microphone access, and speak.

---

## Environment variables

Copy `.env.example` to `.env` and fill in:

```env
DEEPGRAM_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Optional
KOKORO_VOICE=af_heart     # TTS voice (see options below)
HOST=localhost
PORT=8765
```

---

## Kokoro TTS voices

| Voice | Description |
|-------|-------------|
| `af_heart` | Warm female English (default) |
| `af_sky` | Lighter female English |
| `am_echo` | Male English |
| `am_michael` | Deep male English |
| `bf_emma` | British female |
| `bm_george` | British male |

Change by setting `KOKORO_VOICE` in your `.env`.

---

## Swap the LLM

Edit `brain.py` to change the model. To use a different provider:

```python
# Current (OpenAI GPT-4o-mini)
from pipecat.services.openai.llm import OpenAILLMService
llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Claude Sonnet
from pipecat.services.anthropic.llm import AnthropicLLMService
llm = AnthropicLLMService(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-sonnet-4-5")

# GPT-4o
llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
```

---

## Caregiver API

Start the API server (optional, separate terminal):

```bash
uvicorn server:app --port 8000 --reload
```

Example calls:

```bash
# View session history
curl http://localhost:8000/api/sessions/demo_user

# View memory graph
curl http://localhost:8000/api/memory/demo_user

# Add a fact manually
curl -X POST http://localhost:8000/api/memory/demo_user/fact \
  -H "Content-Type: application/json" \
  -d '{"entity": "Alice", "type": "person", "detail": "wife, married 1975"}'
```

---

## Cost estimate (100 daily users, 20-min sessions)

| Component | Provider | Monthly cost |
|-----------|----------|-------------|
| STT | Deepgram Nova-3 | ~$354 |
| LLM | GPT-4o-mini | ~$20 |
| TTS | Kokoro (CPU) | $0 |
| Server | 2 vCPU / 4GB VPS | ~$20 |
| **Total** | | **~$394/mo** |

---

## Tech stack

- [Pipecat](https://github.com/pipecat-ai/pipecat) — voice pipeline framework
- [Deepgram](https://deepgram.com) — streaming speech-to-text
- [OpenAI](https://openai.com) — language model
- [Kokoro](https://github.com/hexgrad/kokoro) — local TTS (CPU)
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [FastAPI](https://fastapi.tiangolo.com) — REST API
- [Vite](https://vitejs.dev) — frontend dev server

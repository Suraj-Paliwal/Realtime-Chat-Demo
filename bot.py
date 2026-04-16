"""
Jmoshi Pipecat bot — wires together:

  Deepgram Nova-3 (streaming ASR, ~150ms first word)
      ↓
  Silero VAD (voice activity detection, handles turn-taking)
      ↓
  OpenAI GPT-4o-mini (LLM brain with Jmoshi system prompt + memory context)
      ↓
  Kokoro TTS (warm voice synthesis, runs on CPU)

Architecture overview:

  [Browser mic] ──WebSocket──▶ [Pipecat Pipeline]
                                    ├─ transport.input()           ← raw audio frames
                                    ├─ SileroVAD                   ← detect speech vs silence
                                    ├─ DeepgramSTT                 ← speech → text
                                    ├─ context_aggregator.user()   ← accumulate user turn
                                    ├─ OpenAILLM                   ← text → text
                                    ├─ KokoroTTS                   ← text → audio
                                    ├─ transport.output()          ← audio → browser
                                    └─ context_aggregator.assistant()
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMContextFrame, EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    UserTurnStoppedMessage,
    AssistantTurnStoppedMessage,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram.stt import LiveOptions

from brain import (
    get_system_prompt,
    detect_emotion,
    get_distress_bridge,
    summarise_session,
    OPENAI_MODEL,
)
from graph import MemoryGraph

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
KOKORO_VOICE     = os.getenv("KOKORO_VOICE", "af_heart")
HOST             = os.getenv("HOST", "localhost")
PORT             = int(os.getenv("PORT", "8765"))


# ── Server entry point ─────────────────────────────────────────────────────────

async def run_server():
    """
    Start the Jmoshi WebSocket voice server.
    Client connects to: ws://HOST:PORT
    """
    if not DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY not set in .env")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    memory = MemoryGraph()
    memory.init()
    logger.info(f"[Server] Memory graph initialised at {memory.db_path}")
    logger.info(f"[Server] Jmoshi voice server starting on ws://{HOST}:{PORT}")
    logger.info(f"[Server] STT: Deepgram Nova-3 | LLM: {OPENAI_MODEL} | TTS: Kokoro {KOKORO_VOICE}")

    # Per-session state (resets on each client connection)
    session_state: dict = {}

    # ── Transport ─────────────────────────────────────────────────────────────
    params = WebsocketServerParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        add_wav_header=False,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=24000,  # Kokoro outputs 24kHz
        serializer=ProtobufFrameSerializer(),
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=1.2,   # wait 1.2s of silence before end-of-turn
                confidence=0.6,
            )
        ),
    )
    transport = WebsocketServerTransport(
        params=params,
        host=HOST,
        port=PORT,
        output_name="jmoshi-audio-out",
    )

    # ── STT — Deepgram Nova-3 ─────────────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        live_options=LiveOptions(
            model="nova-3-general",
            language="en-US",
            punctuate=True,
            interim_results=False,
            endpointing=True,
            vad_events=True,
        ),
    )

    # ── LLM context ───────────────────────────────────────────────────────────
    user_id = "demo_user"
    memory_ctx = memory.get_context(user_id, "")
    system_prompt = get_system_prompt(memory_context=memory_ctx if memory_ctx else None)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "assistant",
            "content": (
                "Hello, it's so lovely to speak with you today. "
                "I'd love to hear about your life. "
                "Where did you grow up?"
            ),
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        max_tokens=100,
    )

    # ── TTS — Kokoro ──────────────────────────────────────────────────────────
    tts = KokoroTTSService(voice_id=KOKORO_VOICE)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    # ── Event handlers ────────────────────────────────────────────────────────

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, websocket):
        logger.info(f"[Bot] Client connected")
        session_state.clear()
        session_state.update({
            "session_id": str(uuid.uuid4()),
            "started_at": datetime.utcnow().isoformat(),
            "history": [],
        })
        # Trigger opening greeting
        await task.queue_frames([LLMContextFrame(context)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, websocket):
        logger.info(f"[Bot] Client disconnected")
        await task.queue_frames([EndFrame()])
        asyncio.create_task(_end_session(memory, user_id, session_state))

    @context_aggregator.user().event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        text = message.content if hasattr(message, "content") else str(message)
        if not text.strip():
            return

        history = session_state.get("history", [])
        history.append({"role": "user", "content": text})

        # Run emotion detection in background — does NOT block pipeline
        asyncio.create_task(_handle_emotion(task, text))

        # Update memory context with new user text for next turn
        new_ctx = memory.get_context(user_id, text)
        if new_ctx:
            new_system = get_system_prompt(memory_context=new_ctx)
            messages[0]["content"] = new_system

    @context_aggregator.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        content = message.content if hasattr(message, "content") else str(message)
        if content:
            history = session_state.get("history", [])
            history.append({"role": "assistant", "content": content})

    # ── Run ───────────────────────────────────────────────────────────────────
    runner = PipelineRunner()
    await runner.run(task)


async def _handle_emotion(task: PipelineTask, text: str):
    """Detect distress; inject a gentle redirect phrase if triggered."""
    result = await detect_emotion(text)
    logger.info(
        f"[Emotion] {result['emotion']} "
        f"({result['confidence']:.0%}) — distress={result['is_distress']}"
    )
    if result["is_distress"] and result["confidence"] > 0.75:
        logger.warning("[Emotion] Distress detected — injecting redirect phrase")
        bridge = get_distress_bridge()
        await task.queue_frames([TextFrame(text=bridge)])


async def _end_session(memory: MemoryGraph, user_id: str, session_state: dict):
    """Summarise session and persist facts to memory graph."""
    history = session_state.get("history", [])
    if len(history) < 2:
        return
    session_id = session_state.get("session_id", str(uuid.uuid4()))
    started_at = session_state.get("started_at", datetime.utcnow().isoformat())

    logger.info(f"[Bot] Ending session {session_id} — summarising...")
    try:
        summary_data = await summarise_session(history)

        if summary_data.get("new_facts"):
            memory.save_facts(user_id, session_id, summary_data["new_facts"])
            logger.info(f"[Bot] Saved {len(summary_data['new_facts'])} memory facts")

        memory.save_session(
            session_id=session_id,
            user_id=user_id,
            started_at=started_at,
            ended_at=datetime.utcnow().isoformat(),
            summary=summary_data.get("summary", ""),
            mood=summary_data.get("mood", "unknown"),
            suggested_topics=summary_data.get("suggested_topics", []),
        )
        logger.info(
            f"[Bot] Session saved. Mood={summary_data.get('mood')} "
            f"Topics={summary_data.get('suggested_topics')}"
        )
    except Exception as e:
        logger.error(f"[Bot] End session failed: {e}")


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            logger.info("[Server] Shutting down.")
            break
        except Exception as e:
            logger.error(f"[Server] Crashed: {e} — restarting in 2s...")
            import time; time.sleep(2)

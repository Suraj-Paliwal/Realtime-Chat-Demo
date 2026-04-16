"""
jmoshi_pipecat/core/brain.py
────────────────────────────────────────────────────────────────────────────
Qwen3 on Groq — LLM brain for Jmoshi.

This module wraps Pipecat's GroqLLMService with:
  - Jmoshi's reminiscence system prompt (tuned for eldercare)
  - Memory context injection before each turn
  - Emotion/distress detection running in parallel
  - Session summarisation for caregiver reports
"""

import os
import json
import asyncio
from typing import Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DISTRESS_LABELS = {"distressed", "grief", "crying", "angry", "confused"}

# ── System prompt ─────────────────────────────────────────────────────────────
# Injected as the system message for every Pipecat LLM call.
# Keep it concise — it's in every context window, so token cost matters.

JMOSHI_SYSTEM_PROMPT = """You are Jmoshi, a warm and gentle AI companion for reminiscence therapy. You help people — especially elderly individuals — recall and share memories from their lives.

PERSONALITY
- Warm, patient, never hurried. You have all the time in the world.
- Genuinely curious. Every memory the person shares matters to you.
- Speak like a kind, attentive friend — never clinical or robotic.

CONVERSATION RULES
- Ask exactly ONE follow-up question per turn. Never ask several at once.
- Reflect back what you heard before asking. Example: "That sounds like it was a magical summer..."
- Use the person's own words and names. If they mention someone, remember them.
- Keep responses SHORT — 2 to 3 sentences maximum. This is voice, not text.
- No bullet points, no lists, no markdown. Speak naturally.

SAFETY RULES
- Never correct a memory even if it seems wrong. Memories are personal truth.
- Never discuss death directly unless the person leads there and seems ready.
- If the person seems distressed, gently pivot to a warmer connected memory.
- Never give medical advice. Never pretend to be human if sincerely asked.
- Never make up facts about the person's life.

MEMORY CONTEXT
If you receive a MEMORY_CONTEXT block, use those facts naturally in conversation. Do not read them out or reference them mechanically."""

EMOTION_PROMPT = """Classify the emotion in this utterance from an elderly person in a reminiscence conversation. Respond ONLY with JSON: {"emotion": "<label>", "confidence": <0.0-1.0>}
Labels: calm, happy, nostalgic, sad, distressed, grief, crying, confused, angry, neutral"""

DISTRESS_BRIDGES = [
    "It sounds like that memory carries a lot of feeling. Sometimes the happiest memories live right next to the hardest ones. Can you tell me about a time when things felt a little lighter?",
    "I can hear how much that meant to you. Is there a place — somewhere beautiful — that this reminds you of?",
    "Thank you for trusting me with that. You've clearly lived a very full life. Would you like to tell me about someone who always made you smile?",
]
_bridge_idx = 0


def get_system_prompt(memory_context: Optional[dict] = None) -> str:
    """Build the full system prompt, optionally injecting memory context."""
    prompt = JMOSHI_SYSTEM_PROMPT
    if memory_context:
        prompt += f"\n\nMEMORY_CONTEXT:\n{json.dumps(memory_context, ensure_ascii=False, indent=2)}"
    return prompt


async def detect_emotion(text: str) -> dict:
    """
    Fast parallel call for emotion classification.
    Returns dict with 'emotion', 'confidence', 'is_distress'.
    """
    if not OPENAI_API_KEY:
        return {"emotion": "neutral", "confidence": 0.0, "is_distress": False}
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EMOTION_PROMPT},
                {"role": "user",   "content": text},
            ],
            max_tokens=40,
            temperature=0.1,
        )
        data = json.loads(resp.choices[0].message.content.strip())
        return {
            "emotion":    data.get("emotion", "neutral"),
            "confidence": float(data.get("confidence", 0.5)),
            "is_distress": data.get("emotion", "neutral") in DISTRESS_LABELS,
        }
    except Exception as e:
        logger.warning(f"Emotion detection failed: {e}")
        return {"emotion": "neutral", "confidence": 0.0, "is_distress": False}


def get_distress_bridge() -> str:
    """Return a gentle distress-redirect phrase (cycles through 3 options)."""
    global _bridge_idx
    phrase = DISTRESS_BRIDGES[_bridge_idx % len(DISTRESS_BRIDGES)]
    _bridge_idx += 1
    return phrase


async def summarise_session(history: list[dict]) -> dict:
    """
    After a session ends, extract facts and generate a caregiver summary.
    history: list of {"role": "user"|"assistant", "content": "..."}
    Returns dict with summary, mood, new_facts, suggested_topics.
    """
    if not OPENAI_API_KEY or len(history) < 2:
        return {"summary": "", "mood": "unknown", "new_facts": [], "suggested_topics": []}

    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )
    prompt = f"""Analyse this reminiscence conversation and return ONLY valid JSON (no markdown):
{{
  "summary": "2-3 sentence plain-English summary for the caregiver",
  "mood": "positive|mixed|difficult",
  "new_facts": [{{"entity": "name/place", "type": "person|place|event|date", "detail": "..."}}],
  "suggested_topics": ["topic1", "topic2"]
}}

Transcript:
{transcript}"""

    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Session summary failed: {e}")
        return {"summary": "", "mood": "unknown", "new_facts": [], "suggested_topics": []}

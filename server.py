"""
jmoshi_pipecat/api/server.py
────────────────────────────────────────────────────────────────────────────
FastAPI server exposing the caregiver dashboard API.

Run alongside the voice bot:
  Terminal 1: python -m voice.bot         (WebSocket voice on :8765)
  Terminal 2: uvicorn api.server:app --port 8000  (REST API on :8000)

Endpoints:
  GET  /health                         — health check
  GET  /api/sessions/{user_id}         — list sessions for caregiver
  GET  /api/memory/{user_id}           — view memory graph facts
  POST /api/memory/{user_id}/fact      — manually add a fact
  POST /api/memory/{user_id}/facts/bulk — bulk import from family intake form
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from graph import MemoryGraph

memory = MemoryGraph()

@asynccontextmanager
async def lifespan(app: FastAPI):
    memory.init()
    yield

app = FastAPI(title="Jmoshi API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)


class FactInput(BaseModel):
    entity: str
    type: str    # person | place | event | date | other
    detail: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "voice_port": os.getenv("PORT", "8765"),
        "model": os.getenv("GROQ_MODEL", "qwen-qwq-32b"),
        "tts": f"Kokoro {os.getenv('KOKORO_VOICE', 'af_heart')}",
    }


@app.get("/api/sessions/{user_id}")
async def get_sessions(user_id: str, n: int = 5):
    """Caregiver: list recent sessions with summaries and mood."""
    return memory.get_sessions(user_id, n)


@app.get("/api/memory/{user_id}")
async def get_memory(user_id: str):
    """Caregiver: view all memory graph facts for a user."""
    return memory.get_all_facts(user_id)


@app.post("/api/memory/{user_id}/fact")
async def add_fact(user_id: str, fact: FactInput):
    """
    Caregiver: manually add a fact before a session.
    Example: {"entity": "Keiko", "type": "person", "detail": "wife, married 1968"}
    """
    memory.add_manual_fact(user_id, fact.entity, fact.type, fact.detail)
    return {"status": "saved"}


@app.post("/api/memory/{user_id}/facts/bulk")
async def add_facts_bulk(user_id: str, facts: list[FactInput]):
    """
    Bulk import from a family intake form.
    Example: provide 10-20 known facts before the first session.
    """
    for f in facts:
        memory.add_manual_fact(user_id, f.entity, f.type, f.detail)
    return {"status": "saved", "count": len(facts)}

"""
jmoshi_pipecat/memory/graph.py
────────────────────────────────────────────────────────────────────────────
Lightweight memory graph backed by SQLite (dev) or PostgreSQL (prod).

On every turn: retrieves relevant life facts and injects them into the
LLM system prompt as MEMORY_CONTEXT.

After every session: stores new facts extracted by brain.summarise_session().
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "jmoshi_memory.db"


@dataclass
class MemoryFact:
    user_id: str
    entity: str
    type: str          # person | place | event | date | other
    detail: str
    session_id: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class MemoryGraph:
    """
    Simple life-facts store for Jmoshi.

    Usage:
        graph = MemoryGraph()
        graph.init()

        # After session ends:
        graph.save_facts(user_id, session_id, facts_list)

        # Before each LLM turn:
        context = graph.get_context(user_id, user_text)
        # → inject into get_system_prompt(memory_context=context)
    """

    def __init__(self):
        self.db_path = DB_PATH

    def init(self):
        """Create tables if needed."""
        con = sqlite3.connect(self.db_path)
        con.executescript("""
            CREATE TABLE IF NOT EXISTS memory_facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                entity      TEXT NOT NULL,
                type        TEXT NOT NULL,
                detail      TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id       TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL,
                started_at       TEXT NOT NULL,
                ended_at         TEXT,
                summary          TEXT,
                mood             TEXT,
                suggested_topics TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_facts_user ON memory_facts(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        """)
        con.commit()
        con.close()

    # ── Facts ──────────────────────────────────────────────────────────────

    def save_facts(self, user_id: str, session_id: str, facts: list[dict]):
        """Persist extracted facts from brain.summarise_session()."""
        if not facts:
            return
        records = [
            MemoryFact(
                user_id=user_id,
                entity=f.get("entity", ""),
                type=f.get("type", "other"),
                detail=f.get("detail", ""),
                session_id=session_id,
            )
            for f in facts
            if f.get("entity") and f.get("detail")
        ]
        if not records:
            return
        con = sqlite3.connect(self.db_path)
        con.executemany(
            "INSERT INTO memory_facts (user_id,entity,type,detail,session_id,created_at) VALUES (?,?,?,?,?,?)",
            [(r.user_id, r.entity, r.type, r.detail, r.session_id, r.created_at) for r in records],
        )
        con.commit()
        con.close()

    def get_context(self, user_id: str, query_text: str) -> dict:
        """
        Return most relevant memory facts as a dict grouped by type.
        This dict is injected into the LLM system prompt.
        """
        facts = self._fetch_facts(user_id)
        if not facts:
            return {}

        # Keyword relevance scoring (simple, no embedding needed at this scale)
        query_words = set(query_text.lower().split())
        scored = []
        for f in facts:
            words = set((f.entity + " " + f.detail).lower().split())
            score = len(query_words & words)
            scored.append((score, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [f for _, f in scored[:10]]  # top 10 relevant facts

        # Group by type for cleaner injection
        grouped: dict[str, list] = {}
        for f in top:
            grouped.setdefault(f.type, []).append({
                "entity": f.entity,
                "detail": f.detail,
            })
        return grouped

    def add_manual_fact(self, user_id: str, entity: str, type_: str, detail: str):
        """Caregiver can manually add facts before a session via the API."""
        fact = MemoryFact(
            user_id=user_id, entity=entity, type=type_,
            detail=detail, session_id="manual"
        )
        con = sqlite3.connect(self.db_path)
        con.execute(
            "INSERT INTO memory_facts (user_id,entity,type,detail,session_id,created_at) VALUES (?,?,?,?,?,?)",
            (fact.user_id, fact.entity, fact.type, fact.detail, fact.session_id, fact.created_at),
        )
        con.commit()
        con.close()

    def _fetch_facts(self, user_id: str) -> list[MemoryFact]:
        con = sqlite3.connect(self.db_path)
        rows = con.execute(
            "SELECT user_id,entity,type,detail,session_id,created_at "
            "FROM memory_facts WHERE user_id=? ORDER BY created_at DESC LIMIT 200",
            (user_id,),
        ).fetchall()
        con.close()
        return [MemoryFact(*r) for r in rows]

    # ── Sessions ───────────────────────────────────────────────────────────

    def save_session(self, session_id: str, user_id: str, started_at: str,
                     ended_at: str, summary: str, mood: str, suggested_topics: list):
        con = sqlite3.connect(self.db_path)
        con.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id,user_id,started_at,ended_at,summary,mood,suggested_topics)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, user_id, started_at, ended_at,
             summary, mood, json.dumps(suggested_topics)),
        )
        con.commit()
        con.close()

    def get_sessions(self, user_id: str, n: int = 5) -> list[dict]:
        con = sqlite3.connect(self.db_path)
        rows = con.execute(
            "SELECT session_id,started_at,ended_at,summary,mood,suggested_topics "
            "FROM sessions WHERE user_id=? ORDER BY started_at DESC LIMIT ?",
            (user_id, n),
        ).fetchall()
        con.close()
        return [
            {
                "session_id": r[0], "started_at": r[1], "ended_at": r[2],
                "summary": r[3], "mood": r[4],
                "suggested_topics": json.loads(r[5]) if r[5] else [],
            }
            for r in rows
        ]

    def get_all_facts(self, user_id: str) -> list[dict]:
        facts = self._fetch_facts(user_id)
        return [
            {"entity": f.entity, "type": f.type, "detail": f.detail,
             "session": f.session_id, "date": f.created_at}
            for f in facts
        ]

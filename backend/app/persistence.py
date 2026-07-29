"""Session persistence.

SQLite is used because it's file-based and needs no server — genuinely runnable
in any environment, including this sandbox. It follows the same tiny interface
(`save_session` / `load_session` / `list_sessions` / `append_turn`) that a
Redis- or Postgres-backed store would use, so swapping backends later means
implementing `SessionStore` again, not touching `main.py`.
"""
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    repo_urls TEXT NOT NULL,        -- JSON list, supports multi-repo sessions
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
"""


class SessionStore:
    """Base interface. `SqliteSessionStore` and `InMemorySessionStore` both implement it."""

    def create_session(self, session_id: str, repo_urls: List[str]): ...
    def append_turn(self, session_id: str, question: str, answer: str): ...
    def get_history(self, session_id: str) -> List[Tuple[str, str]]: ...
    def session_exists(self, session_id: str) -> bool: ...
    def get_repo_urls(self, session_id: str) -> Optional[List[str]]: ...
    def list_sessions(self) -> List[dict]: ...


class SqliteSessionStore(SessionStore):
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_session(self, session_id: str, repo_urls: List[str]):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, repo_urls, created_at) VALUES (?, ?, ?)",
                (session_id, json.dumps(repo_urls), time.time()),
            )
            self._conn.commit()

    def append_turn(self, session_id: str, question: str, answer: str):
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                (session_id, question, answer, time.time()),
            )
            self._conn.commit()

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        cur = self._conn.execute(
            "SELECT question, answer FROM turns WHERE session_id = ? ORDER BY id", (session_id,)
        )
        return [(q, a) for q, a in cur.fetchall()]

    def session_exists(self, session_id: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
        return cur.fetchone() is not None

    def get_repo_urls(self, session_id: str) -> Optional[List[str]]:
        cur = self._conn.execute("SELECT repo_urls FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def list_sessions(self) -> List[dict]:
        cur = self._conn.execute("SELECT session_id, repo_urls, created_at FROM sessions ORDER BY created_at DESC")
        return [{"session_id": r[0], "repo_urls": json.loads(r[1]), "created_at": r[2]} for r in cur.fetchall()]


class InMemorySessionStore(SessionStore):
    """Fallback used if SESSION_STORE=memory. Lost on process restart."""

    def __init__(self):
        self._sessions: Dict[str, dict] = {}

    def create_session(self, session_id: str, repo_urls: List[str]):
        self._sessions[session_id] = {"repo_urls": repo_urls, "created_at": time.time(), "turns": []}

    def append_turn(self, session_id: str, question: str, answer: str):
        self._sessions[session_id]["turns"].append((question, answer))

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        return self._sessions.get(session_id, {}).get("turns", [])

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get_repo_urls(self, session_id: str) -> Optional[List[str]]:
        s = self._sessions.get(session_id)
        return s["repo_urls"] if s else None

    def list_sessions(self) -> List[dict]:
        return [{"session_id": k, "repo_urls": v["repo_urls"], "created_at": v["created_at"]}
                for k, v in self._sessions.items()]


def get_session_store() -> SessionStore:
    if config.SESSION_STORE == "sqlite":
        return SqliteSessionStore(config.SQLITE_PATH)
    return InMemorySessionStore()

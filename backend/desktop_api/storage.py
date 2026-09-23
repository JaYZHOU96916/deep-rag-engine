"""Persistent, Docker-free local document storage for the macOS application."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DesktopStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = self.data_dir / "uploads"
        self.uploads_dir.mkdir(exist_ok=True)
        self.path = self.data_dir / "deep-rag.sqlite3"
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    content_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    page_count INTEGER,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    task_id TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    UNIQUE(document_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS ix_desktop_chunks_document ON document_chunks(document_id);
                CREATE TABLE IF NOT EXISTS document_summaries (
                    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    summary_data TEXT,
                    source_chunk_count INTEGER NOT NULL DEFAULT 0,
                    route_provider TEXT NOT NULL,
                    route_model TEXT NOT NULL,
                    task_id TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                """
            )

    def create_document(self, *, document_id: str, filename: str, path: Path, content_type: str, size: int) -> dict:
        now = _now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO documents
                (id, original_filename, storage_path, content_type, size_bytes, status, chunk_count, task_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)""",
                (document_id, filename, str(path), content_type, size, document_id, now, now),
            )
        return self.get_document(document_id)  # type: ignore[return-value]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def unfinished_documents(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE status NOT IN ('completed', 'failed')"
            ).fetchall()
        return [dict(row) for row in rows]

    def update_document(self, document_id: str, status: str, **fields: Any) -> None:
        allowed = {"page_count", "chunk_count", "error_message", "completed_at"}
        if set(fields) - allowed:
            raise ValueError("Unsupported document field.")
        changes = {"status": status, "updated_at": _now(), **fields}
        assignments = ", ".join(f"{name} = ?" for name in changes)
        with self._connection() as connection:
            connection.execute(
                f"UPDATE documents SET {assignments} WHERE id = ?",
                (*changes.values(), document_id),
            )

    def save_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            connection.executemany(
                """INSERT INTO document_chunks (id, document_id, ordinal, page_number, text, embedding)
                VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk["id"], document_id, chunk["ordinal"], chunk["page_number"],
                        chunk["text"], json.dumps(chunk["embedding"]),
                    )
                    for chunk in chunks
                ],
            )

    def all_chunks(self, document_id: str | None = None) -> list[dict[str, Any]]:
        sql = """SELECT c.*, d.original_filename FROM document_chunks c
                 JOIN documents d ON d.id = c.document_id WHERE d.status = 'completed'"""
        params: tuple[str, ...] = ()
        if document_id:
            sql += " AND d.id = ?"
            params = (document_id,)
        sql += " ORDER BY d.created_at DESC, c.ordinal"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [{**dict(row), "embedding": json.loads(row["embedding"])} for row in rows]

    def document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM document_chunks WHERE document_id = ? ORDER BY ordinal", (document_id,)
            ).fetchall()
        return [{**dict(row), "embedding": json.loads(row["embedding"])} for row in rows]

    def put_summary(self, document_id: str, *, provider: str, model: str, task_id: str) -> dict[str, Any]:
        now = _now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO document_summaries
                (document_id, status, route_provider, route_model, task_id, created_at, updated_at)
                VALUES (?, 'pending', ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    status = 'pending', summary_data = NULL, source_chunk_count = 0,
                    route_provider = excluded.route_provider, route_model = excluded.route_model,
                    task_id = excluded.task_id, error_message = NULL,
                    completed_at = NULL, updated_at = excluded.updated_at""",
                (document_id, provider, model, task_id, now, now),
            )
        return self.get_summary(document_id)  # type: ignore[return-value]

    def get_summary(self, document_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM document_summaries WHERE document_id = ?", (document_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        summary_data = result.pop("summary_data")
        result["summary"] = json.loads(summary_data) if summary_data else None
        return result

    def unfinished_summaries(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT document_id FROM document_summaries WHERE status IN ('pending', 'summarizing')"
            ).fetchall()
        return [self.get_summary(row["document_id"]) for row in rows]  # type: ignore[misc]

    def update_summary(self, document_id: str, status: str, **fields: Any) -> None:
        allowed = {"summary_data", "source_chunk_count", "error_message", "completed_at"}
        if set(fields) - allowed:
            raise ValueError("Unsupported summary field.")
        changes = {"status": status, "updated_at": _now(), **fields}
        assignments = ", ".join(f"{name} = ?" for name in changes)
        with self._connection() as connection:
            connection.execute(
                f"UPDATE document_summaries SET {assignments} WHERE document_id = ?",
                (*changes.values(), document_id),
            )

    def cache_rows(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM semantic_cache WHERE expires_at > ?", (_now(),)).fetchall()
        return [{**dict(row), "embedding": json.loads(row["embedding"]), "citations": json.loads(row["citations"])} for row in rows]

    def put_cache(self, *, cache_id: str, query: str, embedding: list[float], answer: str, citations: list[dict], expires_at: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO semantic_cache
                (id, query, embedding, answer, citations, expires_at) VALUES (?, ?, ?, ?, ?, ?)""",
                (cache_id, query, json.dumps(embedding), answer, json.dumps(citations), expires_at),
            )
            connection.execute("DELETE FROM semantic_cache WHERE expires_at <= ?", (_now(),))

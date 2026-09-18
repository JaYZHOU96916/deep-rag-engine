"""Exercise typed SSE event ordering, semantic-cache signaling, and Last-Event-ID replay."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def parse_events(lines: Iterator[str]) -> list[dict]:
    events: list[dict] = []
    current: dict[str, str] = {}
    for line in lines:
        if not line:
            if current:
                events.append({"id": int(current["id"]), "event": current["event"], "data": json.loads(current["data"])})
                current = {}
            continue
        key, value = line.split(": ", 1)
        current[key] = value
    return events


def stream(method: str, path: str, **kwargs: object) -> tuple[str, list[dict]]:
    with httpx.Client(timeout=30) as client, client.stream(method, f"{API_BASE_URL}{path}", **kwargs) as response:
        response.raise_for_status()
        stream_id = response.headers.get("X-Stream-ID", "")
        return stream_id, parse_events(response.iter_lines())


def main() -> None:
    question = "How is page-level provenance retained from PDF ingestion to academic retrieval?"
    stream_id, first = stream("POST", "/api/v1/chat/stream", json={"question": question})
    assert stream_id
    assert {"thought", "citation", "delta"}.issubset({event["event"] for event in first})
    assert all(event["id"] > 0 for event in first)
    assert any(event["event"] == "delta" for event in first)

    replay_cursor = first[1]["id"]
    _, replay = stream("GET", f"/api/v1/chat/stream/{stream_id}", headers={"Last-Event-ID": str(replay_cursor)})
    assert replay and all(event["id"] > replay_cursor for event in replay)

    _, cached = stream("POST", "/api/v1/chat/stream", json={"question": question})
    cache_thought = next(event for event in cached if event["event"] == "thought")
    assert cache_thought["data"]["cache_hit"] is True
    print(f"Typed SSE verified: {len(first)} initial events, {len(replay)} replayed events, cache-hit replay verified.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis


@dataclass(frozen=True, slots=True)
class StoredSSEEvent:
    event_id: int
    event: str
    data: dict[str, Any]


class SSEReplayStore:
    """Redis event log that permits a browser to resume a dropped SSE connection by event id."""

    prefix = "deep-rag:sse"

    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    def _events_key(self, stream_id: str) -> str:
        return f"{self.prefix}:{stream_id}:events"

    def _sequence_key(self, stream_id: str) -> str:
        return f"{self.prefix}:{stream_id}:sequence"

    def _done_key(self, stream_id: str) -> str:
        return f"{self.prefix}:{stream_id}:done"

    async def append(self, stream_id: str, event: str, data: dict[str, Any]) -> StoredSSEEvent:
        event_id = await self.redis.incr(self._sequence_key(stream_id))
        stored = StoredSSEEvent(event_id=event_id, event=event, data=data)
        payload = json.dumps(
            {"id": event_id, "event": event, "data": data},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.rpush(self._events_key(stream_id), payload)
            pipeline.expire(self._events_key(stream_id), self.ttl_seconds)
            pipeline.expire(self._sequence_key(stream_id), self.ttl_seconds)
            await pipeline.execute()
        return stored

    async def after(self, stream_id: str, after_event_id: int) -> list[StoredSSEEvent]:
        payloads = await self.redis.lrange(self._events_key(stream_id), 0, -1)
        events: list[StoredSSEEvent] = []
        for payload in payloads:
            item = json.loads(_decode(payload))
            if int(item["id"]) > after_event_id:
                events.append(StoredSSEEvent(event_id=int(item["id"]), event=item["event"], data=item["data"]))
        return events

    async def mark_done(self, stream_id: str) -> None:
        await self.redis.set(self._done_key(stream_id), "1", ex=self.ttl_seconds)

    async def is_done(self, stream_id: str) -> bool:
        return bool(await self.redis.exists(self._done_key(stream_id)))

    async def exists(self, stream_id: str) -> bool:
        return bool(await self.redis.exists(self._events_key(stream_id)))


async def follow_sse_stream(
    store: SSEReplayStore,
    stream_id: str,
    *,
    after_event_id: int = 0,
) -> AsyncIterator[str]:
    cursor = after_event_id
    try:
        while True:
            events = await store.after(stream_id, cursor)
            for event in events:
                cursor = event.event_id
                yield encode_sse(event)
            if await store.is_done(stream_id):
                return
            await asyncio.sleep(0.05)
    finally:
        await store.redis.aclose()


def encode_sse(event: StoredSSEEvent) -> str:
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event.event}\ndata: {data}\n\n"


def _decode(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value

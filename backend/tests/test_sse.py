import json

import pytest
from fakeredis.aioredis import FakeRedis

from app.services.sse import SSEReplayStore, follow_sse_stream


@pytest.mark.asyncio
async def test_replay_store_resumes_only_events_after_last_event_id() -> None:
    store = SSEReplayStore(FakeRedis(decode_responses=True), ttl_seconds=60)
    await store.append("stream-1", "thought", {"stage": "planning"})
    await store.append("stream-1", "delta", {"text": "first"})
    await store.append("stream-1", "delta", {"text": "second"})
    await store.mark_done("stream-1")

    frames = [frame async for frame in follow_sse_stream(store, "stream-1", after_event_id=1)]

    assert len(frames) == 2
    assert frames[0].startswith("id: 2\nevent: delta")
    assert json.loads(frames[1].split("data: ", 1)[1]) == {"text": "second"}

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis

from app.core.config import get_settings
from app.schemas.chat import ChatStreamRequest
from app.services.chat import ChatStreamProducer, new_stream_id
from app.services.sse import SSEReplayStore, follow_sse_stream

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def start_chat_stream(payload: ChatStreamRequest) -> StreamingResponse:
    settings = get_settings()
    stream_id = new_stream_id()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    store = SSEReplayStore(redis, ttl_seconds=settings.sse_replay_ttl_seconds)
    task = asyncio.create_task(ChatStreamProducer(settings).produce(stream_id, payload))
    task.add_done_callback(_consume_task_exception)
    response = StreamingResponse(
        follow_sse_stream(store, stream_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Stream-ID": stream_id,
        },
    )
    return response


@router.get("/stream/{stream_id}")
async def resume_chat_stream(
    stream_id: str,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    settings = get_settings()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    store = SSEReplayStore(redis, ttl_seconds=settings.sse_replay_ttl_seconds)
    if not await store.exists(stream_id):
        await redis.aclose()
        raise HTTPException(status_code=404, detail="Stream was not found or has expired.")
    try:
        cursor = max(0, int(last_event_id or "0"))
    except ValueError as exc:
        await redis.aclose()
        raise HTTPException(status_code=400, detail="Last-Event-ID must be an integer.") from exc
    return StreamingResponse(
        follow_sse_stream(store, stream_id, after_event_id=cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _consume_task_exception(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except Exception:
        # The producer has already recorded an `error` event for the client.
        pass

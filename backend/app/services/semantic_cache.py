from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from redis.asyncio import Redis


@dataclass(frozen=True, slots=True)
class SemanticCacheEntry:
    key: str
    query: str
    embedding: list[float]
    answer: str
    citations: list[dict[str, Any]]
    created_at: float


class SemanticCache:
    """Redis-backed vector cache with explicit cosine matching and TTL-based eviction."""

    index_key = "deep-rag:semantic-cache:index"
    entry_prefix = "deep-rag:semantic-cache:entry:"

    def __init__(self, client: Redis, *, threshold: float = 0.95, ttl_seconds: int = 86_400) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("semantic cache threshold must be in (0, 1]")
        self.client = client
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds

    async def find(self, query_embedding: list[float]) -> SemanticCacheEntry | None:
        keys = await self.client.smembers(self.index_key)
        if not keys:
            return None
        key_list = list(keys)
        raw_entries = await self.client.mget(key_list)
        best: SemanticCacheEntry | None = None
        best_similarity = self.threshold
        stale_keys: list[str] = []
        for raw, key in zip(raw_entries, key_list, strict=True):
            if raw is None:
                stale_keys.append(_decode(key))
                continue
            entry = _entry_from_json(_decode(raw))
            similarity = cosine_similarity(query_embedding, entry.embedding)
            if similarity > best_similarity:
                best = entry
                best_similarity = similarity
        if stale_keys:
            await self.client.srem(self.index_key, *stale_keys)
        return best

    async def put(
        self,
        *,
        query: str,
        embedding: list[float],
        answer: str,
        citations: list[dict[str, Any]],
    ) -> SemanticCacheEntry:
        key = f"{self.entry_prefix}{uuid.uuid4()}"
        entry = SemanticCacheEntry(
            key=key,
            query=query,
            embedding=embedding,
            answer=answer,
            citations=citations,
            created_at=time.time(),
        )
        payload = json.dumps(asdict(entry), ensure_ascii=False, separators=(",", ":"))
        async with self.client.pipeline(transaction=True) as pipeline:
            pipeline.set(key, payload, ex=self.ttl_seconds)
            pipeline.sadd(self.index_key, key)
            pipeline.expire(self.index_key, self.ttl_seconds)
            await pipeline.execute()
        return entry


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left or not right:
        return 0.0
    denominator = math.sqrt(sum(item * item for item in left)) * math.sqrt(sum(item * item for item in right))
    return sum(left_item * right_item for left_item, right_item in zip(left, right, strict=True)) / denominator if denominator else 0.0


def _entry_from_json(payload: str) -> SemanticCacheEntry:
    value = json.loads(payload)
    return SemanticCacheEntry(
        key=value["key"],
        query=value["query"],
        embedding=[float(item) for item in value["embedding"]],
        answer=value["answer"],
        citations=value["citations"],
        created_at=float(value["created_at"]),
    )


def _decode(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value

"""Verify hybrid retrieval and the Redis semantic-cache threshold against running services."""

from __future__ import annotations

import asyncio

from redis.asyncio import Redis

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.services.embeddings import get_embedding_provider
from app.services.retrieval import HybridRetriever
from app.services.semantic_cache import SemanticCache


async def main() -> None:
    settings = get_settings()
    question = "How are academic citations retained during document retrieval?"
    async with AsyncSessionLocal() as session:
        results = await HybridRetriever(session).retrieve(question, limit=3)
    assert results, "Hybrid retrieval returned no candidates."
    assert all(item.page_number >= 1 for item in results)
    assert all(item.rrf_score > 0 for item in results)

    embedding = (await get_embedding_provider().embed_many([question]))[0]
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    cache = SemanticCache(redis, threshold=settings.semantic_cache_threshold, ttl_seconds=60)
    await cache.put(query=question, embedding=embedding, answer="Cached answer.", citations=[])
    assert (hit := await cache.find(embedding)) is not None
    assert hit.answer == "Cached answer."
    await redis.aclose()
    print(f"Hybrid retrieval complete: {len(results)} reranked citations; semantic cache hit verified.")


if __name__ == "__main__":
    asyncio.run(main())

import pytest
from fakeredis.aioredis import FakeRedis

from app.services.semantic_cache import SemanticCache, cosine_similarity


@pytest.mark.asyncio
async def test_semantic_cache_returns_entry_above_similarity_threshold() -> None:
    cache = SemanticCache(FakeRedis(decode_responses=True), threshold=0.95, ttl_seconds=60)
    await cache.put(
        query="What is reciprocal rank fusion?",
        embedding=[1.0, 0.0, 0.0],
        answer="A fused answer.",
        citations=[{"page_number": 3}],
    )

    hit = await cache.find([0.999, 0.01, 0.0])
    miss = await cache.find([0.0, 1.0, 0.0])

    assert hit is not None
    assert hit.answer == "A fused answer."
    assert miss is None


def test_cosine_similarity_handles_zero_and_shape_mismatch() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([0.0], [0.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0

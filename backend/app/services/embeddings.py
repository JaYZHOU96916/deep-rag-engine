from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from openai import AsyncOpenAI

from app.core.config import Settings, get_settings

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class EmbeddingProvider(Protocol):
    async def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class LocalHashEmbeddingProvider:
    """Deterministic offline embedding for development and integration tests only."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            integer = int.from_bytes(digest, "big")
            index = integer % self.dimension
            vector[index] += 1.0 if integer & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0 else [value / norm for value in vector]


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.embedding_api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required for openai_compatible embeddings.")
        self.client = AsyncOpenAI(
            api_key=settings.embedding_api_key.get_secret_value(),
            base_url=settings.embedding_base_url,
        )
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimension,
        )
        return [item.embedding for item in response.data]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "local_hash":
        return LocalHashEmbeddingProvider(settings.embedding_dimension)
    return OpenAICompatibleEmbeddingProvider(settings)

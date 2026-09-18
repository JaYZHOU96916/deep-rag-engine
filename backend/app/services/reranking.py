from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Protocol, TYPE_CHECKING

import httpx

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from app.services.retrieval import RetrievedChunk

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class LexicalReranker:
    """Offline fallback used only when no BGE-compatible reranker endpoint is configured."""

    def __init__(self, min_score: float) -> None:
        self.min_score = min_score

    async def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_terms = set(_tokenise(query))
        rescored = []
        for candidate in candidates:
            candidate_terms = set(_tokenise(candidate.text))
            if not query_terms or not candidate_terms:
                score = 0.0
            else:
                score = len(query_terms & candidate_terms) / math.sqrt(len(query_terms) * len(candidate_terms))
            if score >= self.min_score:
                rescored.append(replace(candidate, rerank_score=score))
        return sorted(rescored, key=lambda item: (-item.rerank_score, -item.rrf_score, str(item.chunk_id)))


class BGEHTTPReranker:
    """Adapter for a BGE/Cross-Encoder server exposing the common POST /rerank contract."""

    def __init__(self, endpoint: str, model: str, min_score: float) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.min_score = min_score

    async def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": [candidate.text for candidate in candidates],
            "top_n": len(candidates),
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.endpoint}/rerank", json=payload)
            response.raise_for_status()
        body = response.json()
        results = body.get("results", body.get("data", []))
        rescored: list[RetrievedChunk] = []
        for result in results:
            index = int(result["index"])
            score = float(result.get("relevance_score", result.get("score", 0.0)))
            if index < 0 or index >= len(candidates) or score < self.min_score:
                continue
            rescored.append(replace(candidates[index], rerank_score=score))
        return sorted(rescored, key=lambda item: (-item.rerank_score, -item.rrf_score, str(item.chunk_id)))


def get_reranker(settings: Settings | None = None) -> Reranker:
    current_settings = settings or get_settings()
    if current_settings.reranker_provider == "bge_http":
        if not current_settings.reranker_endpoint:
            raise RuntimeError("RERANKER_ENDPOINT is required when RERANKER_PROVIDER=bge_http.")
        return BGEHTTPReranker(
            endpoint=current_settings.reranker_endpoint,
            model=current_settings.reranker_model,
            min_score=current_settings.reranker_min_score,
        )
    return LexicalReranker(min_score=current_settings.reranker_min_score)


def _tokenise(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())

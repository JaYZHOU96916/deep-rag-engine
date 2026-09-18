from __future__ import annotations

import re
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, replace
from uuid import UUID

from rank_bm25 import BM25Okapi
from sqlalchemy import Select, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models import Document, DocumentChunk, IngestionStatus
from app.services.embeddings import EmbeddingProvider, get_embedding_provider
from app.services.reranking import Reranker, get_reranker

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    document_name: str
    page_number: int
    text: str
    rrf_score: float = 0.0
    rerank_score: float = 0.0


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Iterable[RetrievedChunk]],
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse rank lists with the RRF formula Σ 1 / (k + rank), ranks starting at one."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least one")
    fused: dict[Hashable, RetrievedChunk] = {}
    scores: dict[Hashable, float] = {}
    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list, start=1):
            key = candidate.chunk_id
            fused.setdefault(key, candidate)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(
        (replace(candidate, rrf_score=scores[key]) for key, candidate in fused.items()),
        key=lambda item: (-item.rrf_score, str(item.chunk_id)),
    )


class HybridRetriever:
    """Dense pgvector HNSW + in-process rank_bm25 retrieval, merged by hand-written RRF."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.reranker = reranker or get_reranker(self.settings)

    async def retrieve(
        self,
        question: str,
        *,
        hypothetical_document: str | None = None,
        limit: int = 8,
    ) -> list[RetrievedChunk]:
        expansion = hypothetical_document.strip() if hypothetical_document else ""
        search_text = f"{question}\n{expansion}" if expansion else question
        query_embedding = (await self.embedding_provider.embed_many([search_text]))[0]
        dense, sparse = await self._retrieve_parallel_inputs(search_text, query_embedding)
        fused = reciprocal_rank_fusion((dense, sparse), self.settings.rrf_k)
        reranked = await self.reranker.rerank(question, fused[: max(limit * 3, limit)])
        return reranked[:limit]

    async def _retrieve_parallel_inputs(
        self,
        query_text: str,
        query_embedding: list[float],
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        dense = await self._dense_retrieve(query_embedding)
        sparse = await self._sparse_retrieve(query_text)
        return dense, sparse

    async def _dense_retrieve(self, query_embedding: list[float]) -> list[RetrievedChunk]:
        await self.session.execute(text("SET LOCAL hnsw.ef_search = 100"))
        distance = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
        statement: Select = (
            select(DocumentChunk, Document.original_filename, distance)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                Document.status == IngestionStatus.COMPLETED,
                DocumentChunk.embedding.is_not(None),
            )
            .order_by(distance)
            .limit(self.settings.dense_candidate_limit)
        )
        result = await self.session.execute(statement)
        return [
            _to_retrieved_chunk(chunk, document_name)
            for chunk, document_name, _distance in result.all()
        ]

    async def _sparse_retrieve(self, query_text: str) -> list[RetrievedChunk]:
        statement = (
            select(DocumentChunk, Document.original_filename)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == IngestionStatus.COMPLETED)
        )
        rows = (await self.session.execute(statement)).all()
        if not rows:
            return []
        corpus = [_tokenise(chunk.text) for chunk, _document_name in rows]
        scores = BM25Okapi(corpus).get_scores(_tokenise(query_text))
        ranked_indexes = sorted(range(len(rows)), key=lambda index: (-float(scores[index]), index))
        return [
            _to_retrieved_chunk(*rows[index])
            for index in ranked_indexes[: self.settings.sparse_candidate_limit]
            if scores[index] > 0
        ]


def _to_retrieved_chunk(chunk: DocumentChunk, document_name: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_name=document_name,
        page_number=chunk.page_number,
        text=chunk.text,
    )


def _tokenise(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())

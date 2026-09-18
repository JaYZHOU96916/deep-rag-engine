from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)
    hypothetical_document: str | None = Field(default=None, max_length=16_000)
    limit: int = Field(default=8, ge=1, le=20)


class CitationResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_name: str
    page_number: int
    excerpt: str
    rrf_score: float
    rerank_score: float


class SearchResponse(BaseModel):
    query: str
    citations: list[CitationResponse]

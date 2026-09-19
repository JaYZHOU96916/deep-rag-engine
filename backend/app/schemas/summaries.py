from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import SummaryStatus


class SummaryRoute(BaseModel):
    provider: Literal["local", "openai", "deepseek", "claude"] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)


class BilingualSummaryRequest(BaseModel):
    route: SummaryRoute | None = None
    regenerate: bool = False


class SummaryCitation(BaseModel):
    chunk_id: UUID
    page_number: int = Field(ge=1)


class BilingualClaim(BaseModel):
    english: str = Field(min_length=1, max_length=2_000)
    chinese: str = Field(min_length=1, max_length=2_000)
    citations: list[SummaryCitation] = Field(min_length=1, max_length=5)


class BilingualSummaryData(BaseModel):
    english_summary: str = Field(min_length=1, max_length=6_000)
    chinese_summary: str = Field(min_length=1, max_length=6_000)
    overview_citations: list[SummaryCitation] = Field(min_length=1, max_length=8)
    key_findings: list[BilingualClaim] = Field(default_factory=list, max_length=8)
    methods: list[BilingualClaim] = Field(default_factory=list, max_length=6)
    limitations: list[BilingualClaim] = Field(default_factory=list, max_length=6)
    translation_mode: Literal["llm_bilingual", "local_verification"]


class BilingualSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    status: SummaryStatus
    task_id: str | None
    source_chunk_count: int
    route_provider: str
    route_model: str
    error_message: str | None
    summary: BilingualSummaryData | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

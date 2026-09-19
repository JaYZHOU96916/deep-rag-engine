from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PaperSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2_000)
    limit: int = Field(default=8, ge=1, le=20)
    institution_id: str | None = Field(default=None, max_length=64)


class InstitutionResponse(BaseModel):
    id: str
    name: str


class PaperResult(BaseModel):
    title: str
    authors: list[str]
    publication_year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    landing_page_url: str | None = None
    open_access_url: str | None = None
    open_access_pdf_url: str | None = None
    citation_count: int | None = None
    sources: list[str]
    access_type: Literal["open_access", "institution_login", "metadata_only"]
    institution_name: str | None = None
    institution_access_url: str | None = None


class PaperSearchResponse(BaseModel):
    query: str
    is_doi_lookup: bool
    results: list[PaperResult]


class PaperImportRequest(BaseModel):
    pdf_url: str = Field(min_length=12, max_length=2_048)
    title: str = Field(min_length=1, max_length=500)

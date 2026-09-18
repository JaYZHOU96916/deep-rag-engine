from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.db.models import IngestionStatus


class DocumentUploadResponse(BaseModel):
    id: UUID
    status: IngestionStatus
    task_id: str


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    original_filename: str
    status: IngestionStatus
    page_count: int | None
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

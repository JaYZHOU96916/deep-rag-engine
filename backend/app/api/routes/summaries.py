from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Document, DocumentSummary, IngestionStatus, SummaryStatus
from app.db.session import get_db_session
from app.schemas.summaries import BilingualSummaryData, BilingualSummaryRequest, BilingualSummaryResponse
from app.services.llm import resolve_route
from app.tasks.summaries import generate_bilingual_summary_task

router = APIRouter(prefix="/documents", tags=["bilingual summaries"])


@router.post(
    "/{document_id}/bilingual-summary",
    response_model=BilingualSummaryResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_bilingual_summary(
    document_id: uuid.UUID,
    payload: BilingualSummaryRequest,
    session: AsyncSession = Depends(get_db_session),
) -> BilingualSummaryResponse:
    """Queue an opt-in summary only after the browser explicitly requests it."""
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if document.status != IngestionStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="The document must finish ingestion before it can be summarized.")

    summary = await session.scalar(
        select(DocumentSummary).where(DocumentSummary.document_id == document_id).with_for_update()
    )
    if summary is not None:
        if summary.status in {SummaryStatus.PENDING, SummaryStatus.SUMMARIZING}:
            return _response(summary)
        if summary.status == SummaryStatus.COMPLETED and not payload.regenerate:
            return _response(summary)

    settings = get_settings()
    route = resolve_route(
        payload.route.provider if payload.route else None,
        payload.route.model if payload.route else None,
        settings,
    )
    if summary is None:
        summary = DocumentSummary(
            document_id=document_id,
            status=SummaryStatus.PENDING,
            route_provider=route.provider,
            route_model=route.model,
        )
        session.add(summary)
    else:
        summary.status = SummaryStatus.PENDING
        summary.summary_data = None
        summary.source_chunk_count = 0
        summary.error_message = None
        summary.completed_at = None
        summary.route_provider = route.provider
        summary.route_model = route.model
        summary.task_id = None
    await session.commit()
    await session.refresh(summary)

    task = generate_bilingual_summary_task.delay(str(summary.id))
    summary.task_id = task.id
    await session.commit()
    await session.refresh(summary)
    return _response(summary)


@router.get("/{document_id}/bilingual-summary", response_model=BilingualSummaryResponse)
async def get_bilingual_summary(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> BilingualSummaryResponse:
    summary = await session.scalar(select(DocumentSummary).where(DocumentSummary.document_id == document_id))
    if summary is None:
        raise HTTPException(status_code=404, detail="No bilingual summary has been requested for this document.")
    return _response(summary)


def _response(summary: DocumentSummary) -> BilingualSummaryResponse:
    data = BilingualSummaryData.model_validate(summary.summary_data) if summary.summary_data else None
    return BilingualSummaryResponse(
        document_id=summary.document_id,
        status=summary.status,
        task_id=summary.task_id,
        source_chunk_count=summary.source_chunk_count,
        route_provider=summary.route_provider,
        route_model=summary.route_model,
        error_message=summary.error_message,
        summary=data,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        completed_at=summary.completed_at,
    )

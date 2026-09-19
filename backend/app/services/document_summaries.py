"""Database state machine for opt-in bilingual summary generation."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import DocumentChunk, DocumentSummary, SummaryStatus
from app.db.session import AsyncSessionLocal
from app.services.bilingual_summaries import BilingualSummaryGenerator, SummarySourceChunk
from app.services.llm import get_llm_client, resolve_route


async def generate_document_summary(summary_id: UUID) -> None:
    """Generate one summary outside the request process and persist a terminal state."""
    try:
        async with AsyncSessionLocal() as session:
            summary = await session.get(DocumentSummary, summary_id)
            if summary is None:
                return
            summary.status = SummaryStatus.SUMMARIZING
            summary.error_message = None
            await session.commit()

            chunks = list(
                (
                    await session.scalars(
                        select(DocumentChunk)
                        .where(DocumentChunk.document_id == summary.document_id)
                        .order_by(DocumentChunk.ordinal)
                    )
                ).all()
            )
            settings = get_settings()
            route = resolve_route(summary.route_provider, summary.route_model, settings)
            generator = BilingualSummaryGenerator(
                get_llm_client(route, settings),
                batch_chars=settings.bilingual_summary_batch_chars,
            )
            data = await generator.generate(
                [SummarySourceChunk(id=chunk.id, page_number=chunk.page_number, text=chunk.text) for chunk in chunks],
                local_verification=route.provider == "local",
            )
            summary.summary_data = data.model_dump(mode="json")
            summary.source_chunk_count = len(chunks)
            summary.status = SummaryStatus.COMPLETED
            summary.completed_at = datetime.now(UTC)
            await session.commit()
    except Exception as exc:
        async with AsyncSessionLocal() as failure_session:
            summary = await failure_session.get(DocumentSummary, summary_id)
            if summary is not None:
                summary.status = SummaryStatus.FAILED
                summary.error_message = str(exc)[:2_000]
                await failure_session.commit()
        raise

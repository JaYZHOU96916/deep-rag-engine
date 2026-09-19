from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from app.infrastructure.celery_app import celery_app
from app.services.document_summaries import generate_document_summary

logger = logging.getLogger(__name__)


@celery_app.task(name="deep_rag.generate_bilingual_summary", bind=True)
def generate_bilingual_summary_task(self, summary_id: str) -> dict[str, str]:
    try:
        asyncio.run(generate_document_summary(UUID(summary_id)))
    except Exception:
        logger.exception("Bilingual summary generation failed", extra={"summary_id": summary_id})
        raise
    return {"summary_id": summary_id, "status": "completed"}

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from app.infrastructure.celery_app import celery_app
from app.services.ingestion import ingest_document

logger = logging.getLogger(__name__)


@celery_app.task(name="deep_rag.ingest_document", bind=True)
def ingest_document_task(self, document_id: str) -> dict[str, str]:
    """Run the blocking ingestion state machine outside the request process."""
    try:
        asyncio.run(ingest_document(UUID(document_id)))
    except Exception:
        logger.exception("Document ingestion failed", extra={"document_id": document_id})
        raise
    return {"document_id": document_id, "status": "completed"}

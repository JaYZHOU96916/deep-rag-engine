from __future__ import annotations

import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Document, IngestionStatus
from app.db.session import get_db_session
from app.schemas.documents import DocumentStatusResponse, DocumentUploadResponse
from app.tasks.ingestion import ingest_document_task

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentUploadResponse:
    settings = get_settings()
    filename = file.filename or "document.pdf"
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported.")

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    storage_path = settings.uploads_dir / f"{uuid.uuid4()}.pdf"
    byte_count = 0
    first_bytes = b""
    try:
        async with aiofiles.open(storage_path, "wb") as destination:
            while chunk := await file.read(1024 * 1024):
                if not first_bytes:
                    first_bytes = chunk[:8]
                byte_count += len(chunk)
                if byte_count > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="PDF exceeds the configured upload limit.")
                await destination.write(chunk)
        if not first_bytes.startswith(b"%PDF-"):
            raise HTTPException(status_code=415, detail="Uploaded content is not a PDF.")

        document = Document(
            original_filename=filename,
            storage_path=str(storage_path),
            content_type=file.content_type,
            size_bytes=byte_count,
            status=IngestionStatus.PENDING,
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)

        task = ingest_document_task.delay(str(document.id))
        document.celery_task_id = task.id
        await session.commit()
        return DocumentUploadResponse(id=document.id, status=document.status, task_id=task.id)
    except HTTPException:
        if storage_path.exists():
            storage_path.unlink()
        raise
    except Exception:
        if storage_path.exists():
            storage_path.unlink()
        raise
    finally:
        await file.close()


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentStatusResponse:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentStatusResponse.model_validate(document)

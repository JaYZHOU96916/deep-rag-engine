import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Document, IngestionStatus
from app.db.session import get_db_session
from app.schemas.documents import DocumentUploadResponse
from app.schemas.papers import InstitutionResponse, PaperImportRequest, PaperSearchRequest, PaperSearchResponse
from app.services.open_access_import import RemotePdfDownloadError, UnsafeRemotePdfURL, download_open_access_pdf
from app.services.paper_discovery import (
    InstitutionConfigurationError,
    PaperDiscoveryService,
    PaperDiscoveryUnavailable,
    UnknownInstitution,
)
from app.tasks.ingestion import ingest_document_task

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/institutions", response_model=list[InstitutionResponse])
async def list_institutions() -> list[InstitutionResponse]:
    try:
        return PaperDiscoveryService().institutions()
    except InstitutionConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Institution connectors are unavailable.") from error


@router.post("/search", response_model=PaperSearchResponse)
async def search_papers(payload: PaperSearchRequest) -> PaperSearchResponse:
    try:
        return await PaperDiscoveryService().search(
            payload.query,
            limit=payload.limit,
            institution_id=payload.institution_id,
        )
    except UnknownInstitution as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown institution.") from error
    except InstitutionConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Institution connectors are unavailable.") from error
    except PaperDiscoveryUnavailable as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Paper search is temporarily unavailable.") from error


@router.post("/import", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def import_open_access_pdf(
    payload: PaperImportRequest,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentUploadResponse:
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    storage_path = settings.uploads_dir / f"{uuid.uuid4()}.pdf"
    try:
        size_bytes = await download_open_access_pdf(
            payload.pdf_url,
            storage_path,
            max_bytes=settings.max_upload_bytes,
        )
        document = Document(
            original_filename=_paper_filename(payload.title),
            storage_path=str(storage_path),
            content_type="application/pdf",
            size_bytes=size_bytes,
            status=IngestionStatus.PENDING,
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)
        task = ingest_document_task.delay(str(document.id))
        document.celery_task_id = task.id
        await session.commit()
        return DocumentUploadResponse(id=document.id, status=document.status, task_id=task.id)
    except UnsafeRemotePdfURL as error:
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except RemotePdfDownloadError as error:
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except Exception:
        if storage_path.exists():
            storage_path.unlink()
        raise


def _paper_filename(title: str) -> str:
    stem = re.sub(r"[^\w.-]+", "-", title, flags=re.UNICODE).strip(".-")[:120]
    return f"{stem or 'open-access-paper'}.pdf"

"""Loopback-only API used by the packaged desktop application."""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.schemas.chat import ChatStreamRequest
from app.schemas.documents import DocumentStatusResponse, DocumentUploadResponse
from app.schemas.papers import InstitutionResponse, PaperImportRequest, PaperSearchRequest, PaperSearchResponse
from app.schemas.retrieval import CitationResponse, SearchRequest, SearchResponse
from app.schemas.summaries import BilingualSummaryRequest, BilingualSummaryResponse
from app.services.open_access_import import RemotePdfDownloadError, UnsafeRemotePdfURL, download_open_access_pdf
from app.services.paper_discovery import PaperDiscoveryService, PaperDiscoveryUnavailable, UnknownInstitution

from desktop_api.engine import DesktopEngine

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class DesktopKeyRequest(BaseModel):
    provider: str = Field(pattern="^(openai|deepseek|claude)$")
    api_key: str = Field(min_length=8, max_length=4096)


def create_app(*, data_dir: Path, frontend_dir: Path | None = None, session_token: str | None = None) -> FastAPI:
    engine = DesktopEngine(data_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await engine.recover()
        yield
        for task in tuple(engine.tasks):
            task.cancel()
        if engine.tasks:
            await asyncio.gather(*engine.tasks, return_exceptions=True)

    app = FastAPI(title="Deep-RAG Desktop API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path.startswith("/api/v1/") and session_token:
            if request.headers.get("X-Desktop-Session") != session_token:
                return JSONResponse(status_code=401, content={"detail": "Desktop session is not authorized."})
        return await call_next(request)

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok", "mode": "desktop"}

    @app.get("/api/v1/desktop/settings")
    async def key_status() -> dict:
        return {"configured_providers": await asyncio.to_thread(engine.key_status)}

    @app.post("/api/v1/desktop/settings")
    async def save_key(payload: DesktopKeyRequest) -> dict:
        await asyncio.to_thread(engine.save_key, payload.provider, payload.api_key)
        return {"configured_providers": await asyncio.to_thread(engine.key_status)}

    @app.get("/api/v1/documents", response_model=list[DocumentStatusResponse])
    async def list_documents() -> list[dict]:
        return await asyncio.to_thread(engine.store.list_documents)

    @app.post("/api/v1/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
    async def upload_document(file: UploadFile = File(...)) -> dict:
        filename = file.filename or "document.pdf"
        if Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=415, detail="Only PDF uploads are supported.")
        document_id = str(uuid.uuid4())
        destination = engine.store.uploads_dir / f"{document_id}.pdf"
        count = 0
        prefix = b""
        try:
            async with aiofiles.open(destination, "wb") as output:
                while chunk := await file.read(1024 * 1024):
                    count += len(chunk)
                    if count > _MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="PDF exceeds the upload limit.")
                    if len(prefix) < 8:
                        prefix += chunk[:8 - len(prefix)]
                    await output.write(chunk)
            if not prefix.startswith(b"%PDF-"):
                raise HTTPException(status_code=415, detail="Uploaded content is not a PDF.")
            await asyncio.to_thread(
                engine.store.create_document, document_id=document_id, filename=filename,
                path=destination, content_type=file.content_type or "application/pdf", size=count,
            )
            engine.schedule(engine.ingest(document_id))
            return {"id": document_id, "status": "pending", "task_id": document_id}
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get("/api/v1/documents/{document_id}", response_model=DocumentStatusResponse)
    async def get_document(document_id: uuid.UUID) -> dict:
        document = await asyncio.to_thread(engine.store.get_document, str(document_id))
        if not document:
            raise HTTPException(status_code=404, detail="Document not found.")
        return document

    @app.post("/api/v1/documents/{document_id}/bilingual-summary", response_model=BilingualSummaryResponse, status_code=202)
    async def request_summary(document_id: uuid.UUID, payload: BilingualSummaryRequest) -> dict:
        identifier = str(document_id)
        document = await asyncio.to_thread(engine.store.get_document, identifier)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found.")
        if document["status"] != "completed":
            raise HTTPException(status_code=409, detail="The document must finish ingestion first.")
        existing = await asyncio.to_thread(engine.store.get_summary, identifier)
        if existing and (existing["status"] in {"pending", "summarizing"} or (existing["status"] == "completed" and not payload.regenerate)):
            return existing
        route = engine.route(payload.route.provider if payload.route else None, payload.route.model if payload.route else None)
        summary = await asyncio.to_thread(
            engine.store.put_summary, identifier, provider=route.provider,
            model=route.model, task_id=str(uuid.uuid4()),
        )
        engine.schedule(engine.summarize(identifier))
        return summary

    @app.get("/api/v1/documents/{document_id}/bilingual-summary", response_model=BilingualSummaryResponse)
    async def get_summary(document_id: uuid.UUID) -> dict:
        summary = await asyncio.to_thread(engine.store.get_summary, str(document_id))
        if not summary:
            raise HTTPException(status_code=404, detail="No bilingual summary has been requested.")
        return summary

    @app.post("/api/v1/retrieval/search", response_model=SearchResponse)
    async def search_chunks(payload: SearchRequest) -> dict:
        results = await engine.retrieve(payload.question, hypothetical_document=payload.hypothetical_document or "", limit=payload.limit)
        return {
            "query": payload.question,
            "citations": [
                CitationResponse(
                    chunk_id=item.chunk_id, document_id=item.document_id,
                    document_name=item.document_name, page_number=item.page_number,
                    excerpt=item.text, rrf_score=item.rrf_score, rerank_score=item.rerank_score,
                )
                for item in results
            ],
        }

    @app.post("/api/v1/chat/stream")
    async def start_chat(payload: ChatStreamRequest) -> StreamingResponse:
        stream_id = engine.hub.create()
        engine.schedule(engine.produce_chat(stream_id, payload))
        return StreamingResponse(
            engine.hub.follow(stream_id), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Stream-ID": stream_id},
        )

    @app.get("/api/v1/chat/stream/{stream_id}")
    async def resume_chat(stream_id: str, last_event_id: str | None = Header(default=None, alias="Last-Event-ID")) -> StreamingResponse:
        if stream_id not in engine.hub.streams:
            raise HTTPException(status_code=404, detail="Stream not found or expired.")
        try:
            cursor = int(last_event_id or "0")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Last-Event-ID must be an integer.") from exc
        return StreamingResponse(engine.hub.follow(stream_id, cursor), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/v1/papers/institutions", response_model=list[InstitutionResponse])
    async def institutions() -> list[InstitutionResponse]:
        return PaperDiscoveryService().institutions()

    @app.post("/api/v1/papers/search", response_model=PaperSearchResponse)
    async def search_papers(payload: PaperSearchRequest) -> PaperSearchResponse:
        try:
            return await PaperDiscoveryService().search(payload.query, limit=payload.limit, institution_id=payload.institution_id)
        except UnknownInstitution as exc:
            raise HTTPException(status_code=422, detail="Unknown institution.") from exc
        except PaperDiscoveryUnavailable as exc:
            raise HTTPException(status_code=503, detail="Paper search is temporarily unavailable.") from exc

    @app.post("/api/v1/papers/import", response_model=DocumentUploadResponse, status_code=202)
    async def import_pdf(payload: PaperImportRequest) -> dict:
        document_id = str(uuid.uuid4())
        destination = engine.store.uploads_dir / f"{document_id}.pdf"
        try:
            size = await download_open_access_pdf(payload.pdf_url, destination, max_bytes=_MAX_UPLOAD_BYTES)
            filename = re.sub(r"[^\w.-]+", "-", payload.title, flags=re.UNICODE).strip(".-")[:120] or "paper"
            await asyncio.to_thread(
                engine.store.create_document, document_id=document_id, filename=f"{filename}.pdf",
                path=destination, content_type="application/pdf", size=size,
            )
            engine.schedule(engine.ingest(document_id))
            return {"id": document_id, "status": "pending", "task_id": document_id}
        except (UnsafeRemotePdfURL, RemotePdfDownloadError) as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    if frontend_dir and (frontend_dir / "index.html").exists():
        next_assets = frontend_dir / "_next"
        if next_assets.exists():
            app.mount("/_next", StaticFiles(directory=next_assets), name="next-assets")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

        @app.get("/favicon.ico")
        async def favicon() -> FileResponse:
            icon = frontend_dir / "favicon.ico"
            if not icon.exists():
                raise HTTPException(status_code=404)
            return FileResponse(icon)

    return app


app = create_app(
    data_dir=Path(os.environ.get("DEEP_RAG_DESKTOP_DATA_DIR", Path.home() / "Library" / "Application Support" / "Deep-RAG")),
    frontend_dir=Path(os.environ["DEEP_RAG_DESKTOP_FRONTEND_DIR"]) if "DEEP_RAG_DESKTOP_FRONTEND_DIR" in os.environ else None,
    session_token=os.environ.get("DEEP_RAG_DESKTOP_SESSION_TOKEN"),
)

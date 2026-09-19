from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router
from app.api.routes.papers import router as papers_router
from app.api.routes.retrieval import router as retrieval_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Last-Event-ID"],
)
app.include_router(documents_router, prefix="/api/v1")
app.include_router(papers_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/healthz", tags=["health"])
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

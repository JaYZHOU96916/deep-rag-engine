from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.schemas.retrieval import CitationResponse, SearchRequest, SearchResponse
from app.services.retrieval import HybridRetriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    payload: SearchRequest,
    session: AsyncSession = Depends(get_db_session),
) -> SearchResponse:
    results = await HybridRetriever(session).retrieve(
        payload.question,
        hypothetical_document=payload.hypothetical_document,
        limit=payload.limit,
    )
    return SearchResponse(
        query=payload.question,
        citations=[
            CitationResponse(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                document_name=item.document_name,
                page_number=item.page_number,
                excerpt=item.text,
                rrf_score=item.rrf_score,
                rerank_score=item.rerank_score,
            )
            for item in results
        ],
    )

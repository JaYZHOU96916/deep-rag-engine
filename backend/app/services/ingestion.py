from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select

from app.db.models import Document, DocumentChunk, IngestionStatus
from app.db.session import AsyncSessionLocal
from app.services.chunking import RecursiveTextSplitter
from app.services.embeddings import get_embedding_provider
from app.services.pdf_parser import PDFParser


async def ingest_document(document_id: UUID) -> None:
    """Execute parse -> recursive chunk -> embed -> persist as explicit state transitions."""
    async with AsyncSessionLocal() as session:
        document = await session.scalar(select(Document).where(Document.id == document_id).with_for_update())
        if document is None:
            raise LookupError(f"Document {document_id} was not found.")
        if document.status == IngestionStatus.COMPLETED:
            return

        try:
            document.status = IngestionStatus.PARSING
            document.processing_started_at = datetime.now(timezone.utc)
            document.error_message = None
            await session.commit()

            parsed = PDFParser.parse(document.storage_path)
            document.status = IngestionStatus.CHUNKING
            document.page_count = parsed.page_count
            await session.commit()

            splitter = RecursiveTextSplitter()
            pending_chunks: list[tuple[int, int, str, int, int, int]] = []
            ordinal = 0
            for page in parsed.pages:
                for chunk in splitter.split(page.text):
                    pending_chunks.append(
                        (
                            ordinal,
                            page.page_number,
                            chunk.text,
                            chunk.char_start,
                            chunk.char_end,
                            chunk.token_estimate,
                        )
                    )
                    ordinal += 1

            if not pending_chunks:
                raise ValueError("The PDF does not contain extractable text.")

            document.status = IngestionStatus.EMBEDDING
            await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
            await session.commit()

            provider = get_embedding_provider()
            embeddings = await provider.embed_many([item[2] for item in pending_chunks])
            if len(embeddings) != len(pending_chunks):
                raise RuntimeError("Embedding provider returned an unexpected number of vectors.")

            session.add_all(
                DocumentChunk(
                    document_id=document.id,
                    ordinal=ordinal,
                    page_number=page_number,
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                    token_estimate=token_estimate,
                    metadata_json={"source": document.original_filename},
                    embedding=embedding,
                )
                for (ordinal, page_number, text, char_start, char_end, token_estimate), embedding in zip(
                    pending_chunks,
                    embeddings,
                    strict=True,
                )
            )
            document.chunk_count = len(pending_chunks)
            document.status = IngestionStatus.COMPLETED
            document.completed_at = datetime.now(timezone.utc)
            await session.commit()
        except Exception as exc:
            await session.rollback()
            document = await session.get(Document, document_id)
            if document is not None:
                document.status = IngestionStatus.FAILED
                document.error_message = str(exc)[:4_000]
                document.completed_at = datetime.now(timezone.utc)
                await session.commit()
            raise

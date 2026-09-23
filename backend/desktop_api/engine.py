"""Local ingestion, hybrid retrieval, summaries, and replayable chat streams."""

from __future__ import annotations

import asyncio
import json
import math
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import keyring
from pydantic import SecretStr
from rank_bm25 import BM25Okapi

from app.core.config import Settings
from app.schemas.chat import ChatStreamRequest
from app.schemas.summaries import BilingualSummaryRequest
from app.services.bilingual_summaries import BilingualSummaryGenerator, SummarySourceChunk
from app.services.chat import enforce_grounded_citations
from app.services.chunking import RecursiveTextSplitter
from app.services.embeddings import LocalHashEmbeddingProvider
from app.services.llm import LLMRoute, get_llm_client
from app.services.pdf_parser import PDFParser
from app.services.prompting import build_answer_messages, build_hyde_messages, build_self_rag_verification_messages
from app.services.reranking import LexicalReranker
from app.services.retrieval import RetrievedChunk, reciprocal_rank_fusion

from desktop_api.storage import DesktopStore

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_MODEL_DEFAULTS = {
    "local": "local-verification",
    "openai": "gpt-4.1-mini",
    "deepseek": "deepseek-chat",
    "claude": "claude-sonnet-4-5",
}
_KEYCHAIN_SERVICE = "com.jayzhou.deep-rag.desktop"


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(slots=True)
class _Stream:
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class StreamHub:
    def __init__(self) -> None:
        self.streams: dict[str, _Stream] = {}

    def create(self) -> str:
        cutoff = datetime.now(UTC) - timedelta(hours=1)
        self.streams = {stream_id: item for stream_id, item in self.streams.items() if item.created_at > cutoff}
        stream_id = str(uuid.uuid4())
        self.streams[stream_id] = _Stream()
        return stream_id

    async def append(self, stream_id: str, event: str, data: dict[str, Any]) -> None:
        stream = self.streams[stream_id]
        async with stream.condition:
            stream.events.append({"id": len(stream.events) + 1, "event": event, "data": data})
            stream.condition.notify_all()

    async def finish(self, stream_id: str) -> None:
        stream = self.streams[stream_id]
        async with stream.condition:
            stream.done = True
            stream.condition.notify_all()

    async def follow(self, stream_id: str, after_id: int = 0) -> AsyncIterator[str]:
        stream = self.streams[stream_id]
        cursor = max(after_id, 0)
        while True:
            async with stream.condition:
                await stream.condition.wait_for(lambda: len(stream.events) > cursor or stream.done)
                pending = stream.events[cursor:]
                done = stream.done
            for item in pending:
                cursor = item["id"]
                yield f"id: {cursor}\nevent: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
            if done:
                return


class DesktopEngine:
    def __init__(self, data_dir: Path) -> None:
        self.store = DesktopStore(data_dir)
        self.embedder = LocalHashEmbeddingProvider(384)
        self.hub = StreamHub()
        self.tasks: set[asyncio.Task[Any]] = set()
        self.ingestion_limit = asyncio.Semaphore(2)
        self.summary_limit = asyncio.Semaphore(2)
        self.settings = Settings(_env_file=None)

    def schedule(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def recover(self) -> None:
        for document in await asyncio.to_thread(self.store.unfinished_documents):
            self.schedule(self.ingest(document["id"]))
        for summary in await asyncio.to_thread(self.store.unfinished_summaries):
            if summary:
                self.schedule(self.summarize(summary["document_id"]))

    async def ingest(self, document_id: str) -> None:
        async with self.ingestion_limit:
            try:
                document = await asyncio.to_thread(self.store.get_document, document_id)
                if not document:
                    return
                await asyncio.to_thread(self.store.update_document, document_id, "parsing", error_message=None)
                parsed = await asyncio.to_thread(PDFParser.parse, Path(document["storage_path"]))
                await asyncio.to_thread(self.store.update_document, document_id, "chunking", page_count=parsed.page_count)
                splitter = RecursiveTextSplitter()
                chunks: list[dict[str, Any]] = []
                for page in parsed.pages:
                    for item in splitter.split(page.text):
                        chunks.append({
                            "id": str(uuid.uuid4()),
                            "ordinal": len(chunks),
                            "page_number": page.page_number,
                            "text": item.text,
                        })
                if not chunks:
                    raise ValueError("This PDF contains no extractable text.")
                await asyncio.to_thread(self.store.update_document, document_id, "embedding")
                for start in range(0, len(chunks), 64):
                    batch = chunks[start:start + 64]
                    vectors = await self.embedder.embed_many([item["text"] for item in batch])
                    for item, vector in zip(batch, vectors, strict=True):
                        item["embedding"] = vector
                await asyncio.to_thread(self.store.save_chunks, document_id, chunks)
                await asyncio.to_thread(
                    self.store.update_document, document_id, "completed",
                    chunk_count=len(chunks), completed_at=datetime.now(UTC).isoformat(),
                )
            except Exception as exc:
                await asyncio.to_thread(self.store.update_document, document_id, "failed", error_message=str(exc)[:2000])

    async def retrieve(self, question: str, *, hypothetical_document: str = "", limit: int = 8) -> list[RetrievedChunk]:
        rows = await asyncio.to_thread(self.store.all_chunks)
        if not rows:
            return []
        expanded = f"{question}\n{hypothetical_document}" if hypothetical_document else question
        query_embedding = (await self.embedder.embed_many([expanded]))[0]
        candidates = [
            RetrievedChunk(
                chunk_id=uuid.UUID(row["id"]), document_id=uuid.UUID(row["document_id"]),
                document_name=row["original_filename"], page_number=row["page_number"], text=row["text"],
            )
            for row in rows
        ]
        dense_indexes = sorted(
            range(len(rows)), key=lambda index: -_cosine(query_embedding, rows[index]["embedding"])
        )[:24]
        corpus = [_tokens(row["text"]) or [""] for row in rows]
        sparse_scores = BM25Okapi(corpus).get_scores(_tokens(expanded)) if _tokens(expanded) else [0.0] * len(rows)
        sparse_indexes = sorted(range(len(rows)), key=lambda index: -float(sparse_scores[index]))[:24]
        sparse = [candidates[index] for index in sparse_indexes if sparse_scores[index] > 0]
        fused = reciprocal_rank_fusion(
            ([candidates[index] for index in dense_indexes], sparse), rrf_k=60
        )
        ranked = await LexicalReranker(min_score=0.0).rerank(question, fused[:max(limit * 3, limit)])
        return ranked[:limit]

    def route(self, provider: str | None, model: str | None) -> LLMRoute:
        chosen = provider or "local"
        if chosen not in _MODEL_DEFAULTS:
            raise ValueError("Unknown model provider.")
        return LLMRoute(provider=chosen, model=model or _MODEL_DEFAULTS[chosen])  # type: ignore[arg-type]

    def llm(self, route: LLMRoute):
        if route.provider == "local":
            return get_llm_client(route, self.settings)
        key = keyring.get_password(_KEYCHAIN_SERVICE, route.provider)
        if not key:
            raise ValueError(f"Please add a {route.provider} API key in desktop settings first.")
        field = {"openai": "openai_api_key", "deepseek": "deepseek_api_key", "claude": "claude_api_key"}[route.provider]
        settings = self.settings.model_copy(update={field: SecretStr(key)})
        return get_llm_client(route, settings)

    def key_status(self) -> dict[str, bool]:
        return {provider: bool(keyring.get_password(_KEYCHAIN_SERVICE, provider)) for provider in ("openai", "deepseek", "claude")}

    def save_key(self, provider: str, key: str) -> None:
        if provider not in {"openai", "deepseek", "claude"}:
            raise ValueError("Unknown model provider.")
        keyring.set_password(_KEYCHAIN_SERVICE, provider, key)

    async def summarize(self, document_id: str) -> None:
        async with self.summary_limit:
            try:
                summary = await asyncio.to_thread(self.store.get_summary, document_id)
                if not summary:
                    return
                await asyncio.to_thread(self.store.update_summary, document_id, "summarizing", error_message=None)
                chunks = await asyncio.to_thread(self.store.document_chunks, document_id)
                route = self.route(summary["route_provider"], summary["route_model"])
                generator = BilingualSummaryGenerator(self.llm(route), batch_chars=8000)
                data = await generator.generate(
                    [
                        SummarySourceChunk(id=uuid.UUID(item["id"]), page_number=item["page_number"], text=item["text"])
                        for item in chunks
                    ],
                    local_verification=route.provider == "local",
                )
                await asyncio.to_thread(
                    self.store.update_summary, document_id, "completed",
                    summary_data=data.model_dump_json(), source_chunk_count=len(chunks),
                    completed_at=datetime.now(UTC).isoformat(),
                )
            except Exception as exc:
                await asyncio.to_thread(self.store.update_summary, document_id, "failed", error_message=str(exc)[:2000])

    async def produce_chat(self, stream_id: str, payload: ChatStreamRequest) -> None:
        try:
            question = payload.question.strip()
            embedding = (await self.embedder.embed_many([question]))[0]
            if payload.use_semantic_cache:
                rows = await asyncio.to_thread(self.store.cache_rows)
                hits = [row for row in rows if _cosine(embedding, row["embedding"]) > 0.95]
                if hits:
                    hit = max(hits, key=lambda row: _cosine(embedding, row["embedding"]))
                    await self.hub.append(stream_id, "thought", {"stage": "cache", "summary": "语义缓存命中。", "cache_hit": True})
                    for citation in hit["citations"]:
                        await self.hub.append(stream_id, "citation", citation)
                    await self._deltas(stream_id, hit["answer"], cache_hit=True)
                    return
            await self.hub.append(stream_id, "thought", {"stage": "planning", "summary": "正在检索本地文档。", "subqueries": [question], "cache_hit": False})
            route = self.route(payload.route.provider if payload.route else None, payload.route.model if payload.route else None)
            llm = self.llm(route)
            hypothetical = await llm.complete(build_hyde_messages(question))
            contexts = await self.retrieve(question, hypothetical_document=hypothetical, limit=8)
            if not contexts:
                await self._deltas(stream_id, f"检索到的参考资料中未包含关于 {question} 的确切信息。", cache_hit=False)
                return
            citations = [
                {
                    "chunk_id": str(item.chunk_id), "document_id": str(item.document_id),
                    "document_name": item.document_name, "page_number": item.page_number,
                    "excerpt": item.text, "rrf_score": item.rrf_score, "rerank_score": item.rerank_score,
                }
                for item in contexts
            ]
            await self.hub.append(stream_id, "thought", {"stage": "evidence", "summary": f"已筛选 {len(contexts)} 条页码级证据。", "cache_hit": False})
            for citation in citations:
                await self.hub.append(stream_id, "citation", citation)
            draft = await llm.complete(build_answer_messages(question, contexts))
            verified = await llm.complete(build_self_rag_verification_messages(question, draft, contexts))
            answer = enforce_grounded_citations(verified, contexts, question)
            if payload.use_semantic_cache:
                await asyncio.to_thread(
                    self.store.put_cache, cache_id=str(uuid.uuid4()), query=question,
                    embedding=embedding, answer=answer, citations=citations,
                    expires_at=(datetime.now(UTC) + timedelta(days=1)).isoformat(),
                )
            await self._deltas(stream_id, answer, cache_hit=False)
        except Exception as exc:
            await self.hub.append(stream_id, "error", {"code": "generation_failed", "message": str(exc), "retryable": False})
        finally:
            await self.hub.finish(stream_id)

    async def _deltas(self, stream_id: str, answer: str, *, cache_hit: bool) -> None:
        for start in range(0, len(answer), 32):
            await self.hub.append(stream_id, "delta", {"text": answer[start:start + 32], "cache_hit": cache_hit})
            await asyncio.sleep(0)

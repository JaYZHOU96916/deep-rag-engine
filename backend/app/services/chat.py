from __future__ import annotations

import asyncio
from collections.abc import Sequence
from uuid import uuid4

from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.db.session import AsyncSessionLocal
from app.schemas.chat import ChatStreamRequest
from app.schemas.retrieval import CitationResponse
from app.services.embeddings import get_embedding_provider
from app.services.llm import get_llm_client, resolve_route
from app.services.prompting import (
    build_answer_messages,
    build_hyde_messages,
    build_self_rag_verification_messages,
)
from app.services.retrieval import HybridRetriever, RetrievedChunk
from app.services.semantic_cache import SemanticCache
from app.services.sse import SSEReplayStore

import re

_CITATION_PATTERN = re.compile(r"\[Ref: ([0-9a-fA-F-]{36}), Page (\d+)\]")


class ChatStreamProducer:
    """Produces typed, replayable SSE events independently from an individual browser connection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def produce(self, stream_id: str, payload: ChatStreamRequest) -> None:
        redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        store = SSEReplayStore(redis, ttl_seconds=self.settings.sse_replay_ttl_seconds)
        try:
            await self._produce_events(stream_id, payload, store, redis)
        except Exception as exc:
            await store.append(
                stream_id,
                "error",
                {"code": "generation_failed", "message": str(exc), "retryable": False},
            )
        finally:
            await store.mark_done(stream_id)
            await redis.aclose()

    async def _produce_events(
        self,
        stream_id: str,
        payload: ChatStreamRequest,
        store: SSEReplayStore,
        redis: Redis,
    ) -> None:
        question = payload.question.strip()
        embedding = (await get_embedding_provider().embed_many([question]))[0]
        cache = SemanticCache(
            redis,
            threshold=self.settings.semantic_cache_threshold,
            ttl_seconds=self.settings.semantic_cache_ttl_seconds,
        )
        if payload.use_semantic_cache and (cached := await cache.find(embedding)) is not None:
            await store.append(
                stream_id,
                "thought",
                {"stage": "cache", "summary": "语义缓存命中，正在复用已验证答案。", "cache_hit": True},
            )
            for citation in cached.citations:
                await store.append(stream_id, "citation", citation)
            await self._emit_deltas(store, stream_id, cached.answer, cache_hit=True)
            return

        await store.append(
            stream_id,
            "thought",
            {"stage": "planning", "summary": "正在拆解问题并检索相关文档片段。", "subqueries": [question], "cache_hit": False},
        )
        route = resolve_route(
            payload.route.provider if payload.route else None,
            payload.route.model if payload.route else None,
            self.settings,
        )
        llm = get_llm_client(route, self.settings)
        await store.append(
            stream_id,
            "thought",
            {"stage": "hyde", "summary": "正在生成假设性学术检索扩展。", "cache_hit": False},
        )
        hypothetical_document = await llm.complete(build_hyde_messages(question))
        async with AsyncSessionLocal() as session:
            contexts = await HybridRetriever(session).retrieve(
                question,
                hypothetical_document=hypothetical_document,
                limit=8,
            )
        if not contexts:
            await self._emit_deltas(
                store,
                stream_id,
                f"检索到的参考资料中未包含关于 {question} 的确切信息。",
                cache_hit=False,
            )
            return

        citations = [_citation_payload(context) for context in contexts]
        await store.append(
            stream_id,
            "thought",
            {"stage": "evidence", "summary": f"已筛选 {len(citations)} 条页码级证据并开始生成答案。", "cache_hit": False},
        )
        for citation in citations:
            await store.append(stream_id, "citation", citation)

        draft = await llm.complete(build_answer_messages(question, contexts))
        verification_messages = build_self_rag_verification_messages(question, draft, contexts)
        verified_parts: list[str] = []
        async for token in llm.stream(verification_messages):
            verified_parts.append(token)
        verified = "".join(verified_parts)
        answer = enforce_grounded_citations(verified, contexts, question)
        if payload.use_semantic_cache:
            await cache.put(query=question, embedding=embedding, answer=answer, citations=citations)
        await self._emit_deltas(store, stream_id, answer, cache_hit=False)

    async def _emit_deltas(self, store: SSEReplayStore, stream_id: str, answer: str, *, cache_hit: bool) -> None:
        for index in range(0, len(answer), 32):
            await store.append(stream_id, "delta", {"text": answer[index : index + 32], "cache_hit": cache_hit})
            await asyncio.sleep(0)


def enforce_grounded_citations(answer: str, contexts: Sequence[RetrievedChunk], question: str) -> str:
    allowed = {(str(item.document_id), item.page_number) for item in contexts}
    accepted: list[str] = []
    for clause in re.split(r"(?<=\])(?=\s|$)", answer):
        stripped = clause.strip()
        if not stripped:
            continue
        citations = _CITATION_PATTERN.findall(stripped)
        if citations and all((document_id, int(page)) in allowed for document_id, page in citations):
            accepted.append(stripped)
    return " ".join(accepted) or f"检索到的参考资料中未包含关于 {question} 的确切信息。"


def _citation_payload(context: RetrievedChunk) -> dict:
    return CitationResponse(
        chunk_id=context.chunk_id,
        document_id=context.document_id,
        document_name=context.document_name,
        page_number=context.page_number,
        excerpt=context.text,
        rrf_score=context.rrf_score,
        rerank_score=context.rerank_score,
    ).model_dump(mode="json")


def new_stream_id() -> str:
    return str(uuid4())

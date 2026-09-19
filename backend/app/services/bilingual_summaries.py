"""Evidence-preserving, optional bilingual document summaries.

The model never receives authority to create provenance: every returned citation is
validated against an ingested chunk id and its original PDF page before persistence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from app.schemas.summaries import BilingualClaim, BilingualSummaryData, SummaryCitation
from app.services.llm import AsyncLLMClient


@dataclass(frozen=True, slots=True)
class SummarySourceChunk:
    id: UUID
    page_number: int
    text: str


class BilingualSummaryGenerator:
    """Maps long documents into evidence notes, then synthesizes cited bilingual output."""

    def __init__(self, llm: AsyncLLMClient, *, batch_chars: int = 8_000) -> None:
        self.llm = llm
        self.batch_chars = batch_chars

    async def generate(
        self,
        chunks: Sequence[SummarySourceChunk],
        *,
        local_verification: bool,
    ) -> BilingualSummaryData:
        if not chunks:
            raise ValueError("The completed document has no extracted chunks to summarize.")
        if local_verification:
            return self._local_verification(chunks)

        evidence_notes: list[str] = []
        for batch in _chunk_batches(chunks, self.batch_chars):
            evidence_notes.append(await self.llm.complete(_build_evidence_note_messages(batch)))

        raw_summary = await self.llm.complete(_build_synthesis_messages(evidence_notes))
        candidate = _parse_summary_json(raw_summary)
        return _validate_and_filter(candidate, chunks)

    def _local_verification(self, chunks: Sequence[SummarySourceChunk]) -> BilingualSummaryData:
        evidence = list(chunks[:3])
        citations = [_citation_for(chunk) for chunk in evidence]
        excerpts = "\n\n".join(
            f"[p. {chunk.page_number}] {chunk.text.strip()[:700]}" for chunk in evidence
        )
        claims = [
            BilingualClaim(
                english=f"Source excerpt: {chunk.text.strip()[:500]}",
                chinese="本地验证模式未调用翻译模型，因此仅保留原文证据，不生成中文译文。",
                citations=[_citation_for(chunk)],
            )
            for chunk in evidence
            if chunk.text.strip()
        ]
        return BilingualSummaryData(
            english_summary=(
                "Local verification mode has not called an LLM or translation service. "
                "The following source excerpts are shown only to verify page-level evidence:\n\n"
                f"{excerpts}"
            ),
            chinese_summary=(
                "本地验证模式没有调用大模型或翻译服务，因此不会生成可能失真的中文总结。"
                "请选择已配置的 OpenAI、DeepSeek 或 Claude，再主动生成双语总结。"
            ),
            overview_citations=citations,
            key_findings=claims,
            methods=[],
            limitations=[],
            translation_mode="local_verification",
        )


def _chunk_batches(chunks: Sequence[SummarySourceChunk], limit: int) -> list[list[SummarySourceChunk]]:
    batches: list[list[SummarySourceChunk]] = []
    current: list[SummarySourceChunk] = []
    current_size = 0
    for chunk in chunks:
        size = len(chunk.text) + 90
        if current and current_size + size > limit:
            batches.append(current)
            current, current_size = [], 0
        current.append(chunk)
        current_size += size
    if current:
        batches.append(current)
    return batches


def _build_evidence_note_messages(chunks: Sequence[SummarySourceChunk]) -> list[dict[str, str]]:
    references = "\n\n".join(
        f"[Chunk: {chunk.id}; Page: {chunk.page_number}]\n{chunk.text}" for chunk in chunks
    )
    return [
        {
            "role": "system",
            "content": (
                "Extract compact academic evidence notes only from the supplied chunks. "
                "Do not translate, speculate, or add facts. Each note must retain one or more exact "
                "[Chunk: UUID; Page: N] identifiers from its supporting source."
            ),
        },
        {"role": "user", "content": f"Source chunks:\n{references}"},
    ]


def _build_synthesis_messages(evidence_notes: Sequence[str]) -> list[dict[str, str]]:
    schema = {
        "english_summary": "concise evidence-grounded English overview",
        "chinese_summary": "faithful Chinese translation of the overview",
        "overview_citations": [{"chunk_id": "source UUID", "page_number": 1}],
        "key_findings": [{"english": "finding", "chinese": "faithful Chinese translation", "citations": [{"chunk_id": "source UUID", "page_number": 1}]}],
        "methods": [],
        "limitations": [],
        "translation_mode": "llm_bilingual",
    }
    return [
        {
            "role": "system",
            "content": (
                "Create an academic bilingual summary only from the evidence notes. Return JSON only, matching "
                "the requested schema. Every overview and every claim must use source UUID/page citations found "
                "in the notes. English and Chinese must convey the same meaning. Omit anything without evidence."
            ),
        },
        {
            "role": "user",
            "content": f"Required JSON schema example:\n{json.dumps(schema)}\n\nEvidence notes:\n" + "\n\n".join(evidence_notes),
        },
    ]


def _parse_summary_json(raw: str) -> BilingualSummaryData:
    candidate = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("The selected model did not return a valid bilingual-summary JSON payload.") from exc
    if isinstance(payload, dict):
        payload["translation_mode"] = "llm_bilingual"
    return BilingualSummaryData.model_validate(payload)


def _validate_and_filter(
    summary: BilingualSummaryData,
    chunks: Sequence[SummarySourceChunk],
) -> BilingualSummaryData:
    allowed = {str(chunk.id): chunk.page_number for chunk in chunks}

    def valid_citations(citations: Sequence[SummaryCitation]) -> list[SummaryCitation]:
        deduplicated: dict[tuple[UUID, int], SummaryCitation] = {}
        for citation in citations:
            if allowed.get(str(citation.chunk_id)) == citation.page_number:
                deduplicated[(citation.chunk_id, citation.page_number)] = citation
        return list(deduplicated.values())

    overview = valid_citations(summary.overview_citations)
    if not overview:
        raise ValueError("The selected model returned no valid page-level evidence for its overview.")

    def valid_claims(claims: Sequence[BilingualClaim]) -> list[BilingualClaim]:
        verified: list[BilingualClaim] = []
        for claim in claims:
            citations = valid_citations(claim.citations)
            if citations:
                verified.append(claim.model_copy(update={"citations": citations}))
        return verified

    return summary.model_copy(
        update={
            "overview_citations": overview,
            "key_findings": valid_claims(summary.key_findings),
            "methods": valid_claims(summary.methods),
            "limitations": valid_claims(summary.limitations),
            "translation_mode": "llm_bilingual",
        }
    )


def _citation_for(chunk: SummarySourceChunk) -> SummaryCitation:
    return SummaryCitation(chunk_id=chunk.id, page_number=chunk.page_number)

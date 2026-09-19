import json
from uuid import UUID

import pytest

from app.services.bilingual_summaries import BilingualSummaryGenerator, SummarySourceChunk


class FakeSummaryLLM:
    async def complete(self, messages: list[dict[str, str]]) -> str:
        if "Extract compact academic evidence notes" in messages[0]["content"]:
            return "[Chunk: 00000000-0000-0000-0000-000000000001; Page: 2] Evidence note."
        return json.dumps(
            {
                "english_summary": "The paper reports evidence-backed retrieval.",
                "chinese_summary": "该论文报告了有证据支撑的检索。",
                "overview_citations": [
                    {"chunk_id": "00000000-0000-0000-0000-000000000001", "page_number": 2},
                    {"chunk_id": "00000000-0000-0000-0000-000000000099", "page_number": 9},
                ],
                "key_findings": [
                    {
                        "english": "A grounded finding.",
                        "chinese": "一项有依据的发现。",
                        "citations": [{"chunk_id": "00000000-0000-0000-0000-000000000001", "page_number": 2}],
                    },
                    {
                        "english": "An unsupported finding.",
                        "chinese": "一项无依据的发现。",
                        "citations": [{"chunk_id": "00000000-0000-0000-0000-000000000099", "page_number": 9}],
                    },
                ],
                "methods": [],
                "limitations": [],
                "translation_mode": "llm_bilingual",
            }
        )


def source() -> SummarySourceChunk:
    return SummarySourceChunk(
        id=UUID(int=1),
        page_number=2,
        text="The method retains page-level evidence for retrieval.",
    )


@pytest.mark.asyncio
async def test_bilingual_generator_filters_fabricated_chunk_citations() -> None:
    summary = await BilingualSummaryGenerator(FakeSummaryLLM()).generate([source()], local_verification=False)

    assert summary.translation_mode == "llm_bilingual"
    assert summary.overview_citations == [summary.overview_citations[0]]
    assert summary.overview_citations[0].chunk_id == UUID(int=1)
    assert [claim.english for claim in summary.key_findings] == ["A grounded finding."]


@pytest.mark.asyncio
async def test_local_verification_mode_never_claims_to_translate() -> None:
    summary = await BilingualSummaryGenerator(FakeSummaryLLM()).generate([source()], local_verification=True)

    assert summary.translation_mode == "local_verification"
    assert "not called an LLM or translation service" in summary.english_summary
    assert "不会生成" in summary.chinese_summary
    assert summary.key_findings[0].citations[0].page_number == 2

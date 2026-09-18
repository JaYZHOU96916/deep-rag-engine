from uuid import UUID

from app.services.chat import enforce_grounded_citations
from app.services.retrieval import RetrievedChunk


def context() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=1),
        document_id=UUID(int=2),
        document_name="paper.pdf",
        page_number=3,
        text="Evidence",
    )


def test_grounding_guard_keeps_only_valid_page_citations() -> None:
    valid = "Supported fact. [Ref: 00000000-0000-0000-0000-000000000002, Page 3]"
    invalid = " Unsupported fact. [Ref: 00000000-0000-0000-0000-000000000099, Page 9]"

    answer = enforce_grounded_citations(valid + invalid, [context()], "test question")

    assert "Supported fact" in answer
    assert "Unsupported fact" not in answer


def test_grounding_guard_returns_insufficiency_when_no_citation_is_valid() -> None:
    answer = enforce_grounded_citations("Uncited claim.", [context()], "test question")

    assert "检索到的参考资料中未包含关于 test question 的确切信息。" == answer

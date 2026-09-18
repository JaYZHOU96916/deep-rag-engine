from app.services.prompting import build_answer_messages, build_hyde_messages
from app.services.retrieval import RetrievedChunk
from uuid import UUID


def sample_context() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=1),
        document_id=UUID(int=2),
        document_name="paper.pdf",
        page_number=7,
        text="Evidence text.",
    )


def test_answer_prompt_requires_source_grounded_page_citations() -> None:
    prompt = build_answer_messages("Question?", [sample_context()])

    assert "Every factual or inferential sentence" in prompt[0]["content"]
    assert "[Ref: 00000000-0000-0000-0000-000000000002, Page 7]" in prompt[1]["content"]


def test_hyde_prompt_marks_its_output_as_hypothetical() -> None:
    prompt = build_hyde_messages("Question?")

    assert "hypothetical" in prompt[0]["content"].lower()

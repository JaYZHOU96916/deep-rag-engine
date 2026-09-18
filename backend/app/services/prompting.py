from __future__ import annotations

from app.services.retrieval import RetrievedChunk


def build_hyde_messages(question: str) -> list[dict[str, str]]:
    """Request a hypothetical academic passage for query expansion, not a user-facing answer."""
    return [
        {
            "role": "system",
            "content": "You generate a compact hypothetical academic passage solely for retrieval expansion. Do not claim it is sourced or factual.",
        },
        {"role": "user", "content": f"Question: {question}\nWrite a 120-word hypothetical passage likely to appear in a relevant paper."},
    ]


def build_answer_messages(question: str, contexts: list[RetrievedChunk]) -> list[dict[str, str]]:
    references = "\n\n".join(
        f"[Ref: {chunk.document_id}, Page {chunk.page_number}]\n{chunk.text}" for chunk in contexts
    )
    return [
        {
            "role": "system",
            "content": (
                "Answer only from the supplied references. Every factual or inferential sentence must end with an exact "
                "citation in the form [Ref: ID, Page X]. If evidence is insufficient, state exactly: "
                "检索到的参考资料中未包含关于 [具体问题] 的确切信息。 Do not invent citations, facts, or page numbers."
            ),
        },
        {"role": "user", "content": f"Question:\n{question}\n\nReferences:\n{references}"},
    ]


def build_self_rag_verification_messages(
    question: str,
    draft_answer: str,
    contexts: list[RetrievedChunk],
) -> list[dict[str, str]]:
    evidence = "\n".join(
        f"[Ref: {chunk.document_id}, Page {chunk.page_number}] {chunk.text}" for chunk in contexts
    )
    return [
        {
            "role": "system",
            "content": (
                "Perform a Self-RAG factual-consistency audit. Return only a corrected final answer. "
                "Remove unsupported claims; ensure every remaining factual sentence has a matching [Ref: ID, Page X] citation."
            ),
        },
        {"role": "user", "content": f"Question: {question}\n\nDraft:\n{draft_answer}\n\nEvidence:\n{evidence}"},
    ]

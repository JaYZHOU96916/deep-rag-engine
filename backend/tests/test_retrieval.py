from uuid import UUID

from app.services.retrieval import RetrievedChunk, reciprocal_rank_fusion


def candidate(number: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=number),
        document_id=UUID(int=100 + number),
        document_name=f"paper-{number}.pdf",
        page_number=number,
        text=f"chunk {number}",
    )


def test_rrf_rewards_consensus_across_dense_and_sparse_lists() -> None:
    dense = [candidate(1), candidate(2), candidate(3)]
    sparse = [candidate(2), candidate(3), candidate(1)]

    fused = reciprocal_rank_fusion([dense, sparse], rrf_k=60)

    assert [item.chunk_id for item in fused] == [candidate(2).chunk_id, candidate(1).chunk_id, candidate(3).chunk_id]
    assert fused[0].rrf_score == (1 / 62) + (1 / 61)


def test_rrf_rejects_invalid_constant() -> None:
    try:
        reciprocal_rank_fusion([[candidate(1)]], rrf_k=0)
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("Expected rrf_k validation to fail")

from app.services.chunking import RecursiveTextSplitter


def test_recursive_splitter_keeps_chunks_bounded_with_overlap() -> None:
    splitter = RecursiveTextSplitter(chunk_size=40, chunk_overlap=8)
    chunks = splitter.split("First paragraph contains useful context.\n\nSecond paragraph is also relevant.")

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= 48 for chunk in chunks)
    assert chunks[0].char_start == 0
    assert all(chunk.char_end > chunk.char_start for chunk in chunks)


def test_recursive_splitter_handles_cjk_without_spaces() -> None:
    splitter = RecursiveTextSplitter(chunk_size=12, chunk_overlap=2)
    chunks = splitter.split("这是第一句。这里是第二句。这是第三句。")

    assert len(chunks) >= 2
    assert "第一句" in chunks[0].text

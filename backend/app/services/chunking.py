from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    char_start: int
    char_end: int
    token_estimate: int


class RecursiveTextSplitter:
    """Dependency-free recursive splitter with bounded chunks and deterministic overlap."""

    def __init__(self, chunk_size: int = 1_000, chunk_overlap: int = 180) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ("\n\n", "\n", "。", "！", "？", ". ", " ", "")

    def split(self, text: str) -> list[TextChunk]:
        normalised = text.strip()
        if not normalised:
            return []
        raw_chunks = self._split_recursive(normalised, self.separators)
        chunks: list[TextChunk] = []
        search_from = 0
        for raw_chunk in raw_chunks:
            chunk_text = raw_chunk.strip()
            if not chunk_text:
                continue
            start = normalised.find(chunk_text, max(0, search_from - self.chunk_overlap))
            if start < 0:
                start = search_from
            end = start + len(chunk_text)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                    token_estimate=max(1, len(chunk_text.split())),
                )
            )
            search_from = start
        return chunks

    def _split_recursive(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        separator = next((item for item in separators if item == "" or item in text), "")
        remaining = separators[separators.index(separator) + 1 :]
        pieces = list(text) if separator == "" else text.split(separator)

        expanded: list[str] = []
        for index, piece in enumerate(pieces):
            if not piece:
                continue
            suffix = separator if separator and index < len(pieces) - 1 else ""
            expanded.append(f"{piece}{suffix}")

        fragments: list[str] = []
        for piece in expanded:
            if len(piece) > self.chunk_size and remaining:
                fragments.extend(self._split_recursive(piece, remaining))
            else:
                fragments.append(piece)
        return self._merge_with_overlap(fragments)

    def _merge_with_overlap(self, fragments: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for fragment in fragments:
            if len(fragment) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(
                    fragment[index : index + self.chunk_size]
                    for index in range(0, len(fragment), self.chunk_size - self.chunk_overlap)
                )
                continue
            if current and len(current) + len(fragment) > self.chunk_size:
                chunks.append(current)
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = f"{overlap}{fragment}"
            else:
                current += fragment
        if current:
            chunks.append(current)
        return chunks

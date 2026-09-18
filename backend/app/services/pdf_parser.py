from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


class PDFExtractionError(ValueError):
    """Raised when a file cannot be safely parsed as a PDF."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int
    text: str
    paragraph_count: int


@dataclass(frozen=True, slots=True)
class ParsedPDF:
    pages: list[ExtractedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PDFParser:
    """Extract ordered text blocks while preserving one-based source page metadata."""

    @staticmethod
    def parse(file_path: Path) -> ParsedPDF:
        try:
            pdf = pymupdf.open(file_path)
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise PDFExtractionError("The uploaded file is not a readable PDF.") from exc

        try:
            if pdf.is_encrypted and not pdf.authenticate(""):
                raise PDFExtractionError("Encrypted PDFs are not supported.")

            pages: list[ExtractedPage] = []
            for page_index, page in enumerate(pdf, start=1):
                blocks = page.get_text("blocks", sort=True)
                paragraphs = [
                    _normalise_block(block[4])
                    for block in blocks
                    if len(block) >= 7 and block[6] == 0 and block[4].strip()
                ]
                paragraphs = [paragraph for paragraph in paragraphs if paragraph]
                pages.append(
                    ExtractedPage(
                        page_number=page_index,
                        text="\n\n".join(paragraphs),
                        paragraph_count=len(paragraphs),
                    )
                )
            return ParsedPDF(pages=pages)
        finally:
            pdf.close()


def _normalise_block(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

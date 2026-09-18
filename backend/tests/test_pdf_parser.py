from pathlib import Path

import fitz

from app.services.pdf_parser import PDFParser


def test_pdf_parser_preserves_one_based_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf = fitz.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "First page abstract.")
    second = pdf.new_page()
    second.insert_text((72, 72), "Second page methodology.")
    pdf.save(pdf_path)
    pdf.close()

    parsed = PDFParser.parse(pdf_path)

    assert parsed.page_count == 2
    assert parsed.pages[0].page_number == 1
    assert "abstract" in parsed.pages[0].text
    assert parsed.pages[1].page_number == 2
    assert "methodology" in parsed.pages[1].text

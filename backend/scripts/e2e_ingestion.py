"""Smoke-test the running API, Celery worker, Redis broker, and pgvector persistence."""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path

import pymupdf
import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def build_fixture(path: Path) -> None:
    pdf = pymupdf.open()
    for text in (
        "Deep retrieval augments academic document research with citations.",
        "A second page verifies page-level provenance is retained during ingestion.",
    ):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def main() -> int:
    fixture = Path("/tmp/deep-rag-ingestion-smoke.pdf")
    build_fixture(fixture)
    try:
        with httpx.Client(timeout=15) as client, fixture.open("rb") as source:
            response = client.post(
                f"{API_BASE_URL}/api/v1/documents",
                files={"file": (fixture.name, source, "application/pdf")},
            )
            response.raise_for_status()
            document = response.json()

        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            with httpx.Client(timeout=10) as client:
                status_response = client.get(f"{API_BASE_URL}/api/v1/documents/{document['id']}")
                status_response.raise_for_status()
                current = status_response.json()
            if current["status"] == "completed":
                assert current["page_count"] == 2
                assert current["chunk_count"] >= 2
                print(f"Ingestion complete: {current['chunk_count']} chunks across {current['page_count']} pages.")
                return 0
            if current["status"] == "failed":
                raise RuntimeError(current["error_message"])
            time.sleep(1)
        raise TimeoutError("Ingestion did not complete within 45 seconds.")
    finally:
        fixture.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())

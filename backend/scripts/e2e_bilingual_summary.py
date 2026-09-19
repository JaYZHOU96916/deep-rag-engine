"""Verify the opt-in bilingual-summary endpoint and Celery state machine."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
import pymupdf

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def build_fixture(path: Path) -> None:
    pdf = pymupdf.open()
    for text in (
        "This paper preserves page-level evidence during academic retrieval.",
        "The method uses a combined ranking process and reports its provenance.",
    ):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def wait_for_document(client: httpx.Client, document_id: str) -> dict:
    for _ in range(60):
        payload = client.get(f"/api/v1/documents/{document_id}").json()
        if payload["status"] == "completed":
            return payload
        if payload["status"] == "failed":
            raise RuntimeError(payload["error_message"])
        time.sleep(1)
    raise TimeoutError("Document ingestion did not complete.")


def main() -> int:
    fixture = Path("/tmp/deep-rag-bilingual-summary-smoke.pdf")
    build_fixture(fixture)
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=15) as client, fixture.open("rb") as source:
            upload = client.post("/api/v1/documents", files={"file": (fixture.name, source, "application/pdf")})
            upload.raise_for_status()
            document_id = upload.json()["id"]
            wait_for_document(client, document_id)

            absent = client.get(f"/api/v1/documents/{document_id}/bilingual-summary")
            assert absent.status_code == 404, "A summary must not be generated before an explicit request."

            requested = client.post(
                f"/api/v1/documents/{document_id}/bilingual-summary",
                json={"route": {"provider": "local"}},
            )
            requested.raise_for_status()
            for _ in range(60):
                summary = client.get(f"/api/v1/documents/{document_id}/bilingual-summary")
                summary.raise_for_status()
                payload = summary.json()
                if payload["status"] == "completed":
                    assert payload["summary"]["translation_mode"] == "local_verification"
                    assert payload["summary"]["overview_citations"]
                    print("Optional bilingual-summary state machine verified with local evidence-only mode.")
                    return 0
                if payload["status"] == "failed":
                    raise RuntimeError(payload["error_message"])
                time.sleep(1)
        raise TimeoutError("Bilingual summary did not complete.")
    finally:
        fixture.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())

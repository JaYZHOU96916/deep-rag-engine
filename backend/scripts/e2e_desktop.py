"""Exercise the packaged desktop API without Docker or a development Python server."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pymupdf


def _wait(client: httpx.Client, path: str, *, expected: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(path)
        response.raise_for_status()
        payload = response.json()
        if payload["status"] == expected:
            return payload
        if payload["status"] == "failed":
            raise RuntimeError(payload.get("error_message"))
        time.sleep(0.3)
    raise TimeoutError(f"Did not reach {expected}: {path}")


def main() -> int:
    base_url = os.environ.get("DESKTOP_TEST_API_BASE_URL", "http://127.0.0.1:8769")
    token = os.environ["DESKTOP_TEST_TOKEN"]
    with tempfile.TemporaryDirectory(prefix="deep-rag-desktop-smoke-") as directory:
        fixture = Path(directory) / "desktop-study.pdf"
        pdf = pymupdf.open()
        for text in (
            "Academic retrieval retains evidence with original PDF page numbers.",
            "Hybrid ranking combines dense and sparse results before citation checking.",
        ):
            page = pdf.new_page()
            page.insert_text((72, 72), text)
        pdf.save(fixture)
        pdf.close()

        with httpx.Client(base_url=base_url, timeout=15) as client:
            assert client.get("/healthz").json()["mode"] == "desktop"
            assert client.get("/api/v1/documents").status_code == 401
            assert "Deep-RAG" in client.get("/").text

        with httpx.Client(base_url=base_url, headers={"X-Desktop-Session": token}, timeout=15) as client:
            with fixture.open("rb") as source:
                upload = client.post("/api/v1/documents", files={"file": (fixture.name, source, "application/pdf")})
            upload.raise_for_status()
            document_id = upload.json()["id"]
            document = _wait(client, f"/api/v1/documents/{document_id}", expected="completed")
            assert document["page_count"] == 2
            assert document["chunk_count"] >= 2
            assert any(item["id"] == document_id for item in client.get("/api/v1/documents").json())

            retrieval = client.post("/api/v1/retrieval/search", json={"question": "How are PDF page numbers retained?"})
            retrieval.raise_for_status()
            assert retrieval.json()["citations"]

            with client.stream("POST", "/api/v1/chat/stream", json={"question": "How are PDF page numbers retained?"}) as chat:
                chat.raise_for_status()
                assert chat.headers.get("X-Stream-ID")
                stream_text = "\n".join(chat.iter_lines())
            assert "event: thought" in stream_text
            assert "event: citation" in stream_text
            assert "event: delta" in stream_text

            with client.stream("POST", "/api/v1/chat/stream", json={"question": "How are PDF page numbers retained?"}) as cached_chat:
                cached_chat.raise_for_status()
                cached_stream_id = cached_chat.headers["X-Stream-ID"]
                cached_text = "\n".join(cached_chat.iter_lines())
            assert '"cache_hit": true' in cached_text
            resumed = client.get(
                f"/api/v1/chat/stream/{cached_stream_id}", headers={"Last-Event-ID": "1"}
            )
            resumed.raise_for_status()
            assert "id: 1\n" not in resumed.text
            assert "event: delta" in resumed.text

            summary_path = f"/api/v1/documents/{document_id}/bilingual-summary"
            assert client.get(summary_path).status_code == 404
            requested = client.post(summary_path, json={"route": {"provider": "local"}})
            requested.raise_for_status()
            summary = _wait(client, summary_path, expected="completed")
            assert summary["summary"]["translation_mode"] == "local_verification"
            assert summary["summary"]["overview_citations"]

        print(f"Desktop API verified: {document['page_count']} pages, retrieval, typed SSE, opt-in summary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

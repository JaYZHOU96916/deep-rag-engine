"""Smoke-test public paper search and the configured University of Melbourne handoff."""

from __future__ import annotations

import asyncio
import os

import httpx


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


async def main() -> None:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=20) as client:
        institutions_response = await client.get("/api/v1/papers/institutions")
        institutions_response.raise_for_status()
        institutions = institutions_response.json()
        assert any(item["id"] == "unimelb" for item in institutions), "University of Melbourne connector missing"

        doi_response = await client.post(
            "/api/v1/papers/search",
            json={
                "query": "10.1038/s41586-023-06221-2",
                "limit": 3,
                "institution_id": "unimelb",
            },
        )
        doi_response.raise_for_status()
        doi_payload = doi_response.json()
        assert doi_payload["is_doi_lookup"] is True
        assert doi_payload["results"]
        assert doi_payload["results"][0]["institution_access_url"]

        keyword_response = await client.post(
            "/api/v1/papers/search",
            json={"query": "retrieval augmented generation", "limit": 3},
        )
        keyword_response.raise_for_status()
        keyword_payload = keyword_response.json()
        assert keyword_payload["results"]
        assert any(item["open_access_url"] for item in keyword_payload["results"])

    print(
        "Paper discovery complete: DOI lookup, University of Melbourne handoff, "
        "and open-access keyword results verified."
    )


if __name__ == "__main__":
    asyncio.run(main())

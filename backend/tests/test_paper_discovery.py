import json

import httpx
import pytest

from app.core.config import Settings
from app.services.paper_discovery import PaperDiscoveryService, UnknownInstitution, normalize_doi


def settings() -> Settings:
    return Settings(
        paper_search_max_results=10,
        institution_connectors_json=json.dumps(
            [
                {
                    "id": "example-u",
                    "name": "Example University",
                    "catalog_search_url_template": "https://library.example.edu/search?q={query}",
                }
            ]
        ),
    )


@pytest.mark.asyncio
async def test_search_merges_public_metadata_and_adds_a_safe_library_link() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openalex.org":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Grounded Retrieval for Academic Research",
                            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                            "publication_year": 2025,
                            "doi": "https://doi.org/10.1000/deep-rag",
                            "open_access": {"is_oa": True},
                            "best_oa_location": {"pdf_url": "https://repository.example.edu/paper.pdf"},
                            "primary_location": {"landing_page_url": "https://doi.org/10.1000/deep-rag"},
                            "abstract_inverted_index": {"Grounded": [0], "retrieval": [1]},
                            "cited_by_count": 42,
                        }
                    ]
                },
            )
        if request.url.host == "api.crossref.org":
            return httpx.Response(
                200,
                json={
                    "message": {
                        "items": [
                            {
                                "title": ["Grounded Retrieval for Academic Research"],
                                "DOI": "10.1000/deep-rag",
                                "abstract": "<jats:p>Verified abstract.</jats:p>",
                                "author": [{"given": "Ada", "family": "Lovelace"}],
                                "issued": {"date-parts": [[2025]]},
                                "URL": "https://doi.org/10.1000/deep-rag",
                            }
                        ]
                    }
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PaperDiscoveryService(settings(), client=client).search(
            "grounded retrieval",
            limit=6,
            institution_id="example-u",
        )

    assert response.is_doi_lookup is False
    assert len(response.results) == 1
    paper = response.results[0]
    assert paper.open_access_url == "https://repository.example.edu/paper.pdf"
    assert paper.open_access_pdf_url == "https://repository.example.edu/paper.pdf"
    assert paper.access_type == "open_access"
    assert paper.sources == ["Crossref", "OpenAlex"]
    assert paper.institution_name == "Example University"
    assert paper.institution_access_url == "https://library.example.edu/search?q=10.1000%2Fdeep-rag"


@pytest.mark.asyncio
async def test_doi_lookup_uses_public_provider_fallback_when_one_source_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openalex.org":
            return httpx.Response(503, json={"error": "temporary"})
        if request.url.host == "api.crossref.org":
            return httpx.Response(
                200,
                json={
                    "message": {
                        "title": ["A DOI-resolved paper"],
                        "DOI": "10.5555/lookup",
                        "author": [{"family": "Researcher"}],
                        "published-online": {"date-parts": [[2024]]},
                        "URL": "https://doi.org/10.5555/lookup",
                    }
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await PaperDiscoveryService(settings(), client=client).search(
            "https://doi.org/10.5555/lookup",
            limit=4,
        )

    assert response.is_doi_lookup is True
    assert response.results[0].doi == "10.5555/lookup"
    assert response.results[0].sources == ["Crossref"]
    assert response.results[0].access_type == "metadata_only"


@pytest.mark.asyncio
async def test_unknown_institution_is_rejected_before_external_requests() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: pytest.fail(str(request.url)))) as client:
        with pytest.raises(UnknownInstitution):
            await PaperDiscoveryService(settings(), client=client).search("papers", limit=4, institution_id="missing")


def test_normalize_doi_accepts_a_resolver_url_and_rejects_regular_text() -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC_def.") == "10.1000/abc_def"
    assert normalize_doi("retrieval augmented generation") is None

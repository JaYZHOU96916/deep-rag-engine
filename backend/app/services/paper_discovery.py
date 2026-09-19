"""Public scholarly metadata discovery with optional, user-mediated library access."""

from __future__ import annotations

import asyncio
import html
import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode, urlparse

import httpx

from app.core.config import Settings, get_settings
from app.schemas.papers import InstitutionResponse, PaperResult, PaperSearchResponse

_DOI_PATTERN = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:a-z0-9]+)$", re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_CONNECTOR_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class PaperDiscoveryUnavailable(RuntimeError):
    """Every selected public metadata source failed for the current request."""


class UnknownInstitution(ValueError):
    """The caller selected an institution that is not configured server-side."""


class InstitutionConfigurationError(ValueError):
    """The administrator supplied an unsafe or malformed institution connector."""


@dataclass(frozen=True, slots=True)
class InstitutionConnector:
    id: str
    name: str
    catalog_search_url_template: str
    openurl_base_url: str | None = None


@dataclass(slots=True)
class _PaperCandidate:
    title: str
    authors: list[str] = field(default_factory=list)
    publication_year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    landing_page_url: str | None = None
    open_access_url: str | None = None
    open_access_pdf_url: str | None = None
    citation_count: int | None = None
    sources: set[str] = field(default_factory=set)


class PaperDiscoveryService:
    """Queries public metadata APIs; it never authenticates against institutional systems."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client

    def institutions(self) -> list[InstitutionResponse]:
        return [
            InstitutionResponse(id=connector.id, name=connector.name)
            for connector in self._connectors().values()
        ]

    async def search(
        self,
        query: str,
        *,
        limit: int,
        institution_id: str | None = None,
    ) -> PaperSearchResponse:
        normalized_query = query.strip()
        doi = normalize_doi(normalized_query)
        institution = self._institution_for(institution_id)
        effective_limit = min(limit, self.settings.paper_search_max_results)

        if self._client is not None:
            candidates = await self._search_with_client(self._client, normalized_query, doi, effective_limit)
        else:
            async with httpx.AsyncClient(
                timeout=self.settings.paper_search_timeout_seconds,
                follow_redirects=True,
                headers=self._request_headers(),
            ) as client:
                candidates = await self._search_with_client(client, normalized_query, doi, effective_limit)

        return PaperSearchResponse(
            query=normalized_query,
            is_doi_lookup=doi is not None,
            results=[self._to_result(candidate, institution, normalized_query) for candidate in candidates[:effective_limit]],
        )

    async def _search_with_client(
        self,
        client: httpx.AsyncClient,
        query: str,
        doi: str | None,
        limit: int,
    ) -> list[_PaperCandidate]:
        requests = (
            (self._openalex_by_doi(client, doi), self._crossref_by_doi(client, doi))
            if doi
            else (self._openalex_search(client, query, limit), self._crossref_search(client, query, limit))
        )
        settled = await asyncio.gather(*requests, return_exceptions=True)
        successful = [result for result in settled if isinstance(result, list)]
        if not successful and all(isinstance(result, Exception) for result in settled):
            raise PaperDiscoveryUnavailable("Public scholarly metadata providers are temporarily unavailable.")
        return _merge_candidates(candidate for result in successful for candidate in result)

    async def _openalex_by_doi(self, client: httpx.AsyncClient, doi: str) -> list[_PaperCandidate]:
        payload = await self._get_json(
            client,
            f"https://api.openalex.org/works/{quote(f'https://doi.org/{doi}', safe='')}",
        )
        return [_candidate_from_openalex(payload)] if payload else []

    async def _crossref_by_doi(self, client: httpx.AsyncClient, doi: str) -> list[_PaperCandidate]:
        payload = await self._get_json(client, f"https://api.crossref.org/works/{quote(doi, safe='')}")
        message = payload.get("message") if payload else None
        return [_candidate_from_crossref(message)] if isinstance(message, dict) else []

    async def _openalex_search(self, client: httpx.AsyncClient, query: str, limit: int) -> list[_PaperCandidate]:
        params: dict[str, str | int] = {"search": query, "per-page": limit}
        api_key = self.settings.openalex_api_key
        if api_key:
            params["api_key"] = api_key.get_secret_value()
        payload = await self._get_json(client, "https://api.openalex.org/works", params=params)
        rows = payload.get("results", []) if payload else []
        return [_candidate_from_openalex(row) for row in rows if isinstance(row, dict)]

    async def _crossref_search(self, client: httpx.AsyncClient, query: str, limit: int) -> list[_PaperCandidate]:
        payload = await self._get_json(
            client,
            "https://api.crossref.org/works",
            params={"query.bibliographic": query, "rows": limit},
        )
        message = payload.get("message", {}) if payload else {}
        rows = message.get("items", []) if isinstance(message, dict) else []
        return [_candidate_from_crossref(row) for row in rows if isinstance(row, dict)]

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any] | None:
        try:
            response = await client.get(url, params=params)
        except httpx.HTTPError as error:
            raise PaperDiscoveryUnavailable(str(error)) from error
        if response.status_code == 404:
            return None
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise PaperDiscoveryUnavailable(f"Invalid response from {response.url.host}.") from error
        return payload if isinstance(payload, dict) else None

    def _request_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "Deep-RAG/0.1 (+https://github.com/JaYZHOU96916/deep-rag-engine)"}
        if self.settings.crossref_mailto:
            headers["mailto"] = self.settings.crossref_mailto
        return headers

    def _connectors(self) -> dict[str, InstitutionConnector]:
        try:
            raw_connectors = json.loads(self.settings.institution_connectors_json)
        except json.JSONDecodeError as error:
            raise InstitutionConfigurationError("INSTITUTION_CONNECTORS_JSON must be valid JSON.") from error
        if not isinstance(raw_connectors, list):
            raise InstitutionConfigurationError("INSTITUTION_CONNECTORS_JSON must be a JSON array.")

        connectors: dict[str, InstitutionConnector] = {}
        for raw in raw_connectors:
            if not isinstance(raw, dict):
                raise InstitutionConfigurationError("Every institution connector must be an object.")
            connector = _connector_from_mapping(raw)
            if connector.id in connectors:
                raise InstitutionConfigurationError(f"Duplicate institution connector: {connector.id}")
            connectors[connector.id] = connector
        return connectors

    def _institution_for(self, institution_id: str | None) -> InstitutionConnector | None:
        if institution_id is None:
            return None
        connector = self._connectors().get(institution_id)
        if connector is None:
            raise UnknownInstitution(institution_id)
        return connector

    @staticmethod
    def _to_result(
        candidate: _PaperCandidate,
        institution: InstitutionConnector | None,
        original_query: str,
    ) -> PaperResult:
        institution_access_url = _build_institution_access_url(institution, candidate, original_query)
        if candidate.open_access_url:
            access_type = "open_access"
        elif institution_access_url:
            access_type = "institution_login"
        else:
            access_type = "metadata_only"
        return PaperResult(
            title=candidate.title,
            authors=candidate.authors,
            publication_year=candidate.publication_year,
            abstract=candidate.abstract,
            doi=candidate.doi,
            landing_page_url=candidate.landing_page_url,
            open_access_url=candidate.open_access_url,
            open_access_pdf_url=candidate.open_access_pdf_url,
            citation_count=candidate.citation_count,
            sources=sorted(candidate.sources),
            access_type=access_type,
            institution_name=institution.name if institution_access_url else None,
            institution_access_url=institution_access_url,
        )


def normalize_doi(value: str) -> str | None:
    match = _DOI_PATTERN.fullmatch(value.strip().rstrip(".,;"))
    return match.group(1).lower() if match else None


def _connector_from_mapping(raw: dict[str, Any]) -> InstitutionConnector:
    connector_id = str(raw.get("id", ""))
    name = str(raw.get("name", ""))
    search_template = str(raw.get("catalog_search_url_template", ""))
    openurl_base_url = raw.get("openurl_base_url")
    if not _CONNECTOR_ID_PATTERN.fullmatch(connector_id) or not name:
        raise InstitutionConfigurationError("Institution connector requires a safe id and a name.")
    if "{query}" not in search_template or not _is_safe_https_url(search_template.replace("{query}", "query")):
        raise InstitutionConfigurationError("catalog_search_url_template must be an HTTPS URL containing {query}.")
    if openurl_base_url is not None and not _is_safe_https_url(str(openurl_base_url)):
        raise InstitutionConfigurationError("openurl_base_url must be an HTTPS URL.")
    return InstitutionConnector(
        id=connector_id,
        name=name,
        catalog_search_url_template=search_template,
        openurl_base_url=str(openurl_base_url) if openurl_base_url else None,
    )


def _is_safe_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _candidate_from_openalex(raw: dict[str, Any]) -> _PaperCandidate:
    best_location = raw.get("best_oa_location") if isinstance(raw.get("best_oa_location"), dict) else {}
    primary_location = raw.get("primary_location") if isinstance(raw.get("primary_location"), dict) else {}
    is_open_access = bool((raw.get("open_access") or {}).get("is_oa"))
    open_access_pdf_url = _clean_text(best_location.get("pdf_url"))
    open_access_url = None
    if is_open_access:
        open_access_url = open_access_pdf_url or best_location.get("landing_page_url")
        open_access_url = open_access_url or primary_location.get("landing_page_url")
    doi = normalize_doi(str(raw.get("doi") or ""))
    return _PaperCandidate(
        title=_clean_text(raw.get("title")) or "Untitled work",
        authors=[
            name
            for authorship in raw.get("authorships", [])
            if isinstance(authorship, dict)
            if (name := _clean_text((authorship.get("author") or {}).get("display_name")))
        ],
        publication_year=_as_year(raw.get("publication_year")),
        abstract=_abstract_from_openalex(raw.get("abstract_inverted_index")),
        doi=doi,
        landing_page_url=_clean_text((primary_location or {}).get("landing_page_url")),
        open_access_url=_clean_text(open_access_url),
        open_access_pdf_url=open_access_pdf_url,
        citation_count=_as_int(raw.get("cited_by_count")),
        sources={"OpenAlex"},
    )


def _candidate_from_crossref(raw: dict[str, Any]) -> _PaperCandidate:
    author_names = []
    for author in raw.get("author", []):
        if not isinstance(author, dict):
            continue
        name = " ".join(part for part in (author.get("given"), author.get("family")) if part)
        if name:
            author_names.append(name)
    return _PaperCandidate(
        title=_first_text(raw.get("title")) or "Untitled work",
        authors=author_names,
        publication_year=_crossref_year(raw),
        abstract=_clean_text(raw.get("abstract")),
        doi=normalize_doi(str(raw.get("DOI") or "")),
        landing_page_url=_clean_text(raw.get("URL")),
        citation_count=_as_int(raw.get("is-referenced-by-count")),
        sources={"Crossref"},
    )


def _merge_candidates(candidates: Any) -> list[_PaperCandidate]:
    merged: dict[str, _PaperCandidate] = {}
    for candidate in candidates:
        key = candidate.doi.lower() if candidate.doi else f"{candidate.title.lower()}:{candidate.publication_year or ''}"
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue
        existing.authors = existing.authors or candidate.authors
        existing.publication_year = existing.publication_year or candidate.publication_year
        existing.abstract = existing.abstract or candidate.abstract
        existing.landing_page_url = existing.landing_page_url or candidate.landing_page_url
        existing.open_access_url = existing.open_access_url or candidate.open_access_url
        existing.open_access_pdf_url = existing.open_access_pdf_url or candidate.open_access_pdf_url
        existing.citation_count = existing.citation_count or candidate.citation_count
        existing.sources.update(candidate.sources)
    return sorted(merged.values(), key=lambda item: (-int(item.citation_count or 0), item.title.lower()))


def _build_institution_access_url(
    institution: InstitutionConnector | None,
    candidate: _PaperCandidate,
    original_query: str,
) -> str | None:
    if institution is None:
        return None
    if institution.openurl_base_url:
        params = {"url_ver": "Z39.88-2004", "rft_val_fmt": "info:ofi/fmt:kev:mtx:journal", "rft.atitle": candidate.title}
        if candidate.doi:
            params["rft_id"] = f"info:doi/{candidate.doi}"
        separator = "&" if "?" in institution.openurl_base_url else "?"
        return f"{institution.openurl_base_url}{separator}{urlencode(params)}"
    search_term = candidate.doi or candidate.title or original_query
    return institution.catalog_search_url_template.replace("{query}", quote(search_term, safe=""))


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(_HTML_TAG_PATTERN.sub(" ", html.unescape(value)).split())
    return text or None


def _first_text(value: Any) -> str | None:
    if isinstance(value, list):
        return _clean_text(value[0]) if value else None
    return _clean_text(value)


def _crossref_year(raw: dict[str, Any]) -> int | None:
    for field_name in ("published-print", "published-online", "issued", "created"):
        parts = (raw.get(field_name) or {}).get("date-parts", [])
        if parts and isinstance(parts[0], list):
            return _as_year(parts[0][0] if parts[0] else None)
    return None


def _abstract_from_openalex(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    positions = [position for offsets in value.values() if isinstance(offsets, list) for position in offsets if isinstance(position, int)]
    if not positions:
        return None
    words = [""] * (max(positions) + 1)
    for word, offsets in value.items():
        if isinstance(offsets, list):
            for offset in offsets:
                if isinstance(offset, int) and offset < len(words):
                    words[offset] = str(word)
    return _clean_text(" ".join(words))


def _as_year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1000 <= year <= 9999 else None


def _as_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None

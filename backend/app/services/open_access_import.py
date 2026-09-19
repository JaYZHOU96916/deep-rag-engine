"""Controlled download of a user-selected, publicly reachable PDF for ingestion."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiofiles
import httpx


class UnsafeRemotePdfURL(ValueError):
    """The proposed URL is not safe for a server-side document fetch."""


class RemotePdfDownloadError(RuntimeError):
    """A public URL could not be retrieved as a bounded PDF."""


def validate_remote_pdf_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise UnsafeRemotePdfURL("Only public HTTPS PDF URLs are accepted.")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return value
    if not address.is_global:
        raise UnsafeRemotePdfURL("Private, loopback, and local network addresses are not accepted.")
    return value


async def download_open_access_pdf(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    client: httpx.AsyncClient | None = None,
    host_validator: Callable[[str], Awaitable[None]] | None = None,
) -> int:
    """Download one public HTTPS PDF with redirect, size, DNS, and signature safeguards."""

    current_url = validate_remote_pdf_url(url)
    validate_host = host_validator or _ensure_public_host
    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=20, follow_redirects=False)
    try:
        for _hop in range(4):
            parsed = urlparse(current_url)
            await validate_host(parsed.hostname or "")
            async with active_client.stream("GET", current_url, headers={"Accept": "application/pdf"}) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise RemotePdfDownloadError("The PDF host sent an invalid redirect.")
                    current_url = validate_remote_pdf_url(urljoin(current_url, location))
                    continue
                if response.status_code != 200:
                    raise RemotePdfDownloadError(f"The PDF host returned HTTP {response.status_code}.")
                return await _write_verified_pdf(response, destination, max_bytes)
        raise RemotePdfDownloadError("The PDF host redirected too many times.")
    except httpx.HTTPError as error:
        raise RemotePdfDownloadError("The open-access PDF could not be reached.") from error
    finally:
        if owns_client:
            await active_client.aclose()


async def _write_verified_pdf(response: httpx.Response, destination: Path, max_bytes: int) -> int:
    content_type = response.headers.get("content-type", "").lower()
    if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
        raise RemotePdfDownloadError("The selected open-access link did not return a PDF.")

    total = 0
    prefix = b""
    try:
        async with aiofiles.open(destination, "wb") as file:
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise RemotePdfDownloadError("The selected PDF exceeds the configured upload limit.")
                if len(prefix) < 8:
                    prefix += chunk[: 8 - len(prefix)]
                await file.write(chunk)
        if not prefix.startswith(b"%PDF-"):
            raise RemotePdfDownloadError("The selected open-access link did not return a PDF.")
        return total
    except Exception:
        if destination.exists():
            destination.unlink()
        raise


async def _ensure_public_host(hostname: str) -> None:
    try:
        resolved = await asyncio.get_running_loop().getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise RemotePdfDownloadError("The PDF host could not be resolved.") from error
    addresses = {entry[4][0] for entry in resolved}
    if not addresses:
        raise RemotePdfDownloadError("The PDF host could not be resolved.")
    for value in addresses:
        if not ipaddress.ip_address(value).is_global:
            raise UnsafeRemotePdfURL("The PDF host resolved to a non-public network address.")

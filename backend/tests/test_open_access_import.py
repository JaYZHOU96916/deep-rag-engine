from pathlib import Path

import httpx
import pytest

from app.services.open_access_import import UnsafeRemotePdfURL, download_open_access_pdf, validate_remote_pdf_url


async def _allow_host(_: str) -> None:
    return None


@pytest.mark.asyncio
async def test_open_access_downloader_writes_only_a_bounded_pdf(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://papers.example.edu/article.pdf"
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7\nfixture")

    output = tmp_path / "article.pdf"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        byte_count = await download_open_access_pdf(
            "https://papers.example.edu/article.pdf",
            output,
            max_bytes=1_000,
            client=client,
            host_validator=_allow_host,
        )

    assert byte_count == len(b"%PDF-1.7\nfixture")
    assert output.read_bytes().startswith(b"%PDF-")


def test_open_access_downloader_rejects_a_private_network_target() -> None:
    with pytest.raises(UnsafeRemotePdfURL):
        validate_remote_pdf_url("http://127.0.0.1/private.pdf")

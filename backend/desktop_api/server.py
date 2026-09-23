"""Entry point bundled into the macOS app by PyInstaller."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "desktop_api.main:app",
        host="127.0.0.1",
        port=int(os.environ.get("DEEP_RAG_DESKTOP_PORT", "8000")),
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()

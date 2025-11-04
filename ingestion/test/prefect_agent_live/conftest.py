from __future__ import annotations

import os
import socket
import contextlib
from typing import Generator
import pytest
import httpx


DEFAULT_BASE_URLS = {
    "autoshot": os.getenv("AUTOSHOT_BASE_URL", "http://100.113.186.28:8001"),
    "asr": os.getenv("ASR_BASE_URL", "http://100.113.186.28:8002"),
    "image_embedding": os.getenv("IMAGE_EMBED_BASE_URL", "http://100.113.186.28:8003"),
    "llm": os.getenv("LLM_BASE_URL", "http://100.113.186.28:8004"),
    "text_embedding": os.getenv("TEXT_EMBED_BASE_URL", "http://100.113.186.28:8005"),
}


CONSTANT = {
    'image_path': '/home/tinhanhnguyen/Desktop/HK7/Capstone/CAPSTONE_PROJECT/ingestion/test/asset/test.jpg',
    'video_s3_url': 's3://testbucket/2025-10-14 15-07-14.mkv'
}

def _can_connect(url: str) -> bool:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with contextlib.closing(socket.create_connection((host, port), timeout=1.0)):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def base_urls() -> dict[str, str]:
    return dict(DEFAULT_BASE_URLS)

@pytest.fixture(scope='session')
def constants_path() -> dict[str,str]:
    return dict(CONSTANT)
    


@pytest.fixture
def http(timeout: float = 30.0) -> Generator[httpx.Client]:
    with httpx.Client(timeout=timeout) as client:
        yield client




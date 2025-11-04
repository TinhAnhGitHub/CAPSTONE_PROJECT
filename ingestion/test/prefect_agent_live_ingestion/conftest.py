from __future__ import annotations

import os
import contextlib
import socket
from typing import Generator

import pytest
import httpx


DEFAULT_BASE_URLS = {
    "ingestion": os.getenv("INGESTION_BASE_URL", "http://localhost:8000"),
}


@pytest.fixture(scope="session")
def base_urls() -> dict[str, str]:
    return dict(DEFAULT_BASE_URLS)


@pytest.fixture
def http(timeout: float = 30.0) -> Generator[httpx.Client, None, None]:
    with httpx.Client(timeout=timeout) as client:
        yield client



@pytest.fixture(scope='session')
def mock_data_sample()-> list[list[str]]:
    video_id = [
        'video1_111',
        'video2_222',
        # 'video2_333',  
    ]
    # video_id = [
    #     'video1_444',
    #     'video2_555',
    #     'video2_666',
    # ]
    s3_url = [
        's3://testbucket/videoplayback_1.mp4',
        's3://testbucket/videoplayback_2.mp4',
        # 's3://testbucket/videoplayback_3.mp4',
    ]

    # s3_url = [
    #     's3://testbucket/videoplayback_4.mp4',
    #     's3://testbucket/videoplayback_5.mp4',
    #     's3://testbucket/videoplayback_6.mp4',
    # ]

    return [[vid, s3] for vid, s3 in zip(video_id, s3_url)]


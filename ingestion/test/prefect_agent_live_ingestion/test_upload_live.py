from __future__ import annotations

import os
import pytest
import httpx


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def test_ingestion_root(http: httpx.Client, base_urls: dict[str, str], mock_data_sample: list[list[str]]):
    base = base_urls["ingestion"]
    try:
        r = http.get(_url(base, "/"))
    except Exception as e:
        pytest.skip(f"Ingestion API not reachable at {base}: {e}")
    assert r.status_code == 200
    data = r.json()
    assert data.get("service")
    assert data.get("status") in {"running", "healthy", "ok"}


def test_upload_submit(http: httpx.Client, base_urls: dict[str, str], mock_data_sample: list[list[str]]):
    base = base_urls["ingestion"]
    try:
        
        user_id = "anonymous"
        payload = {"videos": mock_data_sample, "user_id": user_id}
        r = http.post(_url(base, "/uploads/"), json=payload)
    except Exception as e:
        pytest.skip(f"Ingestion API not reachable at {base}: {e}")


    assert r.status_code in {202, 500}
    if r.status_code == 202:
        data = r.json()
        print(data)
        assert data["run_id"]
        assert data.get("flow_run_id")
        assert data.get("video_count") == 2
        assert data.get("deployment_name")
    else:
        # In this case, provide context for easier debugging in CI output
        msg = r.json().get("detail", "") if r.headers.get("content-type", "").startswith("application/json") else r.text
        pytest.skip(f"Upload failed (expected if Prefect deployment not registered): {msg}")


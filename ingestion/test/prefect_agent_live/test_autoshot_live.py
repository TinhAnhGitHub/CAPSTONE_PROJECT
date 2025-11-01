from __future__ import annotations

import os
import pytest
import httpx


# os.environ['TEST_AUTOSHOT_VIDEO_S3'] = "s3://testbucket/2025-10-14 15-07-14.mkv"


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def test_autoshot_health(http: httpx.Client, base_urls: dict[str, str], constants_path:dict[str,str] ):
    base = base_urls["autoshot"]
    try:
        r = http.get(_url(base, "/health"))
    except Exception as e:
        pytest.skip(f"Autoshot not reachable at {base}: {e}")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_autoshot_models_status(http: httpx.Client, base_urls: dict[str, str], constants_path:dict[str,str] ):
    base = base_urls["autoshot"]
    try:
        r = http.get(_url(base, "/autoshot/models"))
    except Exception as e:
        pytest.skip(f"Autoshot not reachable at {base}: {e}")
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data


def test_autoshot_load_infer_unload_cycle(http: httpx.Client, base_urls: dict[str, str], constants_path:dict[str,str] ):
    base = base_urls["autoshot"]

    # Discover models
    try:
        r = http.get(_url(base, "/autoshot/models"))
    except Exception as e:
        pytest.skip(f"Autoshot not reachable at {base}: {e}")
    assert r.status_code == 200
    models = r.json().get("available_models", [])
    if not models:
        pytest.skip("No Autoshot models available to load")

    model_name = os.getenv("TEST_AUTOSHOT_MODEL", models[0])

    # Load CPU by default
    r = http.post(_url(base, "/autoshot/load"), json={"model_name": model_name, "device": "cuda"})
    assert r.status_code == 200

    # Inference requires an s3 URL; skip if not provided
    s3_url = constants_path['video_s3_url']
    if s3_url:
        r = http.post(_url(base, "/autoshot/infer"), json={"s3_minio_url": s3_url, "metadata": {}})
        assert r.status_code == 200
        out = r.json()
        assert out.get("status") == "success"
        assert isinstance(out.get("scenes"), list)
    else:
        pytest.skip("TEST_AUTOSHOT_VIDEO_S3 not set; skipping Autoshot inference")

    # Unload
    r = http.post(_url(base, "/autoshot/unload"), json={"cleanup_memory": True})
    assert r.status_code == 200


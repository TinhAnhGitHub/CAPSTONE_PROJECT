from __future__ import annotations

import os
import pytest
import httpx



# os.environ['TEST_ASR_VIDEO_S3'] = "s3://testbucket/2025-10-14 15-07-14.mkv"


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def test_asr_health(http: httpx.Client, base_urls: dict[str, str], constants_path:dict[str,str] ):
    base = base_urls["asr"]
    try:
        r = http.get(_url(base, "/health"))
    except Exception as e:
        pytest.skip(f"ASR service not reachable at {base}: {e}")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "healthy"


def test_asr_models_status(http: httpx.Client, base_urls: dict[str, str], constants_path: dict[str,str]):
    base = base_urls["asr"]
    try:
        r = http.get(_url(base, "/asr/models"))
    except Exception as e:
        pytest.skip(f"ASR service not reachable at {base}: {e}")
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data


def test_asr_load_infer_unload_cycle(http: httpx.Client, base_urls: dict[str, str], constants_path: dict[str,str]):
    base = base_urls["asr"]

    try:
        r = http.get(_url(base, "/asr/models"))
    except Exception as e:
        pytest.skip(f"ASR service not reachable at {base}: {e}")
    assert r.status_code == 200
    models = r.json().get("available_models", [])
    if not models:
        pytest.skip("No ASR models available to load")

    print(models)
    model_name = os.getenv("TEST_ASR_MODEL", models[0])

    # Load model on CPU by default
    r = http.post(_url(base, "/asr/load"), json={"model_name": model_name, "device": "cuda"})
    assert r.status_code == 200

    # Infer only when input is provided
    video_url = constants_path['video_s3_url']
    if video_url:
        r = http.post(
            _url(base, "/asr/infer"),
            json={"video_minio_url": video_url, "metadata": {}},
        )
        assert r.status_code == 200
        out = r.json()
        assert out.get("status") == "success"
        assert out.get("video_minio_url")
    else:
        pytest.skip("TEST_ASR_VIDEO_S3 not set; skipping ASR inference")


    r = http.post(_url(base, "/asr/unload"), json={"cleanup_memory": True})
    assert r.status_code == 200


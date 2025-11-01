from __future__ import annotations

import os
import pytest
import httpx


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def test_text_embedding_health(http: httpx.Client, base_urls: dict[str, str]):
    base = base_urls["text_embedding"]
    try:
        r = http.get(_url(base, "/health"))
    except Exception as e:
        pytest.skip(f"Text-Embedding not reachable at {base}: {e}")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_text_embedding_models_status(http: httpx.Client, base_urls: dict[str, str]):
    base = base_urls["text_embedding"]
    try:
        r = http.get(_url(base, "/text-embedding/models"))
    except Exception as e:
        pytest.skip(f"Text-Embedding not reachable at {base}: {e}")
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data


def test_text_embedding_load_infer_unload(http: httpx.Client, base_urls: dict[str, str]):
    base = base_urls["text_embedding"]

    # Discover models
    try:
        r = http.get(_url(base, "/text-embedding/models"))
    except Exception as e:
        pytest.skip(f"Text-Embedding not reachable at {base}: {e}")
    assert r.status_code == 200
    models = r.json().get("available_models", [])
    if not models:
        pytest.skip("No text embedding models available to load")

    model_name = os.getenv("TEST_TEXT_EMBED_MODEL", models[0])

    # Load model
    r = http.post(_url(base, "/text-embedding/load"), json={"model_name": model_name, "device": "cuda"})
    assert r.status_code == 200

    # Inference simple texts
    r = http.post(
        _url(base, "/text-embedding/infer"),
        json={"texts": ["alpha", "beta"], "metadata": {}},
    )
    assert r.status_code == 200
    out = r.json()
    assert out.get("status") == "success"
    assert out.get("embeddings") and len(out["embeddings"]) == 2

    # Unload
    r = http.post(_url(base, "/text-embedding/unload"), json={"cleanup_memory": True})
    assert r.status_code == 200


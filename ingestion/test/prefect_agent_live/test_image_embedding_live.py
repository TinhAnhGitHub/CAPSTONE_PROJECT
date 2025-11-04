from __future__ import annotations

import os
import pytest
import httpx
import base64
import sys
print(__file__)


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def test_image_embedding_health(http: httpx.Client, base_urls: dict[str, str]):
    base = base_urls["image_embedding"]
    try:
        r = http.get(_url(base, "/health"))
    except Exception as e:
        pytest.skip(f"Image-Embedding not reachable at {base}: {e}")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_image_embedding_models_status(http: httpx.Client, base_urls: dict[str, str]):
    base = base_urls["image_embedding"]
    try:
        r = http.get(_url(base, "/image-embedding/models"))
    except Exception as e:
        pytest.skip(f"Image-Embedding not reachable at {base}: {e}")
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data


def test_image_embedding_load_infer_text_only_unload(http: httpx.Client, base_urls: dict[str, str]):
    base = base_urls["image_embedding"]

    # Discover models
    try:
        r = http.get(_url(base, "/image-embedding/models"))
    except Exception as e:
        pytest.skip(f"Image-Embedding not reachable at {base}: {e}")
    assert r.status_code == 200
    models = r.json().get("available_models", [])
    if not models:
        pytest.skip("No image embedding models available to load")

    model_name = os.getenv("TEST_IMAGE_EMBED_MODEL", models[0])

    r = http.post(_url(base, "/image-embedding/load"), json={"model_name": model_name, "device": "cuda"})
    assert r.status_code == 200

    # Use text-
    r = http.post(
        _url(base, "/image-embedding/infer"),
        json={"text_input": ["hello", "world"], "metadata": {}},
    )
    assert r.status_code == 200
    out = r.json()
    assert out.get("status") == "success"
    assert out.get("text_embeddings") and len(out["text_embeddings"]) == 2

    r = http.post(_url(base, "/image-embedding/unload"), json={"cleanup_memory": True})
    assert r.status_code == 200



def test_image_embedding_load_infer_image_unload(http: httpx.Client, base_urls: dict[str,str], constants_path: dict[str, str]):
    base = base_urls["image_embedding"]

    # Discover models
    try:
        r = http.get(_url(base, "/image-embedding/models"))
    except Exception as e:
        pytest.skip(f"Image-Embedding not reachable at {base}: {e}")
    assert r.status_code == 200
    models = r.json().get("available_models", [])
    if not models:
        pytest.skip("No image embedding models available to load")

    model_name = os.getenv("TEST_IMAGE_EMBED_MODEL", models[0])

    r = http.post(_url(base, "/image-embedding/load"), json={"model_name": model_name, "device": "cuda"})
    assert r.status_code == 200

    with open(constants_path['image_path'], 'rb') as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    image_send = [image_b64 for _ in range(5)]
    
    r = http.post(
        _url(base, "/image-embedding/infer"),
        json={"image_base64": image_send, "metadata": {}},
    )

    assert r.status_code == 200
    out = r.json()
    assert out.get("status") == "success"
    assert out.get("image_embeddings") and len(out["image_embeddings"]) == 5

    r = http.post(_url(base, "/image-embedding/unload"), json={"cleanup_memory": True})
    assert r.status_code == 200

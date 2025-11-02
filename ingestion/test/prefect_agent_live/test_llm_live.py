from __future__ import annotations
import base64
import pytest
import httpx


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def test_llm_health(http: httpx.Client, base_urls: dict[str, str]):
    base = base_urls["llm"]
    try:
        r = http.get(_url(base, "/health"))
    except Exception as e:
        pytest.skip(f"LLM not reachable at {base}: {e}")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_llm_models_status(http: httpx.Client, base_urls: dict[str, str]):
    base = base_urls["llm"]
    try:
        r = http.get(_url(base, "/llm/models"))
    except Exception as e:
        pytest.skip(f"LLM not reachable at {base}: {e}")
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data


def test_llm_load_infer_unload_when_available(http: httpx.Client, base_urls: dict[str, str], constants_path: dict[str, str]):
    base = base_urls["llm"]

    # Discover models
    try:
        r = http.get(_url(base, "/llm/models"))
    except Exception as e:
        pytest.skip(f"LLM not reachable at {base}: {e}")
    assert r.status_code == 200
    models = r.json().get("available_models", [])
    if not models:
        pytest.skip("No LLM models available (API keys not configured?)")

    print(models)
    model_name = models[1]

    # Load model
    r = http.post(_url(base, "/llm/load"), json={"model_name": model_name, "device": "cpu"})
    assert r.status_code == 200
    with open(constants_path['image_path'], 'rb') as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')


    # Simple prompt inference
    batch_input = {
        "llm_requests": [
            {"prompt": "Caption this image", "image_base64": [image_b64]} for _ in range(20)
        ],
        "metadata": {"test": "batch_inference"},
    }

    r = http.post(_url(base, "/llm/infer"), json=batch_input)
    assert r.status_code == 200
    result = r.json()
    assert result.get("status") in ("success", "partial_error")
    assert "responses" in result, "Expected 'responses' field in batch output"
    assert isinstance(result["responses"], list), "Responses must be a list"
    assert len(result["responses"]) == len(batch_input["llm_requests"])
    print(result)
    for idx, resp in enumerate(result["responses"]):
        
        assert "answer" in resp, f"Response {idx} missing 'answer'"
        assert isinstance(resp["answer"], str)
        assert resp.get("status") in ("success", "error")
    r = http.post(_url(base, "/llm/unload"), json={"cleanup_memory": True})
    assert r.status_code == 200

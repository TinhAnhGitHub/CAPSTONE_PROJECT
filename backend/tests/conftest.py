"""
Test configuration for the backend.
External services (MongoDB/Beanie, socketio, MinIO, Redis, Google OAuth)
are stubbed at the sys.modules level so no live infrastructure is needed.
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# ── 1. Required env vars (must be set before AppSettings is imported) ──────────
os.environ.setdefault("GOOGLE_OAUTH_CLIENT_ID", "test-google-client-id")
os.environ.setdefault("GOOGLE_OAUTH_CLIENT_SECRET", "test-google-client-secret")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")

# ── 2. Stub heavy external libs ────────────────────────────────────────────────
_sio_stub = MagicMock()
_sio_stub.emit = AsyncMock(return_value=None)

for _mod, _stub in {
    "socketio": MagicMock(AsyncServer=MagicMock(return_value=_sio_stub),
                          ASGIApp=MagicMock(return_value=MagicMock())),
    "motor": MagicMock(),
    "motor.motor_asyncio": MagicMock(),
    "beanie": MagicMock(),
    "redis": MagicMock(),
    "redis.asyncio": MagicMock(),
    "minio": MagicMock(),
    "google.auth.transport.requests": MagicMock(),
    "google.oauth2.id_token": MagicMock(),
}.items():
    sys.modules.setdefault(_mod, _stub)

# ── 3. Now safe to import app code ─────────────────────────────────────────────
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.user import router as user_router, verify_token
from app.core.dependencies import get_user_service, get_chat_service, get_agent

# ── 4. Fixtures ────────────────────────────────────────────────────────────────
FAKE_USER = {
    "user_id": "6916f84e9a79606c0413d5d6",
    "email": "test@example.com",
    "google_id": "109380215299372172369",
}


def _make_user_svc_mock():
    svc = MagicMock()
    svc.get_user_chat_history = AsyncMock(return_value=[{"_id": "sess-1", "name": "Chat 1"}])
    svc.create_new_chat_session = AsyncMock(return_value="new-session-abc")
    svc.get_user_groups = AsyncMock(return_value=[{"_id": "grp-1", "name": "default"}])
    svc.create_user_group = AsyncMock(return_value="new-group-xyz")
    svc.rename_session = AsyncMock(return_value=True)
    svc.rename_group = AsyncMock(return_value=True)
    svc.rename_video = AsyncMock(return_value=True)
    svc.delete_session = AsyncMock(return_value=True)
    svc.delete_group = AsyncMock(return_value=True)
    svc.get_user_videos = AsyncMock(return_value=[])
    svc.select_videos = AsyncMock(return_value=None)
    svc.search_text_messages = AsyncMock(return_value=[])
    svc.get_user_chat_detail = AsyncMock(return_value=[])
    svc.retry_ingestion = AsyncMock(return_value=None)
    svc.create_jwt_token = MagicMock(return_value="fake-jwt-token")
    return svc


@pytest.fixture
def mock_user_svc():
    return _make_user_svc_mock()


@pytest.fixture
def client(mock_user_svc):
    """TestClient with all DB/service dependencies mocked out."""
    app = FastAPI()
    app.dependency_overrides[verify_token] = lambda: FAKE_USER
    app.dependency_overrides[get_user_service] = lambda: mock_user_svc
    app.dependency_overrides[get_chat_service] = lambda: MagicMock()
    app.dependency_overrides[get_agent] = lambda: MagicMock()
    app.include_router(user_router)
    return TestClient(app)

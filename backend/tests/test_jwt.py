"""
TC-JWT-*: JWT token creation & verification tests.
Tests the pure-Python create_jwt_token() in UserService
and verify_token() in the user API module.
"""
import pytest
import jwt
from unittest.mock import MagicMock

from app.schema.user import SECRET_KEY, ALGORITHM
from app.service.user import UserService
from app.api.user import verify_token


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_service() -> UserService:
    """Instantiate UserService with stub dependencies."""
    return UserService(minio_service=MagicMock(), sio=MagicMock())


USER_DATA = {
    "user_id": "abc123",
    "email": "alice@example.com",
    "google_id": "g-999",
}


# ── create_jwt_token ──────────────────────────────────────────────────────────

class TestCreateJwtToken:
    def test_tc_jwt_01_returns_string(self):
        """TC-JWT-01: create_jwt_token returns a non-empty string."""
        svc = make_service()
        token = svc.create_jwt_token(USER_DATA)
        assert isinstance(token, str) and len(token) > 0

    def test_tc_jwt_02_decoded_user_id(self):
        """TC-JWT-02: Decoded token contains correct user_id."""
        svc = make_service()
        token = svc.create_jwt_token(USER_DATA)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["user_id"] == "abc123"

    def test_tc_jwt_03_decoded_email(self):
        """TC-JWT-03: Decoded token contains correct email."""
        svc = make_service()
        token = svc.create_jwt_token(USER_DATA)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["email"] == "alice@example.com"

    def test_tc_jwt_04_decoded_google_id(self):
        """TC-JWT-04: Decoded token contains google_id."""
        svc = make_service()
        token = svc.create_jwt_token(USER_DATA)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["google_id"] == "g-999"

    def test_tc_jwt_05_token_has_exp(self):
        """TC-JWT-05: Token payload includes an expiry (exp) claim."""
        svc = make_service()
        token = svc.create_jwt_token(USER_DATA)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload

    def test_tc_jwt_06_token_has_iat(self):
        """TC-JWT-06: Token payload includes an issued-at (iat) claim."""
        svc = make_service()
        token = svc.create_jwt_token(USER_DATA)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "iat" in payload

    def test_tc_jwt_07_different_users_get_different_tokens(self):
        """TC-JWT-07: Two different users receive different tokens."""
        svc = make_service()
        t1 = svc.create_jwt_token({**USER_DATA, "user_id": "u1"})
        t2 = svc.create_jwt_token({**USER_DATA, "user_id": "u2"})
        assert t1 != t2


# ── verify_token ──────────────────────────────────────────────────────────────

class TestVerifyToken:
    def test_tc_jwt_08_valid_token_returns_payload(self):
        """TC-JWT-08: verify_token decodes a valid token and returns payload."""
        svc = make_service()
        token = svc.create_jwt_token(USER_DATA)

        # Wrap token in a mock credentials object
        creds = MagicMock()
        creds.credentials = token

        payload = verify_token(credentials=creds)
        assert payload["user_id"] == "abc123"
        assert payload["email"] == "alice@example.com"

    def test_tc_jwt_09_expired_token_raises_401(self):
        """TC-JWT-09: Expired token raises HTTP 401 with 'Token expired' detail."""
        from fastapi import HTTPException
        expired_token = jwt.encode(
            {"user_id": "x", "exp": 1},  # exp=1 → definitely expired
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        creds = MagicMock()
        creds.credentials = expired_token
        with pytest.raises(HTTPException) as exc_info:
            verify_token(credentials=creds)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_tc_jwt_10_invalid_token_raises_401(self):
        """TC-JWT-10: Garbage token raises HTTP 401 with 'Invalid token' detail."""
        from fastapi import HTTPException
        creds = MagicMock()
        creds.credentials = "not.a.real.token"
        with pytest.raises(HTTPException) as exc_info:
            verify_token(credentials=creds)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail  # some detail message present

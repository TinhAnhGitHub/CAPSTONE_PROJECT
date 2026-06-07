"""
TC-SCH-*: Pydantic schema validation tests.
Pure unit tests — no DB or HTTP needed.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from app.schema.user import Token, TokenData, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_HOURS
from app.schema.chat import ChatRequest, ChatResponse, SessionInfo


# ── Token ─────────────────────────────────────────────────────────────────────

class TestTokenSchema:
    def test_tc_sch_01_valid_token(self):
        """TC-SCH-01: Token accepts valid access_token and token_type."""
        t = Token(access_token="abc.def.ghi", token_type="Bearer")
        assert t.access_token == "abc.def.ghi"
        assert t.token_type == "Bearer"

    def test_tc_sch_02_token_missing_fields(self):
        """TC-SCH-02: Token raises ValidationError when fields are missing."""
        with pytest.raises(ValidationError):
            Token()

    def test_tc_sch_03_token_data_username_defaults_none(self):
        """TC-SCH-03: TokenData.username defaults to None."""
        td = TokenData()
        assert td.username is None

    def test_tc_sch_04_token_data_accepts_username(self):
        """TC-SCH-04: TokenData accepts a string username."""
        td = TokenData(username="alice")
        assert td.username == "alice"


# ── ChatRequest ───────────────────────────────────────────────────────────────

class TestChatSchemas:
    def test_tc_sch_05_chat_request_valid(self):
        """TC-SCH-05: ChatRequest accepts session_id and message."""
        req = ChatRequest(session_id="sess-1", message="Hello")
        assert req.session_id == "sess-1"
        assert req.message == "Hello"

    def test_tc_sch_06_chat_request_missing_session_id(self):
        """TC-SCH-06: ChatRequest raises ValidationError when session_id is missing."""
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello")

    def test_tc_sch_07_chat_response_valid(self):
        """TC-SCH-07: ChatResponse serialises correctly."""
        now = datetime.now()
        resp = ChatResponse(session_id="sess-1", response="Hi there", timestamp=now)
        assert resp.response == "Hi there"
        assert resp.session_id == "sess-1"

    def test_tc_sch_08_session_info_valid(self):
        """TC-SCH-08: SessionInfo requires all three fields."""
        si = SessionInfo(session_id="s1", last_updated=datetime.now(), message_count=5)
        assert si.message_count == 5


# ── Constants ─────────────────────────────────────────────────────────────────

class TestSchemaConstants:
    def test_tc_sch_09_algorithm_is_hs256(self):
        """TC-SCH-09: ALGORITHM constant equals HS256."""
        assert ALGORITHM == "HS256"

    def test_tc_sch_10_access_token_expire_hours_positive(self):
        """TC-SCH-10: ACCESS_TOKEN_EXPIRE_HOURS is a positive integer."""
        assert isinstance(ACCESS_TOKEN_EXPIRE_HOURS, int)
        assert ACCESS_TOKEN_EXPIRE_HOURS > 0

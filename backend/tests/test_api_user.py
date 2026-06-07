"""
TC-USR-*: User API route tests.
All DB/service calls are replaced by async mocks via dependency_overrides.
"""
import pytest


class TestChatHistoryRoutes:
    def test_tc_usr_01_get_chat_history_200(self, client):
        """TC-USR-01: GET /api/user/chat-history returns 200."""
        resp = client.get("/api/user/chat-history")
        assert resp.status_code == 200

    def test_tc_usr_02_get_chat_history_has_chats_key(self, client):
        """TC-USR-02: chat-history response contains 'chats' key."""
        resp = client.get("/api/user/chat-history")
        assert "chats" in resp.json()

    def test_tc_usr_03_get_chat_history_returns_list(self, client):
        """TC-USR-03: 'chats' value is a list."""
        data = client.get("/api/user/chat-history").json()
        assert isinstance(data["chats"], list)


class TestNewChatRoute:
    def test_tc_usr_04_post_new_chat_200(self, client):
        """TC-USR-04: POST /api/user/new-chat returns 200."""
        assert client.post("/api/user/new-chat").status_code == 200

    def test_tc_usr_05_post_new_chat_returns_session_id(self, client):
        """TC-USR-05: new-chat response contains chat_session_id."""
        data = client.post("/api/user/new-chat").json()
        assert "chat_session_id" in data
        assert data["chat_session_id"] == "new-session-abc"


class TestGroupRoutes:
    def test_tc_usr_06_get_groups_200(self, client):
        """TC-USR-06: GET /api/user/groups returns 200."""
        assert client.get("/api/user/groups").status_code == 200

    def test_tc_usr_07_get_groups_has_groups_key(self, client):
        """TC-USR-07: groups response contains 'groups' key as a list."""
        data = client.get("/api/user/groups").json()
        assert "groups" in data
        assert isinstance(data["groups"], list)

    def test_tc_usr_08_post_create_group_200(self, client):
        """TC-USR-08: POST /api/user/groups/create returns 200."""
        resp = client.post("/api/user/groups/create", json={"group_name": "My Group"})
        assert resp.status_code == 200

    def test_tc_usr_09_post_create_group_returns_group_id(self, client):
        """TC-USR-09: create-group response contains group_id."""
        data = client.post("/api/user/groups/create", json={"group_name": "Test"}).json()
        assert "group_id" in data


class TestRenameRoutes:
    def test_tc_usr_10_rename_session_200(self, client):
        """TC-USR-10: PATCH /api/user/session/{id}/rename returns 200 on success."""
        resp = client.patch(
            "/api/user/session/sess-abc/rename",
            json={"new_name": "Renamed Chat"},
        )
        assert resp.status_code == 200

    def test_tc_usr_11_rename_session_404_when_not_found(self, client, mock_user_svc):
        """TC-USR-11: PATCH rename returns 404 when session doesn't exist."""
        from unittest.mock import AsyncMock
        mock_user_svc.rename_session = AsyncMock(return_value=False)
        resp = client.patch(
            "/api/user/session/nonexistent/rename",
            json={"new_name": "Ghost"},
        )
        assert resp.status_code == 404

    def test_tc_usr_12_rename_group_200(self, client):
        """TC-USR-12: PATCH /api/user/group/{id}/rename returns 200 on success."""
        resp = client.patch(
            "/api/user/group/grp-1/rename",
            json={"new_name": "New Group Name"},
        )
        assert resp.status_code == 200

    def test_tc_usr_13_rename_video_200(self, client):
        """TC-USR-13: PATCH /api/user/video/{id}/rename returns 200 on success."""
        resp = client.patch(
            "/api/user/video/vid-1/rename",
            json={"new_name": "Renamed Video"},
        )
        assert resp.status_code == 200


class TestDeleteRoutes:
    def test_tc_usr_14_delete_session_200(self, client):
        """TC-USR-14: DELETE /api/user/session/{id}/delete returns 200."""
        resp = client.delete("/api/user/session/sess-1/delete")
        assert resp.status_code == 200

    def test_tc_usr_15_delete_session_returns_session_id(self, client):
        """TC-USR-15: delete-session response contains session_id."""
        data = client.delete("/api/user/session/sess-abc/delete").json()
        assert data["session_id"] == "sess-abc"

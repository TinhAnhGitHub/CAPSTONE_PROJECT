import os

os.environ.setdefault("VIDEO_DEEP_SEARCH_TESTING", "1")

from fastapi.testclient import TestClient

from videodeepsearch.main import app
from videodeepsearch.agent.orc_service import WorkflowService
from videodeepsearch.agent.workflow import VideoAgentWorkFlow


def test_workflow_service_attached():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "video-agent-workflow"}

        service = app.state.workflow_service
        assert isinstance(service, WorkflowService)
        assert isinstance(service.orchestration, VideoAgentWorkFlow)

from fastapi import Request
from videodeepsearch.tools.type.factory import ToolFactory
from videodeepsearch.agent.orc_service import WorkflowService
from .app_state import Appstate

def get_workflow_service(request: Request) -> WorkflowService:
    return request.app.state.workflow_service


def get_tool_factory() -> ToolFactory:
    return Appstate().tool_factory
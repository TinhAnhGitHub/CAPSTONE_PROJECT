from typing import TYPE_CHECKING

from fastapi import WebSocket
from videodeepsearch.tools.type.factory import ToolFactory
from .app_state import Appstate

if TYPE_CHECKING:
    from videodeepsearch.agent.orc_service import WorkflowService


def get_workflow_service(websocket: WebSocket) -> "WorkflowService":
    return websocket.app.state.workflow_service


def get_tool_factory() -> ToolFactory:
    return Appstate().tool_factory

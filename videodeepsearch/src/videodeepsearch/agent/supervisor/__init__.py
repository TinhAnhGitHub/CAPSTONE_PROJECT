"""Supervisor teams for VideoDeepSearch."""

from videodeepsearch.agent.supervisor.greeter import build_videodeepsearch_team
from videodeepsearch.agent.supervisor.orchestrator import build_orchestrator_team, WorkerModel

__all__ = [
    "build_videodeepsearch_team",
    "build_orchestrator_team",
    "WorkerModel",
]

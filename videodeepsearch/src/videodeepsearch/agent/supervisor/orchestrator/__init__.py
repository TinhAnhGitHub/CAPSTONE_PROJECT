"""Orchestrator Team - Handles video search and retrieval."""

from videodeepsearch.agent.supervisor.orchestrator.team import build_orchestrator_team
from videodeepsearch.agent.supervisor.orchestrator.spawn_toolkit import SpawnWorkerToolkit, WorkerModel
from videodeepsearch.agent.supervisor.orchestrator.prompt import (
    ORCHESTRATOR_DESCRIPTION,
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_INSTRUCTIONS,
)

__all__ = [
    "build_orchestrator_team",
    "SpawnWorkerToolkit",
    "WorkerModel",
    "ORCHESTRATOR_DESCRIPTION",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "ORCHESTRATOR_INSTRUCTIONS",
]

"""
Lightweight agent identifiers to avoid circular imports during package init.

Do not import subworker here; it indirectly imports orc_events during
initialization via its definition module, which creates a cycle when
orc_events imports worker modules.
"""

GREETER_NAME = "GREETER_AGENT"
PLANNER_NAME = "PLANNER_AGENT"
ORCHESTRATION_NAME = "SUBORCHESTRATION_AGENT"
SUB_WORKER_NAME = "SUB_WORKER_AGENT"

FINAL_RESPONSE_AGENT = "FINAL_RESPONSE_AGENT"

from .greeter import agent as _greeter_agent  # noqa: F401
from .planner import agent as _planner_agent  # noqa: F401
from .suborchestrate import agent as _suborchestrate_agent  # noqa: F401

__all__ = [
    'GREETER_NAME',
    'PLANNER_NAME',
    'ORCHESTRATION_NAME',
    'SUB_WORKER_NAME',
    'FINAL_RESPONSE_AGENT',
]


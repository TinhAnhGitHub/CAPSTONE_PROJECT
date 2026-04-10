"""Scorers for VideoDeepSearch agent evaluation.

Provides pre-configured scorer sets for MLflow GenAI and DeepEval metrics.
"""

from videodeepsearch.evaluation.scorers.config import (
    ScorerSettings,
    get_default_settings,
)
from videodeepsearch.evaluation.scorers.presets import (
    # Preset functions
    get_ground_truth_free_scorers,
    get_with_expectations_scorers,
    get_full_agentic_scorers,
    get_scorers_by_preset,
    # Preset names
    PRESET_GROUND_TRUTH_FREE,
    PRESET_WITH_EXPECTATIONS,
    PRESET_FULL_AGENTIC,
)

__all__ = [
    # Config
    "ScorerSettings",
    "get_default_settings",
    # Presets
    "get_ground_truth_free_scorers",
    "get_with_expectations_scorers",
    "get_full_agentic_scorers",
    "get_scorers_by_preset",
    # Preset names
    "PRESET_GROUND_TRUTH_FREE",
    "PRESET_WITH_EXPECTATIONS",
    "PRESET_FULL_AGENTIC",
]

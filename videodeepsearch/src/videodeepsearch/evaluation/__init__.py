"""
Evaluation package for VideoDeepSearch agent testing and monitoring.

Provides runners, display utilities, scorers, datasets, and metrics
for comprehensive agent evaluation.

Notebook API Usage:
-------------------
```python
from videodeepsearch.evaluation import (
    DatasetBuilder,
    run_single_record,
    run_evaluation_cycle,
    get_ground_truth_free_scorers,
)

# 1. Build dataset
builder = DatasetBuilder("my_eval")
builder.add_record(
    user_id="user1",
    video_ids=["video1"],
    user_demand="Find info about X",
)
builder.save_to_json("eval_dataset.json")

# 2. Run evaluation cycle
results = await run_evaluation_cycle(
    dataset="eval_dataset.json",
    scorer_preset="ground_truth_free",
)
```
"""

# Dataset
from videodeepsearch.evaluation.datasets import (
    DatasetBuilder,
    EvalRecord,
    EvalInput,
    EvalExpectation,
    ExpectedFact,
    EvalTrace,
    ToolCallTrace,
    TokenUsage,
    load_dataset,
    load_from_json,
    load_from_yaml,
    save_to_json,
    save_to_yaml,
)

# Scorers
from videodeepsearch.evaluation.scorers import (
    ScorerSettings,
    get_default_settings,
    get_ground_truth_free_scorers,
    get_with_expectations_scorers,
    get_full_agentic_scorers,
    get_scorers_by_preset,
    PRESET_GROUND_TRUTH_FREE,
    PRESET_WITH_EXPECTATIONS,
    PRESET_FULL_AGENTIC,
)

# # Runners
# from videodeepsearch.evaluation.runners.evaluation_runner import (
#     run_single_record,
#     run_dataset,
#     run_evaluation_cycle,
# )
# from videodeepsearch.evaluation.runners.validation_runner import (
#     run_validation,
# )

# Utilities
from videodeepsearch.evaluation.util.mlflow_runner import (
    mlflow_setup,
)

__all__ = [
    # Dataset - Core
    "DatasetBuilder",
    "EvalRecord",
    "EvalInput",
    "EvalExpectation",
    "ExpectedFact",
    "EvalTrace",
    "ToolCallTrace",
    "TokenUsage",
    # Dataset - Loaders
    "load_dataset",
    "load_from_json",
    "load_from_yaml",
    "save_to_json",
    "save_to_yaml",
    # Scorers - Config
    "ScorerSettings",
    "get_default_settings",
    # Scorers - Presets
    "get_ground_truth_free_scorers",
    "get_with_expectations_scorers",
    "get_full_agentic_scorers",
    "get_scorers_by_preset",
    "PRESET_GROUND_TRUTH_FREE",
    "PRESET_WITH_EXPECTATIONS",
    "PRESET_FULL_AGENTIC",
    # # Runners - Evaluation
    # "run_single_record",
    # "run_dataset",
    # "run_evaluation_cycle",
    # # Runners - Validation
    # "run_validation",
    # # Utilities
    # "mlflow_setup",
]
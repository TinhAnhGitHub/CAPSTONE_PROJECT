"""Manual MLflow cost logging for Agno team runs.

This module provides utilities to manually log metrics that MLflow's autolog misses:
- Cost
- Full token breakdown (cache, reasoning, audio)
- Per-model costs
- Tool call statistics
"""

import mlflow
from agno.metrics import ModelMetrics
from typing import Any, Dict, List, Optional
from loguru import logger


def log_session_metrics(session_metrics: Any, prefix: str = "team") -> None:
    """Log full session metrics from team.aget_session_metrics().

    Args:
        session_metrics: SessionMetrics object from agno
        prefix: Prefix for metric names (e.g., "team", "row_1")
    """
    if session_metrics is None:
        logger.warning("No session metrics to log")
        return

    
    mlflow.log_metric(f"{prefix}/input_tokens", value=session_metrics.input_tokens)
    mlflow.log_metric(f"{prefix}/output_tokens", value=session_metrics.output_tokens)
    mlflow.log_metric(f"{prefix}/total_tokens", value=session_metrics.total_tokens)

    if session_metrics.audio_input_tokens:
        mlflow.log_metric(f"{prefix}/audio_input_tokens", session_metrics.audio_input_tokens)
    if session_metrics.audio_output_tokens:
        mlflow.log_metric(f"{prefix}/audio_output_tokens", session_metrics.audio_output_tokens)
    if session_metrics.audio_total_tokens:
        mlflow.log_metric(f"{prefix}/audio_total_tokens", session_metrics.audio_total_tokens)

    if session_metrics.cache_read_tokens:
        mlflow.log_metric(f"{prefix}/cache_read_tokens", session_metrics.cache_read_tokens)
    if session_metrics.cache_write_tokens:
        mlflow.log_metric(f"{prefix}/cache_write_tokens", session_metrics.cache_write_tokens)
    if session_metrics.reasoning_tokens:
        mlflow.log_metric(f"{prefix}/reasoning_tokens", session_metrics.reasoning_tokens)

    # Log cost
    if session_metrics.cost is not None:
        mlflow.log_metric(f"{prefix}/cost", session_metrics.cost)

    # Log per-model breakdown as a dict artifact
    if session_metrics.details:
        model_costs = extract_model_costs(session_metrics.details)
        mlflow.log_dict(model_costs, f"{prefix}_model_costs.json")

        # Log per-model metrics individually
        for model_key, metrics in model_costs.items():
            safe_model_key = model_key.replace("/", "_").replace(".", "_")
            mlflow.log_metric(f"{prefix}/model_{safe_model_key}_tokens", metrics.get("total_tokens", 0))
            mlflow.log_metric(f"{prefix}/model_{safe_model_key}_cost", metrics.get("cost", 0) or 0)
            mlflow.log_metric(f"{prefix}/model_{safe_model_key}_calls", metrics.get("count", 0))

    logger.info(f"Logged session metrics: tokens={session_metrics.total_tokens}, cost={session_metrics.cost}")


def extract_model_costs(details: Dict[str, List[ModelMetrics]]) -> Dict[str, Dict]:
    """Extract and aggregate model costs from session_metrics.details.

    Args:
        details: Dict keyed by model_type containing List of ModelMetrics dicts

    Returns:
        Dict keyed by provider/model_id with aggregated metrics
    """
    model_costs: Dict[str, Dict] = {}

    for model_type, model_metrics_list in details.items():
        for model_metrics in model_metrics_list:
            model_id = model_metrics.id
            model_provider = model_metrics.provider
            model_key = f"{model_provider}/{model_id}"

            if model_key not in model_costs:
                model_costs[model_key] = {
                    "provider": model_provider,
                    "id": model_id,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "count": 0,
                    "model_types": set(),
                }

            model_costs[model_key]["input_tokens"] += model_metrics.input_tokens
            model_costs[model_key]["output_tokens"] += model_metrics.output_tokens
            model_costs[model_key]["total_tokens"] += model_metrics.total_tokens
            if model_metrics.cost is not None:
                model_costs[model_key]["cost"] += model_metrics.cost
            model_costs[model_key]["count"] += 1
            model_costs[model_key]["model_types"].add(model_type)

    # Convert sets to lists for JSON serialization
    for model_key in model_costs:
        model_costs[model_key]["model_types"] = list(model_costs[model_key]["model_types"])

    return model_costs


def log_tool_statistics(tool_stats: Dict[str, Any], prefix: str = "team") -> None:
    """Log tool call statistics.

    Args:
        tool_stats: Dict with tool call statistics
        prefix: Prefix for metric names
    """
    mlflow.log_metric(f"{prefix}/total_tool_calls", tool_stats.get("total_calls", 0))
    mlflow.log_metric(f"{prefix}/successful_tool_calls", tool_stats.get("successful_calls", 0))
    mlflow.log_metric(f"{prefix}/failed_tool_calls", tool_stats.get("failed_calls", 0))

    # Log per-tool statistics
    per_tool_stats = tool_stats.get("per_tool", {})
    for tool_name, stats in per_tool_stats.items():
        safe_tool_name = tool_name.replace(".", "_").replace("-", "_")
        mlflow.log_metric(f"{prefix}/tool_{safe_tool_name}_calls", stats.get("calls", 0))
        mlflow.log_metric(f"{prefix}/tool_{safe_tool_name}_success", stats.get("success", 0))
        mlflow.log_metric(f"{prefix}/tool_{safe_tool_name}_failed", stats.get("failed", 0))

    # Log tool stats as artifact
    mlflow.log_dict(tool_stats, f"{prefix}_tool_statistics.json")


class ToolCallTracker:
    """Track tool calls during streaming to capture statistics."""

    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.per_tool: Dict[str, Dict[str, int]] = {}

    def track_event(self, event: Any) -> None:
        """Track a tool call event from the stream."""
        ev = getattr(event, "event", None) or type(event).__name__

        # Tool call started
        if ev in ("ToolCallStarted", "TeamToolCallStarted"):
            t = getattr(event, "tool", None)
            tool_name = getattr(t, "tool_name", "?") if t else "?"
            self.total_calls += 1
            if tool_name not in self.per_tool:
                self.per_tool[tool_name] = {"calls": 0, "success": 0, "failed": 0}
            self.per_tool[tool_name]["calls"] += 1

        # Tool call completed successfully
        elif ev in ("ToolCallCompleted", "TeamToolCallCompleted"):
            t = getattr(event, "tool", None)
            tool_name = getattr(t, "tool_name", "?") if t else "?"
            tool_error = getattr(t, "tool_call_error", None) if t else None

            if tool_error:
                self.failed_calls += 1
                if tool_name in self.per_tool:
                    self.per_tool[tool_name]["failed"] += 1
            else:
                self.successful_calls += 1
                if tool_name in self.per_tool:
                    self.per_tool[tool_name]["success"] += 1

        # Tool call error
        elif ev in ("ToolCallError", "TeamToolCallError"):
            t = getattr(event, "tool", None)
            tool_name = getattr(t, "tool_name", "?") if t else "?"
            self.failed_calls += 1
            if tool_name in self.per_tool:
                self.per_tool[tool_name]["failed"] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated tool call statistics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "per_tool": self.per_tool,
        }


def log_row_metrics(
    row_id: str,
    session_metrics: Any,
    tool_stats: Dict[str, Any],
    video_ids: List[str],
    user_demand: str,
    duration_seconds: Optional[float] = None,
) -> None:
    """Log all metrics for a single dataset row processing.

    Args:
        row_id: Unique identifier for the row
        session_metrics: SessionMetrics from team.aget_session_metrics()
        tool_stats: Tool call statistics dict
        video_ids: List of video IDs processed
        user_demand: User query/demand
        duration_seconds: Processing duration
    """
    prefix = f"row/{row_id}"

    # Log params
    mlflow.log_param(f"{prefix}/video_ids", str(video_ids))
    mlflow.log_param(f"{prefix}/user_demand", user_demand[:200])  # Truncate long demands

    if duration_seconds:
        mlflow.log_metric(f"{prefix}/duration_seconds", duration_seconds)

    # Log session metrics
    log_session_metrics(session_metrics, prefix)

    # Log tool statistics
    log_tool_statistics(tool_stats, prefix)

    logger.info(f"Logged row {row_id} metrics to MLflow")


def create_cost_summary_file(rows_data: List[Dict[str, Any]], output_path: str) -> None:
    """Create a CSV file with cost summary for all rows.

    Args:
        rows_data: List of row metric dicts
        output_path: Path to save CSV file
    """
    import csv

    if not rows_data:
        return

    # Flatten data for CSV
    fieldnames = [
        "row_id", "timestamp", "video_ids", "user_demand",
        "input_tokens", "output_tokens", "total_tokens",
        "cost", "total_tool_calls", "successful_tool_calls", "failed_tool_calls",
        "duration_seconds", "model_details"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for row in rows_data:
            # Flatten nested dicts
            flat_row = {
                "row_id": row.get("row_id"),
                "timestamp": row.get("timestamp"),
                "video_ids": str(row.get("video_ids", [])),
                "user_demand": row.get("user_demand", "")[:200],
                "input_tokens": row.get("session_metrics", {}).get("input_tokens", 0),
                "output_tokens": row.get("session_metrics", {}).get("output_tokens", 0),
                "total_tokens": row.get("session_metrics", {}).get("total_tokens", 0),
                "cost": row.get("session_metrics", {}).get("cost", 0) or 0,
                "total_tool_calls": row.get("tool_stats", {}).get("total_calls", 0),
                "successful_tool_calls": row.get("tool_stats", {}).get("successful_calls", 0),
                "failed_tool_calls": row.get("tool_stats", {}).get("failed_calls", 0),
                "duration_seconds": row.get("duration_seconds", 0) or 0,
                "model_details": str(row.get("session_metrics", {}).get("details", {})),
            }
            writer.writerow(flat_row)

    mlflow.log_artifact(output_path)
    logger.info(f"Cost summary CSV logged to MLflow: {output_path}")
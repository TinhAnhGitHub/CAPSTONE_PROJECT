"""
MLflow GenAI Agent Versioning for VideoDeepSearch Team.

Provides manual version control for the team agent - call when you want to
create a new version snapshot before running evaluations.

Usage:
    from videodeepsearch.tracing.agent_versioning import (
        setup_versioned_agent,
        get_version_model_id,
        log_agent_run,
    )

    # Before running the team
    version_ctx = setup_versioned_agent(team, settings)

    # All traces automatically linked to this version
    async for chunk in team.arun(...):
        ...

    # Get model_id for evaluation
    model_id = get_version_model_id(version_ctx)
"""

import subprocess
from pathlib import Path
from typing import Any, Optional

import mlflow
from loguru import logger


def get_git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(status)

        return {
            "commit": commit,
            "commit_short": commit[:8],
            "branch": branch,
            "dirty": dirty,
        }
    except subprocess.CalledProcessError:
        return {
            "commit": "unknown",
            "commit_short": "local",
            "branch": "unknown",
            "dirty": True,
        }


def compute_agent_code_hash() -> str:
    import hashlib

    agent_paths = [
        Path("src/videodeepsearch/agent"),
        Path("src/videodeepsearch/toolkit"),
        Path("src/videodeepsearch/tracing"),
        Path("src/videodeepsearch/schema"),
        Path("src/videodeepsearch/clients"),
    ]

    hasher = hashlib.sha256()

    for agent_path in agent_paths:
        if not agent_path.exists():
            continue
        if agent_path.is_dir():
            for py_file in sorted(agent_path.rglob("*.py")):
                hasher.update(py_file.read_bytes())
        elif agent_path.is_file():
            hasher.update(agent_path.read_bytes())

    return hasher.hexdigest()[:16]


class AgentVersionContext:
    def __init__(
        self,
        model_id: str,
        version_name: str,
        git_info: dict,
        agent_code_hash: str,
    ):
        self.model_id = model_id
        self.version_name = version_name
        self.git_info = git_info
        self.agent_code_hash = agent_code_hash

    def __repr__(self) -> str:
        return (
            f"AgentVersionContext("
            f"version={self.version_name}, "
            f"model_id={self.model_id}, "
            f"git={self.git_info['commit_short']}, "
            f"dirty={self.git_info['dirty']})"
        )


def setup_versioned_agent(
    team: Any,
    settings: Any,
    agent_name: str = "video-search-team",
    version_tag: Optional[str] = None,
    log_config: bool = True,
) -> AgentVersionContext:
    """
    Manually set up MLflow versioning for the team agent.

    Creates a LoggedModel with version name based on git state + optional tag.
    All subsequent traces will automatically link to this version.

    Args:
        team: The Agno team instance
        settings: Settings/configuration object
        agent_name: Base name for the agent (e.g., "video-search-team")
        version_tag: Optional custom tag to append (e.g., "v1", "experiment-a")
        log_config: Whether to log agent configuration to the LoggedModel

    Returns:
        AgentVersionContext with model_id for evaluation and tracking
    """
    git_info = get_git_info()
    agent_code_hash = compute_agent_code_hash()

    version_parts = [agent_name, git_info["commit_short"]]
    if version_tag:
        version_parts.append(version_tag)
    if git_info["dirty"]:
        version_parts.append("dirty")

    version_name = "-".join(version_parts)

    context = mlflow.set_active_model(name=version_name)
    model_id = context.model_id

    logger.info(f"Agent version set: {version_name} (model_id: {model_id})")

    if log_config:
        _log_agent_config(team, settings, git_info, agent_code_hash)

    return AgentVersionContext(
        model_id=model_id,
        version_name=version_name,
        git_info=git_info,
        agent_code_hash=agent_code_hash,
    )


def _log_agent_config(
    team: Any,
    settings: Any,
    git_info: dict,
    agent_code_hash: str,
) -> None:
    """Log team configuration to the active LoggedModel."""
    mlflow.log_params({
        "git/commit": git_info["commit_short"],
        "git/branch": git_info["branch"],
        "git/dirty": git_info["dirty"],
        "agent/code_hash": agent_code_hash,
    })

    if hasattr(team, 'models') and team.models:
        for agent_role, model in team.models.items():
            model_id = getattr(model, 'id', 'unknown')
            provider = getattr(model, 'provider', 'unknown')
            mlflow.log_params({
                f"model/{agent_role}/id": model_id,
                f"model/{agent_role}/provider": provider,
            })

    if hasattr(team, 'to_dict'):
        try:
            team_config = team.to_dict()
            mlflow.log_dict(team_config, "agent/team_config.json")
        except Exception as e:
            logger.warning(f"Could not log team config: {e}")

    if hasattr(settings, 'model_dump'):
        try:
            settings_dict = settings.model_dump()
            mlflow.log_dict(settings_dict, "config/settings_snapshot.yaml")
        except Exception as e:
            logger.warning(f"Could not log settings: {e}")

    if git_info["dirty"]:
        try:
            diff = subprocess.check_output(
                ["git", "diff"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            if diff:
                mlflow.log_text(diff, "git/uncommitted_diff.patch")
                logger.info("Logged uncommitted changes as diff patch")
        except subprocess.CalledProcessError:
            pass


def get_version_model_id(context: AgentVersionContext) -> str:
    """Get the LoggedModel ID for evaluation."""
    return context.model_id


def log_agent_run(
    context: AgentVersionContext,
    session_metrics: Any,
    tool_stats: Optional[dict] = None,
    duration_seconds: Optional[float] = None,
) -> None:
    """
    Log run metrics to the versioned agent's LoggedModel.

    Call this after running the team to capture metrics for the version.

    Args:
        context: AgentVersionContext from setup_versioned_agent
        session_metrics: SessionMetrics from team.aget_session_metrics()
        tool_stats: Tool call statistics dict
        duration_seconds: Run duration in seconds
    """
    if session_metrics:
        mlflow.log_metrics({
            "run/input_tokens": getattr(session_metrics, 'input_tokens', 0),
            "run/output_tokens": getattr(session_metrics, 'output_tokens', 0),
            "run/total_tokens": getattr(session_metrics, 'total_tokens', 0),
        })

        cost = getattr(session_metrics, 'cost', None)
        if cost is not None:
            mlflow.log_metric("run/cost", cost)

    # Log tool statistics
    if tool_stats:
        mlflow.log_metrics({
            "run/total_tool_calls": tool_stats.get("total_calls", 0),
            "run/successful_tool_calls": tool_stats.get("successful_calls", 0),
            "run/failed_tool_calls": tool_stats.get("failed_calls", 0),
        })

        mlflow.log_dict(tool_stats, "run/tool_statistics.json")

    # Log duration
    if duration_seconds:
        mlflow.log_metric("run/duration_seconds", duration_seconds)

    logger.info(f"Logged run metrics to version {context.version_name}")


def get_logged_model_info(model_id: str) -> dict:
    """
    Retrieve LoggedModel metadata including git info.

    Args:
        model_id: The LoggedModel ID

    Returns:
        Dict with model metadata
    """
    logged_model = mlflow.get_logged_model(model_id)

    info = {
        "model_id": logged_model.model_id,
        "name": logged_model.name,
        "creation_timestamp": logged_model.creation_timestamp,
        "tags": dict(logged_model.tags),
    }

    # Extract git tags
    git_tags = {
        k.replace("mlflow.git.", ""): v
        for k, v in logged_model.tags.items()
        if k.startswith("mlflow.git")
    }
    info["git"] = git_tags

    return info


def search_traces_by_version(model_id: str) -> list:
    """
    Search all traces linked to a specific agent version.

    Args:
        model_id: The LoggedModel ID

    Returns:
        List of traces for this version
    """
    traces = mlflow.search_traces(model_id=model_id)
    return traces
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlflow
from loguru import logger

from videodeepsearch.evaluation.runners.print_agno import print_run_event
from videodeepsearch.evaluation.datasets.schema import (
    EvalRecord,
    EvalTrace,
    ToolCallTrace,
    TokenUsage,
)
from videodeepsearch.evaluation.util.prepare_team import (
    return_team,
    initialize_clients,
    cleanup_clients,
    setup_models,
)
from videodeepsearch.core.settings import Settings, load_settings


async def run_single_record(
    record: EvalRecord,
    settings: Settings | None = None,
    models: dict[str, Any] | None = None,
    worker_models: dict[str, Any] | None = None,
    clients: dict[str, Any] | None = None,
    mlflow_run_id: str | None = None,
    verbose: bool = False,
) -> EvalTrace:
    """Run agent on a single evaluation record and collect trace.

    Args:
        record: Evaluation record with inputs
        settings: Settings object (loaded if None)
        models: Pre-initialized models (initialized if None)
        worker_models: Pre-initialized worker models
        clients: Pre-initialized clients (initialized if None)
        mlflow_run_id: Optional MLflow run ID to log under
        verbose: Print events during execution

    Returns:
        EvalTrace with captured tool calls, response, and metrics
    """
    if settings is None:
        settings = load_settings()

    clients_provided = clients is not None
    if clients is None:
        clients = await initialize_clients(settings)

    if models is None or worker_models is None:
        models, worker_models = setup_models(settings)

    session_id = record.inputs.session_id or str(uuid4())

    video_ids = record.inputs.total_video_haystack_ids

    team = await return_team(
        user_id=record.inputs.user_id,
        session_id=session_id,
        video_ids=video_ids,
        user_demand=record.inputs.user_demand,
    )

    initial_session_state: dict[str, Any] = {
        "list_video_ids": video_ids,
        "user_demand": record.inputs.user_demand,
    }

    events: list[Any] = []
    tool_calls: list[ToolCallTrace] = []
    final_response = ""
    start_time = time.time()

    try:
        async for chunk in team.arun(
            input=record.inputs.user_demand,
            session_state=initial_session_state,
            stream=True,
            stream_events=True,
        ):
            events.append(chunk)

            ev = getattr(chunk, "event", None) or type(chunk).__name__

            if ev in ("ToolCallCompleted", "TeamToolCallCompleted"):
                t = getattr(chunk, "tool", None)
                if t:
                    tool_name = getattr(t, "tool_name", "?")
                    tool_args = getattr(t, "tool_args", {})
                    tool_result = getattr(t, "result", None)

                    tool_calls.append(ToolCallTrace(
                        name=tool_name,
                        input_parameters=tool_args if tool_args else None,
                        output=tool_result,
                    ))

            if ev in ("RunContent", "TeamRunContent"):
                content = getattr(chunk, "content", None)
                if content:
                    final_response += str(content)

            if ev in ("RunCompleted", "TeamRunCompleted"):
                content = getattr(chunk, "content", None)
                if content and not final_response:
                    final_response = str(content)

            if verbose:
                logger.info(f"Event: {ev}")

        session_metrics = await team.aget_session_metrics()

        duration_seconds = time.time() - start_time

        token_usage: dict[str, TokenUsage] = {}
        if session_metrics:
            token_usage["total"] = TokenUsage(
                input_tokens=session_metrics.input_tokens or 0,
                output_tokens=session_metrics.output_tokens or 0,
                total_tokens=session_metrics.total_tokens or 0,
            )

        trace = EvalTrace(
            session_id=session_id,
            record=record,
            input=record.inputs.user_demand,
            actual_output=final_response,
            tool_trace=tool_calls,
            token_usage=token_usage,
            duration_seconds=duration_seconds,
            success=True,
            error_msg=None,
        )

        if mlflow_run_id:
            mlflow.log_metric("duration_seconds", duration_seconds)
            mlflow.log_metric("input_tokens", session_metrics.input_tokens or 0)
            mlflow.log_metric("output_tokens", session_metrics.output_tokens or 0)
            mlflow.log_metric("total_tokens", session_metrics.total_tokens or 0)
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("num_tool_calls", len(tool_calls))

    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        duration_seconds = time.time() - start_time

        trace = EvalTrace(
            session_id=session_id,
            record=record,
            input=record.inputs.user_demand,
            actual_output=f"Error: {str(e)}",
            tool_trace=tool_calls,
            token_usage={},
            duration_seconds=duration_seconds,
            success=False,
            error_msg=str(e),
        )

    if not clients_provided:
        await cleanup_clients(clients)

    return trace


async def run_test_single_record(
    record: EvalRecord,
    dataset_name: str,
    output_log_dir: Path,
    mlflow_experiment_name: str,
    mlflow_tracking_uri: str,
    settings: Settings | None = None,
    verbose: bool = True,
) -> EvalTrace:
    """Run agent on a single test record with output logging.

    Args:
        record: Evaluation record with inputs
        dataset_name: Name of the dataset (used for output folder)
        output_log_dir: Base directory for output logs
        mlflow_experiment_name: MLflow experiment name
        mlflow_tracking_uri: MLflow tracking URI
        settings: Settings object (loaded if None)
        verbose: Print events during execution

    Returns:
        EvalTrace with captured events, tool calls, response, and metrics

    Output format:
        <output_log_dir>/<dataset_name>/<run_id>.json
        Contains a list of output event JSON objects
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.agno.autolog()  # type: ignore

    if settings is None:
        settings = load_settings()

    clients = await initialize_clients(settings)
    session_id = record.inputs.session_id or str(uuid4())

    with mlflow.start_run(run_name=f"test-evaluation-{datetime.now().isoformat()}") as run:
        mlflow_run_id = run.info.run_id

        video_ids = record.inputs.total_video_haystack_ids

        team = await return_team(
            user_id=record.inputs.user_id,
            session_id=session_id,
            video_ids=video_ids,
            user_demand=record.inputs.user_demand,
        )

        initial_session_state: dict[str, Any] = {
            "list_video_ids": video_ids,
            "user_demand": record.inputs.user_demand,
        }

        events: list[dict[str, Any]] = []
        tool_calls: list[ToolCallTrace] = []
        final_response = ""
        start_time = time.time()

        try:
            async for chunk in team.arun(
                input=record.inputs.user_demand,
                session_state=initial_session_state,
                stream=True,
                stream_events=True,
            ):
                ev = getattr(chunk, "event", None) or type(chunk).__name__

                event_dict = {
                    "event_type": ev,
                    "timestamp": datetime.now().isoformat(),
                }

                if hasattr(chunk, "content"):
                    event_dict["content"] = str(chunk.content)
                if hasattr(chunk, "tool"):
                    t = chunk.tool
                    event_dict["tool_name"] = getattr(t, "tool_name", "?")
                    event_dict["tool_args"] = getattr(t, "tool_args", {})
                    event_dict["tool_result"] = str(getattr(t, "result", None))

                events.append(event_dict)

                if ev in ("ToolCallCompleted", "TeamToolCallCompleted"):
                    t = getattr(chunk, "tool", None)
                    if t:
                        tool_name = getattr(t, "tool_name", "?")
                        tool_args = getattr(t, "tool_args", {})
                        tool_result = getattr(t, "result", None)

                        tool_calls.append(
                            ToolCallTrace(
                                name=tool_name,
                                input_parameters=tool_args if tool_args else None,
                                output=tool_result,
                            )
                        )

                if ev in ("RunContent", "TeamRunContent"):
                    content = getattr(chunk, "content", None)
                    if content:
                        final_response += str(content)

                if ev in ("RunCompleted", "TeamRunCompleted"):
                    content = getattr(chunk, "content", None)
                    if content and not final_response:
                        final_response = str(content)

                if verbose:
                    print_run_event(chunk)

            session_metrics = await team.aget_session_metrics()
            duration_seconds = time.time() - start_time

            token_usage: dict[str, TokenUsage] = {}
            if session_metrics:
                token_usage["total"] = TokenUsage(
                    input_tokens=session_metrics.input_tokens or 0,
                    output_tokens=session_metrics.output_tokens or 0,
                    total_tokens=session_metrics.total_tokens or 0,
                )

            trace = EvalTrace(
                session_id=session_id,
                record=record,
                input=record.inputs.user_demand,
                actual_output=final_response,
                tool_trace=tool_calls,
                token_usage=token_usage,
                duration_seconds=duration_seconds,
                success=True,
                error_msg=None,
            )

            mlflow.log_metric("duration_seconds", duration_seconds)
            mlflow.log_metric("input_tokens", session_metrics.input_tokens or 0)
            mlflow.log_metric("output_tokens", session_metrics.output_tokens or 0)
            mlflow.log_metric("total_tokens", session_metrics.total_tokens or 0)
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("num_tool_calls", len(tool_calls))
            mlflow.log_param("num_events", len(events))
            mlflow.log_param("ground_truth_video_ids", record.inputs.ground_truth_video_ids)

        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            duration_seconds = time.time() - start_time

            trace = EvalTrace(
                session_id=session_id,
                record=record,
                input=record.inputs.user_demand,
                actual_output=f"Error: {str(e)}",
                tool_trace=tool_calls,
                token_usage={},
                duration_seconds=duration_seconds,
                success=False,
                error_msg=str(e),
            )

            mlflow.log_metric("duration_seconds", duration_seconds)
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("error", str(e))

        output_dir = output_log_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{mlflow_run_id}.json"
        with open(output_file, "w") as f:
            json.dump(events, f, indent=2, default=str)

        logger.info(f"Events logged to: {output_file}")

    await cleanup_clients(clients)

    return trace


async def run_dataset(
    records: list[EvalRecord],
    settings: Settings | None = None,
    mlflow_experiment_name: str = "videodeepsearch-evaluation",
    mlflow_tracking_uri: str | None = None,
    verbose: bool = False,
) -> list[EvalTrace]:
    """Run agent on multiple evaluation records.

    Runs all records under a single MLflow run for aggregation.

    Args:
        records: List of evaluation records
        settings: Settings object (loaded if None)
        mlflow_experiment_name: MLflow experiment name
        mlflow_tracking_uri: MLflow tracking URI (uses env if None)
        verbose: Print events during execution

    Returns:
        List of EvalTrace objects
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.agno.autolog()  # type: ignore

    if settings is None:
        settings = load_settings()

    clients = await initialize_clients(settings)
    models, worker_models = setup_models(settings)

    traces: list[EvalTrace] = []

    with mlflow.start_run(run_name=f"evaluation-{datetime.now().isoformat()}"):
        mlflow.log_param("num_records", len(records))

        for i, record in enumerate(records):
            logger.info(f"Running record {i + 1}/{len(records)}")

            trace = await run_single_record(
                record=record,
                settings=settings,
                models=models,
                worker_models=worker_models,
                clients=clients,
                mlflow_run_id=mlflow.active_run().info.run_id,
                verbose=verbose,
            )

            traces.append(trace)

            mlflow.log_metric(f"record_{i}_duration", trace.duration_seconds)
            mlflow.log_metric(f"record_{i}_tool_calls", len(trace.tool_trace))
            mlflow.log_metric(f"record_{i}_success", 1 if trace.success else 0)

    await cleanup_clients(clients)

    return traces


async def run_evaluation_cycle(
    dataset: list[EvalRecord] | Path | str,
    scorers: list[Any] | None = None,
    scorer_preset: str = "ground_truth_free",
    judge_model_id: str = "gpt-4o-mini",
    mlflow_experiment_name: str = "videodeepsearch-evaluation",
    mlflow_tracking_uri: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Complete evaluation cycle: load dataset → run agent → evaluate.

    This is the main API for notebook usage.

    Args:
        dataset: List of EvalRecord, or path to JSON/YAML dataset file
        scorers: List of scorer instances (uses preset if None)
        scorer_preset: Preset name for scorers if scorers not provided
        judge_model_id: Judge model for scorers
        mlflow_experiment_name: MLflow experiment name
        mlflow_tracking_uri: MLflow tracking URI
        verbose: Print events during execution

    Returns:
        Dictionary with traces and evaluation results
    """
    from videodeepsearch.evaluation.datasets.loaders import load_dataset
    from videodeepsearch.evaluation.scorers.presets import get_scorers_by_preset

    # Load dataset if path provided
    if isinstance(dataset, (Path, str)):
        builder = load_dataset(dataset)
        records = builder.get_records()
    else:
        records = dataset

    # Get scorers
    if scorers is None:
        scorers = get_scorers_by_preset(
            preset=scorer_preset,
            judge_model_id=judge_model_id,
        )

    # Run agent on dataset
    traces = await run_dataset(
        records=records,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        verbose=verbose,
    )

    # Create MLflow dataset for evaluation
    import mlflow.genai.datasets  # type: ignore

    mlflow_dataset = mlflow.genai.datasets.create_dataset(
        name=f"eval-{datetime.now().isoformat()}",
    )

    # Convert traces to MLflow format
    mlflow_records = []
    for trace in traces:
        mlflow_record = {
            "inputs": {
                "user_id": trace.record.inputs.user_id,
                "total_video_haystack_ids": trace.record.inputs.total_video_haystack_ids,
                "user_demand": trace.input,
            },
            "outputs": trace.actual_output,
            "trace": trace,  # MLflow expects trace for ground-truth free scorers
        }

        # Add expectations if available
        if trace.record.expectations:
            expectations = {}
            if trace.record.expectations.expected_response:
                expectations["expected_response"] = trace.record.expectations.expected_response
            if trace.record.expectations.expected_facts:
                expectations["expected_facts"] = [
                    {"fact": f.fact, **({"source": f.source} if f.source else {})}
                    for f in trace.record.expectations.expected_facts
                ]
            if expectations:
                mlflow_record["expectations"] = expectations

        mlflow_records.append(mlflow_record)

    if mlflow_records:
        mlflow_dataset.merge_records(mlflow_records)

    # Run evaluation with scorers
    eval_result = mlflow.genai.evaluate(
        data=mlflow_dataset,
        scorers=scorers,
    )

    return {
        "traces": traces,
        "eval_result": eval_result,
        "dataset": mlflow_dataset,
        "scorers": scorers,
    }


__all__ = [
    "run_single_record",
    "run_test_single_record",
    "run_dataset",
    "run_evaluation_cycle",
]

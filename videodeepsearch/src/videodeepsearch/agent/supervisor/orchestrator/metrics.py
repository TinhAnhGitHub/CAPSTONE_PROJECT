from typing import cast
from loguru import logger
from agno.hooks import hook
from agno.run.team import TeamRunOutput
from agno.metrics import RunMetrics, ModelMetrics
from agno.team import Team


@hook   
def aggregate_worker_metrics(
    run_output: TeamRunOutput,
    team: Team  
):
    """Post hook to aggregate worker metrics into team session metrics.
    
    Reads worker metrics from session_state and:
    1. Adds totals to team RunMetrics
    2. Appends ModelMetrics to details dict for per-model breakdown
    """
    
    worker_metrics_list = run_output.session_state.get('worker_metrics', [])
    
    if not worker_metrics_list:
        return  
    
    total_input_tokens = sum(w.get('input_tokens', 0) for w in worker_metrics_list)
    total_output_tokens = sum(w.get('output_tokens', 0) for w in worker_metrics_list)
    total_tokens = sum(w.get('total_tokens', 0) for w in worker_metrics_list)
    total_cost = sum(w.get('cost', 0.0) for w in worker_metrics_list)
    
    team_metrics: RunMetrics = cast(RunMetrics, run_output.metrics)
    
    team_metrics.input_tokens = (team_metrics.input_tokens or 0) + total_input_tokens
    team_metrics.output_tokens = (team_metrics.output_tokens or 0) + total_output_tokens
    team_metrics.total_tokens = (team_metrics.total_tokens or 0) + total_tokens
    team_metrics.cost = (team_metrics.cost or 0.0) + total_cost
    
    if team_metrics.details is None:
        team_metrics.details = {}
    
    if "model" not in team_metrics.details:
        team_metrics.details["model"] = []
    
    existing_model_metrics: list[ModelMetrics] = team_metrics.details["model"]
    
    for w in worker_metrics_list:
        worker_model_metrics = ModelMetrics(
            id=w.get('model_id', ''),
            provider=w.get('model_provider', ''),
            input_tokens=w.get('input_tokens', 0),
            output_tokens=w.get('output_tokens', 0),
            total_tokens=w.get('total_tokens', 0),
            cost=w.get('cost', 0.0),
        )
        
        found = False
        for existing in existing_model_metrics:
            if existing.id == worker_model_metrics.id and existing.provider == worker_model_metrics.provider:
                existing.accumulate(worker_model_metrics)
                found = True
                break
        
        if not found:
            existing_model_metrics.append(worker_model_metrics)
    
    logger.info(
        f"[MetricsHook] Aggregated {len(worker_metrics_list)} worker metrics: "
        f"+{total_tokens} tokens, +${total_cost:.4f} | "
        f"Model details: {len(existing_model_metrics)} unique models"
    )

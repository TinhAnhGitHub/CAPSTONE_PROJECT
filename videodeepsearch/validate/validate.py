from typing import Any
from pathlib import Path
import os
import asyncio
import uuid
import mlflow
from uuid import uuid4
import yaml
from datetime import datetime
from io import StringIO
from dotenv import load_dotenv
from loguru import logger

from rich.console import Console
from rich.text import Text
from rich.style import Style
from rich.markup import escape

from agno.models.openrouter import OpenRouterResponses
from agno.db.postgres import AsyncPostgresDb

from videodeepsearch.agent.team import build_video_search_team
from videodeepsearch.agent.supervisor.orchestrator.spawn_toolkit import WorkerModel
from videodeepsearch.core.settings import Settings, load_settings

from videodeepsearch.clients.storage.postgre.client import PostgresClient
from videodeepsearch.clients.storage.minio.client import MinioStorageClient
from videodeepsearch.clients.storage.qdrant.image_client import ImageQdrantClient
from videodeepsearch.clients.storage.qdrant.segment_client import SegmentQdrantClient
from videodeepsearch.clients.storage.qdrant.audio_client import AudioQdrantClient
from videodeepsearch.clients.storage.elasticsearch.client import ElasticsearchOCRClient
from videodeepsearch.clients.storage.elasticsearch.schema import ElasticsearchConfig
from videodeepsearch.clients.inference.client import QwenVLEmbeddingClient, MMBertClient, SpladeClient
from videodeepsearch.clients.inference.schema import QwenVLEmbeddingConfig, MMBertConfig, SpladeConfig

from arango import ArangoClient #type:ignore


import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "test" / "test_notebook_agno"))
from print_agno import print_run_event, _stream_flush

# Import manual MLflow cost logger
from mlflow_cost_logger import ToolCallTracker, log_session_metrics, log_tool_statistics
load_dotenv()



mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://100.113.186.28:5000"))
mlflow.set_experiment("videodeepsearch-agent-validation2")



mlflow.agno.autolog()
logger.info(f"MLflow tracing enabled: {os.getenv('MLFLOW_TRACKING_URI', 'http://100.113.186.28:5000')}")
# ---------------------------------------------------------------------------
# Dual Output: Terminal + File
# ---------------------------------------------------------------------------

class DualOutput:
    """Captures rich console output and writes to both terminal and file."""
    
    def __init__(self, file_handle):
        self.file = file_handle
        self.terminal_console = Console()
        self.file_console = Console(file=self.file, force_terminal=False, width=120)
        self._stream_buffer = ""
    
    def write_event(self, event: Any, verbose: bool = True, show_tool_results: bool = True) -> None:
        """Print event to both terminal (rich) and file (plain text)."""
        # Print to terminal using the original print_run_event
        # print_run_event(event, verbose=verbose, show_tool_results=show_tool_results)
        
        # Write simplified version to file
        self._write_to_file(event)
    
    def _write_to_file(self, event: Any) -> None:
        """Write event to file in plain text format."""
        ev = getattr(event, "event", None) or type(event).__name__
        timestamp = datetime.now().isoformat()
        
        # Extract metadata
        name = (
            getattr(event, "team_name", None)
            or getattr(event, "team_id", None)
            or getattr(event, "agent_name", None)
            or getattr(event, "agent_id", None)
            or ""
        )
        run_id = getattr(event, "run_id", "")
        meta = f"{name} · run={run_id}" if name or run_id else ""
        
        # Handle streaming content
        if ev in ("RunContent", "TeamRunContent"):
            content = getattr(event, "content", None)
            if content:
                self.file.write(str(content))
                self.file.flush()
            return
        
        if ev in ("RunIntermediateContent", "TeamRunIntermediateContent"):
            content = getattr(event, "content", None)
            if content:
                self.file.write(str(content))
                self.file.flush()
            return
        
        if ev in ("RunContentCompleted", "TeamRunContentCompleted"):
            self.file.write("\n")
            self.file.flush()
            return
        
        # Handle other events with formatted output
        prefix = "TEAM " if ev.startswith("Team") else ""
        display_ev = ev[4:] if ev.startswith("Team") else ev
        
        self.file.write(f"\n[{timestamp}] ")
        
        # Run lifecycle events
        if ev in ("RunStarted", "TeamRunStarted"):
            self.file.write(f"▶ {prefix}RUN STARTED\n")
            if meta:
                self.file.write(f"  {meta}\n")
            model = getattr(event, "model", "")
            provider = getattr(event, "model_provider", "")
            if model or provider:
                self.file.write(f"  model: {provider}/{model}\n")
        
        elif ev in ("RunCompleted", "TeamRunCompleted"):
            self.file.write(f"✔ {prefix}RUN COMPLETED\n")
            if meta:
                self.file.write(f"  {meta}\n")
            metrics = getattr(event, "metrics", None)
            if metrics:
                in_tok = getattr(metrics, "input_tokens", "-")
                out_tok = getattr(metrics, "output_tokens", "-")
                self.file.write(f"  tokens: in={in_tok} out={out_tok}\n")
            followups = getattr(event, "followups", None)
            if followups:
                self.file.write("  Followups:\n")
                for f in followups:
                    self.file.write(f"    • {f}\n")
        
        elif ev in ("RunError", "TeamRunError"):
            self.file.write(f"✘ {prefix}RUN ERROR\n")
            err_type = getattr(event, "error_type", "")
            msg = getattr(event, "content", "")
            self.file.write(f"  type: {err_type}\n")
            self.file.write(f"  message: {msg}\n")
        
        elif ev in ("RunCancelled", "TeamRunCancelled"):
            self.file.write(f"⊘ {prefix}RUN CANCELLED\n")
            reason = getattr(event, "reason", "")
            self.file.write(f"  reason: {reason}\n")
        
        elif ev in ("RunPaused", "TeamRunPaused"):
            self.file.write(f"⏸ {prefix}RUN PAUSED\n")
        
        # Tool calls
        elif ev in ("ToolCallStarted", "TeamToolCallStarted"):
            t = getattr(event, "tool", None)
            tool_name = getattr(t, "tool_name", "?") if t else "?"
            tool_args = getattr(t, "tool_args", {}) if t else {}
            self.file.write(f"⚙ {prefix}tool → {tool_name}\n")
            if tool_args:
                self.file.write(f"  args: {tool_args}\n")
        
        elif ev in ("ToolCallCompleted", "TeamToolCallCompleted"):
            t = getattr(event, "tool", None)
            tool_name = getattr(t, "tool_name", "?") if t else "?"
            result = getattr(t, "result", None) if t else getattr(event, "content", None)
            tool_error = getattr(t, "tool_call_error", None) if t else None
            
            status = "ERROR" if tool_error else "OK"
            self.file.write(f"✓ {prefix}tool ← {tool_name} [{status}]\n")
            if result:
                # Try to format as JSON if possible
                try:
                    import json
                    parsed = json.loads(result) if isinstance(result, str) else result
                    self.file.write(f"  result: {json.dumps(parsed, indent=2, ensure_ascii=False)}\n")
                except:
                    self.file.write(f"  result: {result}\n")
        
        elif ev in ("ToolCallError", "TeamToolCallError"):
            t = getattr(event, "tool", None)
            tool_name = getattr(t, "tool_name", "?") if t else "?"
            err = getattr(event, "error", "")
            self.file.write(f"✘ {prefix}tool error ← {tool_name}\n")
            self.file.write(f"  error: {err}\n")
        
        # Reasoning
        elif ev in ("ReasoningStarted", "TeamReasoningStarted"):
            self.file.write(f"🧠 {prefix}reasoning ...\n")
        
        elif ev in ("ReasoningContentDelta", "TeamReasoningContentDelta"):
            delta = getattr(event, "reasoning_content", "")
            if delta:
                self.file.write(delta)
                self.file.flush()
        
        elif ev in ("ReasoningCompleted", "TeamReasoningCompleted"):
            self.file.write(f"\n🧠 {prefix}reasoning done\n")
        
        # Memory
        elif ev in ("MemoryUpdateStarted", "TeamMemoryUpdateStarted"):
            self.file.write(f"💾 {prefix}updating memory ...\n")
        
        elif ev in ("MemoryUpdateCompleted", "TeamMemoryUpdateCompleted"):
            memories = getattr(event, "memories", [])
            n = len(memories) if memories else 0
            self.file.write(f"💾 {prefix}memory updated ({n} entries)\n")
        
        # Model requests
        elif ev in ("ModelRequestStarted", "TeamModelRequestStarted"):
            model = getattr(event, "model", "")
            provider = getattr(event, "model_provider", "")
            self.file.write(f"→ {prefix}model request [{provider}/{model}]\n")
        
        elif ev in ("ModelRequestCompleted", "TeamModelRequestCompleted"):
            in_tok = getattr(event, "input_tokens", "-")
            out_tok = getattr(event, "output_tokens", "-")
            self.file.write(f"← {prefix}model done [in={in_tok} out={out_tok}]\n")
        
        # Task events
        elif ev == "TeamTaskIterationStarted":
            iteration = getattr(event, "iteration", 0)
            max_iter = getattr(event, "max_iterations", 0)
            self.file.write(f"📋 task iteration {iteration + 1}/{max_iter}\n")
        
        elif ev == "TeamTaskIterationCompleted":
            iteration = getattr(event, "iteration", 0)
            max_iter = getattr(event, "max_iterations", 0)
            self.file.write(f"📋 task iteration {iteration + 1}/{max_iter} completed\n")
        
        elif ev == "TeamTaskStateUpdated":
            tasks = getattr(event, "tasks", []) or []
            goal_complete = getattr(event, "goal_complete", False)
            self.file.write(f"📋 task state updated ({len(tasks)} tasks)\n")
            if goal_complete:
                self.file.write("  ✅ Goal complete!\n")
        
        elif ev == "TeamTaskCreated":
            title = getattr(event, "title", "Untitled")
            status = getattr(event, "status", "pending")
            self.file.write(f"📋 task created: {title} [{status}]\n")
        
        elif ev == "TeamTaskUpdated":
            title = getattr(event, "title", "Untitled")
            status = getattr(event, "status", "")
            prev_status = getattr(event, "previous_status", None)
            status_change = f" {prev_status} → {status}" if prev_status else f" [{status}]"
            self.file.write(f"📋 task updated: {title}{status_change}\n")
        
        # Compression
        elif ev in ("CompressionStarted", "TeamCompressionStarted"):
            self.file.write(f"🗜 {prefix}compressing tool results ...\n")
        
        elif ev in ("CompressionCompleted", "TeamCompressionCompleted"):
            orig = getattr(event, "original_size", 0)
            comp = getattr(event, "compressed_size", 0)
            pct = round((1 - comp / orig) * 100) if orig else 0
            n = getattr(event, "tool_results_compressed", "?")
            self.file.write(f"🗜 {prefix}compressed {n} results: {orig} → {comp} chars (↓{pct}%)\n")
        
        # Session summary
        elif ev in ("SessionSummaryStarted", "TeamSessionSummaryStarted"):
            self.file.write(f"📝 {prefix}summarising session ...\n")
        
        elif ev in ("SessionSummaryCompleted", "TeamSessionSummaryCompleted"):
            self.file.write(f"📝 {prefix}session summary ready\n")
        
        else:
            self.file.write(f"? {prefix}{display_ev}\n")
        
        self.file.flush()


# ---------------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------------

def _build_model_kwargs(cfg) -> dict:
    """Build model kwargs from config."""
    return {
        "id": cfg.model_id,
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    }


def setup_models(settings: Settings) -> tuple[dict[str, OpenRouterResponses], dict[str, WorkerModel]]:
    """Setup agent models and worker models from settings."""
    llm_cfg = settings.llm_provider

    models: dict[str, OpenRouterResponses] = {}

    greeter_cfg = llm_cfg.agents.greeter
    models["greeter"] = OpenRouterResponses(**_build_model_kwargs(greeter_cfg), )

    orchestrator_cfg = llm_cfg.agents.orchestrator
    models["orchestrator"] = OpenRouterResponses(**_build_model_kwargs(orchestrator_cfg),  )

    planning_cfg = llm_cfg.agents.planning
    models["planning"] = OpenRouterResponses(**_build_model_kwargs(planning_cfg),  )

    llm_tool_cfg = llm_cfg.agents.llm_tool
    if llm_tool_cfg:
        models["llm_tool"] = OpenRouterResponses(**_build_model_kwargs(llm_tool_cfg),  )
    else:
        models["llm_tool"] = models["planning"]
        logger.info("llm_tool model not configured, using planning model as fallback")

    summarizer_cfg = llm_cfg.agents.summarizer
    if summarizer_cfg:
        models["summarizer"] = OpenRouterResponses(**_build_model_kwargs(summarizer_cfg),  )
    else:
        models["summarizer"] = models["planning"]
        logger.info("summarizer model not configured, using planning model as fallback")

    logger.info(f"Agent models initialized: {list(models.keys())}")

    worker_models: dict[str, WorkerModel] = {}
    for worker_cfg in llm_cfg.workers:
        worker_models[worker_cfg.name] = WorkerModel(
            model=OpenRouterResponses(**_build_model_kwargs(worker_cfg),  ),
            description=worker_cfg.description,
            strengths=worker_cfg.strengths,
        )
    logger.info(f"Worker models initialized: {list(worker_models.keys())}")

    return models, worker_models


# ---------------------------------------------------------------------------
# Client Initialization
# ---------------------------------------------------------------------------

async def initialize_clients(settings: Settings) -> dict[str, Any]:
    """Initialize all required storage and inference clients."""
    clients = {}

    clients["postgres_client"] = PostgresClient(
        database_url=settings.storage.postgres.connection_url
    )
    async with clients["postgres_client"].get_session() as session:
        from sqlalchemy import text
        result = await session.execute(text("SELECT version();"))
        version = result.scalar_one()
        logger.info(f"PostgreSQL connected: {version}")

    clients["agno_db"] = AsyncPostgresDb(
        db_url=settings.storage.postgres.connection_url,
        create_schema=True,
    )
    logger.info("Agno AsyncPostgresDb initialized for session storage")

    clients["minio_client"] = MinioStorageClient(
        host=settings.storage.minio.host,
        port=settings.storage.minio.port,
        access_key=settings.storage.minio.access_key,
        secret_key=settings.storage.minio.secret_key,
        secure=settings.storage.minio.secure,
    )
    logger.info("MinIO client initialized")

    qdrant_cfg = settings.storage.qdrant
    clients["image_qdrant_client"] = ImageQdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        grpc_port=qdrant_cfg.grpc_port,
        prefer_grpc=qdrant_cfg.prefer_grpc,
    )
    clients["segment_qdrant_client"] = SegmentQdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        grpc_port=qdrant_cfg.grpc_port,
        prefer_grpc=qdrant_cfg.prefer_grpc,
    )
    clients["audio_qdrant_client"] = AudioQdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        grpc_port=qdrant_cfg.grpc_port,
        prefer_grpc=qdrant_cfg.prefer_grpc,
    )
    logger.info(f"Qdrant clients initialized: {qdrant_cfg.host}:{qdrant_cfg.port}")

    # Elasticsearch
    es_cfg = settings.storage.elasticsearch
    clients["es_ocr_client"] = ElasticsearchOCRClient(
        config=ElasticsearchConfig(
            host=es_cfg.host,
            port=es_cfg.port,
            user=es_cfg.user,
            password=es_cfg.password,
            use_ssl=es_cfg.use_ssl,
            verify_certs=es_cfg.verify_certs,
            index_name=es_cfg.index_name,
            request_timeout=es_cfg.request_timeout,
        )
    )
    await clients["es_ocr_client"].connect()
    logger.info(f"Elasticsearch connected: {es_cfg.host}:{es_cfg.port}")

    # ArangoDB
    arango_cfg = settings.storage.arangodb
    arango_client = ArangoClient(hosts=arango_cfg.host)
    clients["arango_db"] = arango_client.db(
        arango_cfg.database,
        username=arango_cfg.username,
        password=arango_cfg.password,
    )
    logger.info(f"ArangoDB connected: {arango_cfg.database}")

    # Inference clients
    inf_cfg = settings.inference

    clients["qwenvl_client"] = QwenVLEmbeddingClient(
        config=QwenVLEmbeddingConfig(base_url=inf_cfg.qwenvl.base_url)
    )
    logger.info(f"QwenVL client initialized: {inf_cfg.qwenvl.base_url}")

    clients["mmbert_client"] = MMBertClient(
        config=MMBertConfig(
            base_url=inf_cfg.mmbert.base_url,
            model_name=inf_cfg.mmbert.model_name,
        )
    )
    logger.info(f"MMBert client initialized: {inf_cfg.mmbert.base_url}")

    clients["splade_client"] = SpladeClient(
        config=SpladeConfig(
            url=inf_cfg.splade.url,
            model_name=inf_cfg.splade.model_name,
            timeout=inf_cfg.splade.timeout,
            verbose=inf_cfg.splade.verbose,
            max_batch_size=inf_cfg.splade.max_batch_size,
        )
    )
    logger.info(f"SPLADE client initialized: {inf_cfg.splade.url}")

    return clients


async def cleanup_clients(clients: dict[str, Any]) -> None:
    """Cleanup all clients."""
    logger.info("Cleaning up clients...")

    await clients["image_qdrant_client"].close()
    await clients["segment_qdrant_client"].close()
    await clients["audio_qdrant_client"].close()
    await clients["es_ocr_client"].close()
    await clients["qwenvl_client"].close()
    await clients["mmbert_client"].close()
    clients["splade_client"].close()
    await clients["postgres_client"].close()

    logger.info("Clients cleanup complete")


# ---------------------------------------------------------------------------
# Validation Test
# ---------------------------------------------------------------------------

async def run_validation(
    settings: Settings,
    models: dict[str, OpenRouterResponses],
    worker_models: dict[str, WorkerModel],
    clients: dict[str, Any],
    user_id: str,
    session_id: str,
    video_ids: list[str],
    user_demand: str,
    output_file: Path,
) -> None:
    """Run validation test with MLflow tracing and dual output (terminal + file)."""

    logger.info("=" * 60)
    logger.info("  Building Video Search Team")
    logger.info("=" * 60)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Open file for writing events
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"Validation Log - Session: {session_id}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"User: {user_id}\n")
        f.write(f"Videos: {video_ids}\n")
        f.write(f"Demand: {user_demand}\n")
        f.write("=" * 60 + "\n")
        f.write("  Streaming Events\n")
        f.write("=" * 60 + "\n\n")

        team = build_video_search_team(
            session_id=session_id,
            user_id=user_id,
            list_video_ids=video_ids,
            models=models,
            worker_models=worker_models,
            db=clients["agno_db"],
            image_qdrant_client=clients["image_qdrant_client"],
            segment_qdrant_client=clients["segment_qdrant_client"],
            audio_qdrant_client=clients["audio_qdrant_client"],
            qwenvl_client=clients["qwenvl_client"],
            mmbert_client=clients["mmbert_client"],
            splade_client=clients["splade_client"],
            postgres_client=clients["postgres_client"],
            minio_client=clients["minio_client"],
            es_ocr_client=clients["es_ocr_client"],
            arango_db=clients["arango_db"],
        )

        logger.info(f"Team built: {team.name}")
        logger.info(f"Session: {session_id}")
        logger.info(f"User: {user_id}")
        logger.info(f"Videos: {video_ids}")
        logger.info(f"Demand: {user_demand}")
        logger.info(f"Output file: {output_file}")

        f.write(f"Team built: {team.name}\n\n")

        logger.info("=" * 60)
        logger.info("  Running Team Workflow (traces captured by MLflow)")
        logger.info("=" * 60)

        f.write("=" * 60 + "\n")
        f.write("  Running Team Workflow\n")
        f.write("=" * 60 + "\n\n")

        # Create dual output handler
        dual_output = DualOutput(f)

        # Create tool call tracker for capturing tool statistics
        tool_tracker = ToolCallTracker()

        initial_session_state: dict[str, Any] = {
            "list_video_ids": video_ids,
            "user_demand": user_demand,
        }

        # Track start time for duration
        start_time = datetime.now()

        # Run the team - MLflow automatically captures traces
        # print_run_event handles terminal output, DualOutput writes to file
        async for chunk in team.arun(
            input=user_demand,
            session_state=initial_session_state,
            stream=True,
            stream_events=True,
        ):
            # Track tool calls for statistics
            tool_tracker.track_event(chunk)
            # Print to both terminal (rich) and file (plain text)
            dual_output.write_event(chunk, verbose=False, show_tool_results=True)

        # Calculate duration
        duration_seconds = (datetime.now() - start_time).total_seconds()

        # Get session metrics
        session_metrics = await team.aget_session_metrics()
        print(f"Team metric: {session_metrics}")

        # Manually log metrics to MLflow (autolog misses cost and tool stats)
        log_session_metrics(session_metrics, prefix="team")
        log_tool_statistics(tool_tracker.get_statistics(), prefix="team")
        mlflow.log_metric("team/duration_seconds", duration_seconds)

        # Flush any remaining stream
        _stream_flush()
        f.write("\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("  Validation Complete\n")
        f.write(f"  End Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n")

    logger.success(f"Validation test completed successfully. Log saved to: {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    """Main validation entry point."""
    
    settings = load_settings()
    logger.info("Settings loaded from config/settings.yaml")

    clients = await initialize_clients(settings)

    models, worker_models = setup_models(settings)

    user_id = "tinhanhuser"
    session_id = str(uuid.uuid4())
    video_ids = [
        "0e64f1c0da591ca67f07b7f9",
        "c98019fd17ff4420ea47eee7",
        "3636d10a2ad4787733c9700d",
        "c510fac771767405c891bf64",
        "92ba4b2e27f460945fded9e5",
        "533914541945c2060c128da3",
        "946330031ead69b21354d038",
        "3636d10a2ad4787733c9700d",
        "9b17f473300a5436f0a053be",
        "1e1d300356360ed84020821c",
        "02d242459a690605ee3a8ddf",
        "0f48acd4ac783dfbdee85468",
        "b1abb34af1bc67cb712d5ffb"
    ]
    user_demand = "Could you find some information related to SIMEX stock exchange?"

    # Create output file path with session_id
    output_dir = Path(__file__).parent / "logs"
    output_file = output_dir / f"validation_{session_id}.txt"

    try:
        with mlflow.start_run(run_name=f"video_search_team_validation-{uuid4()}"):
            mlflow.log_param("user_id", user_id)
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("video_ids", str(video_ids))
            mlflow.log_param("user_demand", user_demand)
            mlflow.log_param("num_videos", len(video_ids))
            mlflow.log_param("output_file", str(output_file))

            await run_validation(
                settings=settings,
                models=models,
                worker_models=worker_models,
                clients=clients,
                user_id=user_id,
                session_id=session_id,
                video_ids=video_ids,
                user_demand=user_demand,
                output_file=output_file,
            )

            # Log the output file as an artifact
            mlflow.log_artifact(str(output_file))

            logger.success("MLflow run completed")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise
    finally:
        await cleanup_clients(clients)

    logger.info("=" * 60)
    logger.info("  Validation Complete")
    logger.info("=" * 60)
    logger.info(f"View traces at: {os.getenv('MLFLOW_TRACKING_URI', 'http://100.113.186.28:5000')}")
    logger.info(f"Experiment: videodeepsearch-agent-validation")
    logger.info(f"Output file: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

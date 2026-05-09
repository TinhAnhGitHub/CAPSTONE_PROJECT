import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlflow
from dotenv import load_dotenv

load_dotenv()

from videodeepsearch.core.settings import load_settings
from videodeepsearch.evaluation.datasets import EvalRecord
from videodeepsearch.evaluation.util.prepare_team import (
    cleanup_clients,
    initialize_clients,
    return_team,
)
from videodeepsearch.evaluation.runners.print_agno import print_run_event

    
        

class ValidationLogger:
    def __init__(self, total_log_path: Path):
        self.total_log_path = total_log_path
        self.total_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.file_logger = logging.getLogger("validation_total")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.handlers = []  

        file_handler = logging.FileHandler(total_log_path, mode="a")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)

    def info(self, message: str):
        print(f"[INFO] {message}")
        self.file_logger.info(message)

    def error(self, message: str):
        print(f"[ERROR] {message}")
        self.file_logger.error(message)

    def success(self, message: str):
        print(f"[SUCCESS] {message}")
        self.file_logger.info(f"SUCCESS: {message}")

    def event(self, message: str):
        """Log an event from print_run_event."""
        self.file_logger.info(message)


class RunLogWriter:
    """Writer for individual run logs using print_agno format."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(log_path, "w", encoding="utf-8")
        self._write_header()

    def _write_header(self):
        header = f"""
{'='*80}
AGENT RUN LOG
Session ID: {self.log_path.stem.split('_')[-1]}
Started: {datetime.now().isoformat()}
{'='*80}
        """
        self.file.write(header)
        self.file.flush()

    def write_event(self, event_str: str):
        """Write a formatted event string."""
        self.file.write(event_str + "\n")
        self.file.flush()

    def write_output(self, content: str):
        """Write run output content."""
        self.file.write(f"\n--- OUTPUT ---\n{content}\n--- END OUTPUT ---\n")
        self.file.flush()

    def close(self, success: bool, error_msg: str = None):
        footer = f"""
{'='*80}
RUN COMPLETED
Status: {'SUCCESS' if success else 'FAILED'}
Error: {error_msg if error_msg else 'None'}
Finished: {datetime.now().isoformat()}
{'='*80}
        """
        self.file.write(footer)
        self.file.close()


def get_completed_session_ids(dataset_name: str, output_dir: Path) -> set[str]:
    completed_dir = output_dir / dataset_name
    if not completed_dir.exists():
        return set()

    session_ids = set()
    for log_file in completed_dir.glob("log_run_record_*.log"):
        session_id = log_file.stem.replace("log_run_record_", "")
        session_ids.add(session_id)

    return session_ids


async def run_single_record_with_logging(
    record: EvalRecord,
    dataset_name: str,
    output_dir: Path,
    mlflow_experiment_name: str,
    mlflow_tracking_uri: str,
    validation_logger: ValidationLogger,
    session_id: str,
) -> tuple[bool, str]:
    """Run agent on a single record with logging.

    Returns:
        tuple of (success, error_message)
    """
    run_log_path = output_dir / dataset_name / f"log_run_record_{session_id}.log"
    run_log_writer = RunLogWriter(run_log_path)

    validation_logger.info(f"Starting run for session_id: {session_id}")
    validation_logger.info(f"Log file: {run_log_path}")

    try:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        mlflow.agno.autolog() #type:ignore

        settings = load_settings()
        clients = await initialize_clients(settings)

        video_ids = record.inputs.total_video_haystack_ids

        validation_logger.info(f"Initializing team for {len(video_ids)} videos")

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
        tool_calls: list[dict[str, Any]] = []
        final_response = ""

        with mlflow.start_run(
            run_name=f"validation-{dataset_name}-{session_id}"
        ) as run:
            mlflow_run_id = run.info.run_id

            validation_logger.info(f"MLflow run_id: {mlflow_run_id}")

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
                        tool_calls.append({
                            "name": getattr(t, "tool_name", "?"),
                            "args": getattr(t, "tool_args", {}),
                            "result": getattr(t, "result", None),
                        })

                if ev in ("RunContent", "TeamRunContent"):
                    content = getattr(chunk, "content", None)
                    if content:
                        final_response += str(content)

                if ev in ("RunCompleted", "TeamRunCompleted"):
                    content = getattr(chunk, "content", None)
                    if content and not final_response:
                        final_response = str(content)

                print_run_event(chunk)

                event_str = _format_event_for_log(chunk, ev)
                run_log_writer.write_event(event_str)

            session_metrics = await team.aget_session_metrics()

            mlflow.log_param("session_id", session_id)
            mlflow.log_param("num_tool_calls", len(tool_calls))
            mlflow.log_param("num_events", len(events))
            mlflow.log_param("ground_truth_video_ids", record.inputs.ground_truth_video_ids)

            if session_metrics:
                mlflow.log_metric("input_tokens", session_metrics.input_tokens or 0)
                mlflow.log_metric("output_tokens", session_metrics.output_tokens or 0)
                mlflow.log_metric("total_tokens", session_metrics.total_tokens or 0)

            
            run_log_writer.write_output(final_response)

            events_file = output_dir / dataset_name / f"events_{session_id}.json"
            with open(events_file, "w") as f:
                json.dump(events, f, indent=2, default=str)

            validation_logger.success(
                f"Run completed for session_id: {session_id}"
            )
            validation_logger.success(
                f"Tokens: in={session_metrics.input_tokens or 0}, out={session_metrics.output_tokens or 0}" #type:ignore
            )

        await cleanup_clients(clients)
        run_log_writer.close(success=True)
        return True, None #type:ignore

    except Exception as e:
        validation_logger.error(f"Run failed for session_id: {session_id}: {str(e)}")
        run_log_writer.close(success=False, error_msg=str(e))
        return False, str(e)


def _format_event_for_log(chunk: Any, ev: str) -> str:
    """Format an event for log file writing."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    if ev in ("RunStarted", "TeamRunStarted"):
        model = getattr(chunk, "model", "?")
        return f"[{timestamp}] ▶ RUN STARTED | model={model}"

    elif ev in ("RunContent", "TeamRunContent"):
        content = getattr(chunk, "content", "")
        return f"[{timestamp}] 📝 CONTENT: \n\n{content}\n\n"

    elif ev in ("ToolCallStarted", "TeamToolCallStarted"):
        t = getattr(chunk, "tool", None)
        name = getattr(t, "tool_name", "?") if t else "?"
        return f"[{timestamp}] ⚙ TOOL START: {name}"

    elif ev in ("ToolCallCompleted", "TeamToolCallCompleted"):
        t = getattr(chunk, "tool", None)
        name = getattr(t, "tool_name", "?") if t else "?"
        result = getattr(t, "result", "?") if t else "?"
        return f"[{timestamp}] ✓ TOOL DONE: {name} | result=\n\n{result}\n\n"

    elif ev in ("RunCompleted", "TeamRunCompleted"):
        metrics = getattr(chunk, "metrics", None)
        tokens = ""
        if metrics:
            tokens = f" | tokens in={getattr(metrics, 'input_tokens', 0)} out={getattr(metrics, 'output_tokens', 0)}"
        return f"[{timestamp}] ✔ RUN COMPLETED{tokens}"

    elif ev in ("RunError", "TeamRunError"):
        msg = getattr(chunk, "content", "?")
        return f"[{timestamp}] ✘ RUN ERROR: {msg}"

    else:
        return f"[{timestamp}] 📌 {ev}"


async def run_validation(
    dataset_id: str,
    num_records: int,
    output_dir: Path,
    mlflow_tracking_uri: str = "http://100.113.186.28:5000",
    mlflow_experiment_name: str = "vds-agent-validation",
):
    """Main validation function.

    Args:
        dataset_id: MLflow dataset ID
        num_records: Number of records to run
        mlflow_tracking_uri: MLflow tracking URI (default from env)
        mlflow_experiment_name: MLflow experiment name
        output_dir: Directory for logs (default: PROJECT_ROOT/local)
    """

    if output_dir is None:
        output_dir = PROJECT_ROOT / "local"

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    validation_logger = ValidationLogger(
        output_dir / "validation_total.log"
    )

    validation_logger.info(f"Starting validation for dataset_id: {dataset_id}")
    validation_logger.info(f"Number of records to run: {num_records}")
    validation_logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

    try:
        dataset = mlflow.genai.datasets.get_dataset(dataset_id=dataset_id)
        dataset_name = dataset.name
        validation_logger.info(f"Dataset name: {dataset_name}")

    except Exception as e:
        validation_logger.error(f"Failed to get dataset: {e}")
        return

    df = dataset.to_df()
    total_records = len(df)
    validation_logger.info(f"Total records in dataset: {total_records}")

    if num_records > total_records:
        validation_logger.info(f"Adjusting num_records to {total_records}")
        num_records = total_records

    completed_session_ids = get_completed_session_ids(dataset_name, output_dir)
    validation_logger.info(f"Already completed runs: {len(completed_session_ids)}")

    runs_completed = 0
    runs_failed = 0
    runs_skipped = 0

    for idx in range(min(num_records, total_records)):
        row = df.iloc[idx].to_dict()

        try:
            eval_record = EvalRecord.from_dict(row) #type:ignore
        except Exception as e:
            validation_logger.error(f"Failed to parse record {idx}: {e}")
            runs_skipped += 1
            continue

        session_id = eval_record.inputs.session_id or str(uuid4())

        if session_id in completed_session_ids:
            validation_logger.info(f"Skipping record {idx} - already run (session_id: {session_id})")
            runs_skipped += 1
            continue

        validation_logger.info(f"="*60)
        validation_logger.info(f"Running record {idx + 1}/{num_records}")

        success, error_msg = await run_single_record_with_logging(
            record=eval_record,
            dataset_name=dataset_name,
            output_dir=output_dir,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_tracking_uri=mlflow_tracking_uri,
            validation_logger=validation_logger,
            session_id=session_id,
        )

        if success:
            runs_completed += 1
        else:
            runs_failed += 1

    validation_logger.info(f"="*60)
    validation_logger.info(f"VALIDATION SUMMARY")
    validation_logger.info(f"Total records: {total_records}")
    validation_logger.info(f"Records requested: {num_records}")
    validation_logger.info(f"Runs completed: {runs_completed}")
    validation_logger.info(f"Runs failed: {runs_failed}")
    validation_logger.info(f"Runs skipped: {runs_skipped}")
    validation_logger.info(f"="*60)


def main():
    
    dataset_id = "d-ec37df2ccdfa4ce5b9614724fdceb27e"
    num_records = 70
    experiment = "vds-agent-validation"
    output_dir = "/home/tinhanhnguyen/Desktop/HK8/Capstone/CAPSTONE_PROJECT/videodeepsearch/validate/logs"
    
    output_dir = Path(output_dir)

    asyncio.run(
        run_validation(
            dataset_id=dataset_id,
            num_records=num_records,
            mlflow_experiment_name=experiment,
            output_dir=output_dir,
        )
    )


if __name__ == "__main__":
    main()
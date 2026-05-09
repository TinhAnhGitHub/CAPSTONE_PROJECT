import pandas as pd
from tqdm import tqdm
import json
import sys
from typing import cast
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import mlflow
from mlflow.entities import Trace
from mlflow.entities.assessment import Feedback
from mlflow.genai.scorers import (
    ToolCallEfficiency,
    ToolCallCorrectness,
    Correctness,
    RelevanceToQuery,
)
from mlflow.genai.scorers.deepeval import (
    TaskCompletion,
)
from loguru import logger

load_dotenv()
key = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_KEY"] = key  # type: ignore

def setup_logger(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluate_trace_{timestamp}.log"

    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True,
    )

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )

    logger.info(f"Logger initialized | Log file: {log_file}")
    return log_file

JUDGE_MODEL = "openrouter:/qwen/qwen3.6-plus"

SCORERS = [
    TaskCompletion(model=JUDGE_MODEL, threshold=0.5, include_reason=True),  # type: ignore
    ToolCallEfficiency(model=JUDGE_MODEL),  # type: ignore
    ToolCallCorrectness(model=JUDGE_MODEL),  # type: ignore
    RelevanceToQuery(model=JUDGE_MODEL),  # type: ignore
    Correctness(model=JUDGE_MODEL) # type: ignore
]


def get_record_based_on_trace(
    data_df: list[dict],
    trace: Trace
):
    """
    Get the corresponding record based on trace's session id.
    Matches trace input preview with dataset record's user_demand.
    """
    trace_input_preview = cast(str, trace.info.request_preview).strip('"').strip("'").replace('\\n', '')

    filtered_records = [
        r for r in data_df
        if r['inputs']['user_demand'].replace('\n', '') == trace_input_preview
    ]

    if not filtered_records:
        logger.warning(f"No matching record for trace input | Preview: {trace_input_preview[:80]}...")
        return None

    logger.debug(f"Found matching record for trace")
    return filtered_records[0]


def full_evaluate_one_trace(
    dataset_record: list[dict],
    trace: Trace
) -> dict:
    """
    Evaluate a single trace using all configured scorers.

    Returns a dict with:
        - trace_id
        - trace_request_preview
        - trace_response_preview
        - evaluation_results: list of scorer results
        - evaluation_summary: pass/fail counts
    """
    final_result: list[dict] = []
    trace_record = get_record_based_on_trace(dataset_record, trace)

    if trace_record is None:
        return {
            "trace_id": trace.info.trace_id,
            "error": "No matching dataset record found",
            "evaluation_results": [],
        }

    expectations = trace_record.get('expectations', {})

    passes = 0
    failures = 0

    logger.info("  Running scorers...")

    for scorer in SCORERS:
        scorer_name = scorer.__class__.__name__
        scorer_kwargs: dict = {'trace': trace}

        if isinstance(scorer, Correctness):
            if not expectations:
                logger.debug(f"  Skipping {scorer_name} - no expectations")
                continue
            scorer_kwargs['expectations'] = expectations

        try:
            result_scorer = scorer(**scorer_kwargs)

            if result_scorer is None:
                logger.warning(f"  {scorer_name} returned None")
                final_result.append({
                    'name': scorer_name,
                    'result': None,
                    'error': 'Scorer returned None'
                })
                failures += 1
                continue

            result_scorer = cast(Feedback, result_scorer)

            passed = False
            if result_scorer.feedback:
                feedback = result_scorer.feedback
                if feedback.value.value == 'yes':  # type: ignore
                    passes += 1
                    passed = True
                else:
                    failures += 1

                result_value = feedback.value.value  # type: ignore
                status_icon = "✓" if passed else "✗"
                logger.info(f"    {status_icon} {scorer_name}: {result_value}")
            else:
                logger.warning(f"    ? {scorer_name}: No feedback")
                failures += 1

            final_result.append({
                'name': result_scorer.name,
                'result': result_scorer.to_dictionary(),
                'passed': passed
            })

        except Exception as e:
            logger.error(f"    ✗ {scorer_name}: Error - {e}")
            final_result.append({
                'name': scorer_name,
                'result': None,
                'error': str(e)
            })
            failures += 1

    pass_rate = passes / len(SCORERS) * 100 if SCORERS else 0
    logger.info(f"  Trace summary: {passes}/{len(SCORERS)} passed ({pass_rate:.0f}%)")

    return {
        'trace_id': trace.info.trace_id,
        'trace_request_preview': trace.info.request_preview,
        'trace_response_preview': trace.info.response_preview,
        'evaluation_results': final_result,
        'evaluation_summary': {
            'total_scorers': len(SCORERS),
            'passes': passes,
            'failures': failures,
            'pass_rate': passes / len(SCORERS) if SCORERS else 0
        }
    }


def save_results(
    results: list[dict],
    output_dir: Path,
    experiment_name: str
) -> Path:
    """
    Save evaluation results to JSON and CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"evaluation_{experiment_name}_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"Saved JSON results: {json_path}")

    summary_data = []
    for r in results:
        summary_data.append({
            'trace_id': r.get('trace_id'),
            'request_preview': r.get('trace_request_preview', '')[:100],
            'response_preview': r.get('trace_response_preview', '')[:100],
            'passes': r.get('evaluation_summary', {}).get('passes', 0),
            'failures': r.get('evaluation_summary', {}).get('failures', 0),
            'pass_rate': r.get('evaluation_summary', {}).get('pass_rate', 0),
            'error': r.get('error', '')
        })

    csv_path = output_dir / f"evaluation_summary_{experiment_name}_{timestamp}.csv"
    import pandas as pd
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_path, index=False)
    logger.success(f"Saved CSV summary: {csv_path}")

    return json_path

def main():
    log_dir = Path("/home/tinhanhnguyen/Desktop/HK8/Capstone/CAPSTONE_PROJECT/videodeepsearch/validate/logs")
    setup_logger(log_dir)

    logger.info("=" * 60)
    logger.info("TRACE EVALUATION PIPELINE")
    logger.info("=" * 60)

    mlflow.set_tracking_uri("http://100.113.186.28:5000")
    mlflow.set_experiment("vds-agent-validation")
    logger.info("MLflow configured | Tracking URI: http://100.113.186.28:5000")
    logger.info("Experiment: vds-agent-validation")

    dataset_id = "d-ec37df2ccdfa4ce5b9614724fdceb27e"
    logger.info(f"Fetching dataset: {dataset_id}")

    dataset = mlflow.genai.datasets.get_dataset(dataset_id=dataset_id)  # type: ignore
    dataset_df = json.loads(dataset.to_json())
    records = dataset_df['records']

    logger.info(f"Dataset loaded | Name: {dataset.name} | Records: {len(records)}")

    logger.info("Searching for traces...")
    traces = cast(pd.DataFrame, mlflow.search_traces(
        locations=['6'],
        filter_string="trace.text LIKE '%I want%'",
        return_type='pandas'
    ))
    logger.info(f"Found {len(traces)} traces to evaluate")

    all_results: list[dict] = []

    for index, trace_row in tqdm(traces.iterrows(), total=len(traces), desc="Evaluating traces ..."):
        trace_id = trace_row['trace_id']

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"TRACE [{index + 1}/{len(traces)}] | ID: {trace_id}") #type:ignore
        logger.info("=" * 60)

        try:
            actual_trace = mlflow.get_trace(trace_id)
            if actual_trace is None:
                logger.error(f"Trace not found: {trace_id}")
                continue

            trace_request_preview = cast(str, actual_trace.info.request_preview)
            trace_response_preview = cast(str, actual_trace.info.response_preview)

            logger.debug(f"Request preview: {trace_request_preview[:80]}...")
            logger.debug(f"Response preview: {trace_response_preview[:80]}...")

            result = full_evaluate_one_trace(records, actual_trace)
            all_results.append(result)

        except Exception as e:
            logger.error(f"Failed to process trace {trace_id} | Error: {e}")
            all_results.append({
                'trace_id': trace_id,
                'error': str(e),
                'evaluation_results': []
            })

    logger.info("")
    logger.info("=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    save_results(all_results, log_dir, "vds-agent-validation")

    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    total_traces = len(all_results)
    successful_evals = sum(1 for r in all_results if 'evaluation_summary' in r)
    failed_evals = total_traces - successful_evals
    total_passes = sum(r.get('evaluation_summary', {}).get('passes', 0) for r in all_results)
    total_failures = sum(r.get('evaluation_summary', {}).get('failures', 0) for r in all_results)

    logger.info(f"  Total traces evaluated: {total_traces}")
    logger.info(f"  Successful evaluations: {successful_evals}")
    logger.info(f"  Failed evaluations:     {failed_evals}")
    logger.info(f"  Total scorer passes:    {total_passes}")
    logger.info(f"  Total scorer failures:  {total_failures}")

    if total_passes + total_failures > 0:
        overall_pass_rate = total_passes / (total_passes + total_failures) * 100
        logger.success(f"  Overall pass rate:      {overall_pass_rate:.1f}%")
    else:
        logger.warning("  No scorer results available")

    logger.info("=" * 60)
    logger.success("Evaluation pipeline completed")


if __name__ == "__main__":
    main()
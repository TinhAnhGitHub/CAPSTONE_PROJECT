"""
Matched Temporal Grounding Score (MTGS) and VideoRecall Calculator

For each query:
1. Find matched video IDs (intersection of predicted and ground truth)
2. VideoRecall = |matched_videos| / |ground_truth_videos| * 100
3. For each matched video, compute temporal IoU over merged intervals
4. MTGS = mean IoU across all matched video IDs

If no matched video IDs, MTGS = 0.

Logs results directly to MLflow traces using mlflow.log_expectation.
"""

import json
import re
from pathlib import Path
from typing import Optional, cast
from tqdm import tqdm
import mlflow
from mlflow.entities import Trace, AssessmentSource, AssessmentSourceType
from loguru import logger

MLFLOW_TRACKING_URI = "http://100.113.186.28:5000"
EXPERIMENT_NAME = "vds-agent-validation"
EXPERIMENT_ID = "6"
TRACE_FILTER = 'trace.text LIKE "%I want%"'

BASE_DIR = Path("/home/tinhanhnguyen/Desktop/HK8/Capstone/CAPSTONE_PROJECT/videodeepsearch")
EVAL_RECORDS_PATH = BASE_DIR / "local/mlflow_eval_records.json"
PREDICTED_SEGMENTS_PATH = BASE_DIR / "test/notebooks/extracted_segments.json"
OUTPUT_DIR = BASE_DIR / "test/notebooks/analysis_results"

Intervals = list[tuple[float, float]]
SegmentMap = dict[str, Intervals]


def parse_timestamp(timestamp: str) -> float:
    """Convert HH:MM:SS, MM:SS, or SS to seconds."""
    parts = timestamp.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(timestamp)


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def merge_intervals(intervals: Intervals) -> Intervals:
    if not intervals:
        return []

    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged_ivs = [sorted_ivs[0]]

    for start, end in sorted_ivs[1:]:
        prev_start, prev_end = merged_ivs[-1]
        if start <= prev_end:
            merged_ivs[-1] = (prev_start, max(prev_end, end))
        else:
            merged_ivs.append((start, end))

    return merged_ivs


def interval_duration(intervals: Intervals) -> float:
    return sum(end - start for start, end in intervals)


def intersect_intervals(a: Intervals, b: Intervals) -> Intervals:
    """Intersection of two sorted, merged interval lists."""
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def union_intervals(a: Intervals, b: Intervals) -> Intervals:
    return merge_intervals(a + b)


def temporal_iou(gt: Intervals, pred: Intervals) -> float:
    """Temporal IoU for a single video."""
    if not gt or not pred:
        return 0.0
    gt_m, pred_m = merge_intervals(gt), merge_intervals(pred)
    union_dur = interval_duration(union_intervals(gt_m, pred_m))
    if union_dur == 0:
        return 0.0
    return interval_duration(intersect_intervals(gt_m, pred_m)) / union_dur


def compute_mtgs(gt: SegmentMap, pred: SegmentMap) -> float:
    """Mean temporal IoU over matched (intersection) video IDs."""
    matched = set(gt) & set(pred)
    if not matched:
        return 0.0
    return sum(temporal_iou(gt[vid], pred[vid]) for vid in matched) / len(matched)


def compute_video_recall(gt: SegmentMap, pred: SegmentMap) -> float:
    """
    VideoRecall = percentage of ground truth video IDs that were correctly predicted.

    Formula: |V_M| / |V_G| * 100
    """
    gt_videos = set(gt.keys())
    if len(gt_videos) == 0:
        return 100.0

    matched = gt_videos & set(pred.keys())
    return (len(matched) / len(gt_videos)) * 100.0


def load_json(path: Path) -> object:
    with open(path) as f:
        return json.load(f)


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def find_matching_record(trace: Trace, eval_records: list) -> Optional[dict]:
    """Get the corresponding record based on trace's request matching user_demand."""
    trace_input_preview = cast(
        str,
        trace.info.request_preview
    ).strip('"').strip("'").replace('\\n', '')

    filter_data_record = next(
        filter(
            lambda x: x['inputs']['user_demand'].replace('\n', '') == trace_input_preview,
            eval_records
        ),
        None
    )

    return filter_data_record


def gt_segments_from_record(record: dict) -> SegmentMap:
    """Parse ground-truth segments from an eval record's expected_response."""
    text = record.get("expectations", {}).get("expected_response", "")
    video_ids = re.findall(r"Video ID:\s*([a-zA-Z0-9]+)", text)
    segments = re.findall(r"Segment:\s*\(Start:\s*([\d:]+),\s*End:\s*([\d:]+)\)", text)

    result: SegmentMap = {}
    for vid, (start_ts, end_ts) in zip(video_ids, segments):
        result.setdefault(vid, []).append(
            (parse_timestamp(start_ts), parse_timestamp(end_ts))
        )
    return result


def pred_segments_from_store(trace_id: str, predicted_segments: dict) -> SegmentMap:
    """Extract predicted segments for one trace from the pre-loaded store."""
    result: SegmentMap = {}
    for vid, segs in predicted_segments.get(trace_id, {}).items():
        result[vid] = [
            (parse_timestamp(s[0]), parse_timestamp(s[1]))
            for s in segs
            if len(s) >= 2
        ]
    return result


def log_evaluation_to_trace(
    trace_id: str,
    mtgs: float,
    video_recall: float,
) -> None:
    """Log MTGS and VideoRecall expectations directly to MLflow trace."""
    mlflow.log_feedback(
        trace_id=trace_id,
        name="mtgs",
        value=mtgs,
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN
        ),
    )

    mlflow.log_feedback(
        trace_id=trace_id,
        name="video_recall_pct",
        value=video_recall,
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN
        ),
    )


def analyse_and_log_trace(
    trace: Trace,
    eval_records: list,
    predicted_segments: dict,
    log_to_mlflow: bool = True,
) -> dict:
    """Compute MTGS, VideoRecall and log to MLflow trace."""
    trace_id = trace.info.trace_id
    record = find_matching_record(trace, eval_records)

    if record is None:
        return {
            "trace_id": trace_id,
            "error": "No matching ground truth record",
            "mtgs": None,
            "video_recall": None
        }

    gt = gt_segments_from_record(record)
    pred = pred_segments_from_store(trace_id, predicted_segments)
    matched = set(gt) & set(pred)

    mtgs = compute_mtgs(gt, pred)
    video_recall = compute_video_recall(gt, pred)

    per_video = {
        vid: {
            "iou": temporal_iou(gt[vid], pred[vid]),
            "gt_segments_count": len(gt[vid]),
            "pred_segments_count": len(pred[vid]),
            "gt_duration": interval_duration(merge_intervals(gt[vid])),
            "pred_duration": interval_duration(merge_intervals(pred[vid])),
        }
        for vid in matched
    }

    result = {
        "trace_id": trace_id,
        "mtgs": mtgs,
        "video_recall": video_recall,
        "ground_truth_videos": list(gt),
        "predicted_videos": list(pred),
        "matched_videos": list(matched),
        "num_gt_videos": len(gt),
        "num_pred_videos": len(pred),
        "num_matched_videos": len(matched),
        "per_video_iou": per_video,
        "gt_total_duration": sum(interval_duration(merge_intervals(gt[v])) for v in gt),
        "pred_total_duration": sum(interval_duration(merge_intervals(pred[v])) for v in pred),
        "request_preview": (trace.info.request_preview or "")[:200],
    }

    if log_to_mlflow:
        try:
            log_evaluation_to_trace(
                trace_id=trace_id,
                mtgs=mtgs,
                video_recall=video_recall,  
            )
            result["logged_to_mlflow"] = True
        except Exception as e:
            logger.error(f"Failed to log to MLflow for trace {trace_id}: {e}")
            result["logged_to_mlflow"] = False
            result["log_error"] = str(e)

    return result


def build_summary(results: list[dict], total_traces: int) -> dict:
    valid = [r for r in results if r.get("mtgs") is not None]
    if not valid:
        return {"error": "No valid MTGS calculations"}

    mtgs_values = [r["mtgs"] for r in valid]
    recall_values = [r["video_recall"] for r in valid]

    return {
        "total_traces": total_traces,
        "matched_traces": len(valid),
        # MTGS metrics
        "avg_mtgs": sum(mtgs_values) / len(mtgs_values),
        "min_mtgs": min(mtgs_values),
        "max_mtgs": max(mtgs_values),
        # VideoRecall metrics
        "avg_video_recall_pct": sum(recall_values) / len(recall_values),
        "min_video_recall_pct": min(recall_values),
        "max_video_recall_pct": max(recall_values),
        # MTGS distribution
        "mtgs_distribution": {
            "zero":          sum(1 for v in mtgs_values if v == 0),
            "low_0_25":      sum(1 for v in mtgs_values if 0 < v <= 0.25),
            "medium_25_50":  sum(1 for v in mtgs_values if 0.25 < v <= 0.5),
            "high_50_75":    sum(1 for v in mtgs_values if 0.5 < v <= 0.75),
            "perfect_75_100": sum(1 for v in mtgs_values if v > 0.75),
        },
        # VideoRecall distribution
        "video_recall_distribution": {
            "zero":           sum(1 for v in recall_values if v == 0),
            "partial_1_49":   sum(1 for v in recall_values if 0 < v < 50),
            "half_50":        sum(1 for v in recall_values if v == 50),
            "good_51_74":     sum(1 for v in recall_values if 50 < v < 75),
            "most_75_99":     sum(1 for v in recall_values if 75 <= v < 100),
            "perfect_100":    sum(1 for v in recall_values if v == 100),
        },
        # Video counts
        "avg_matched_videos": sum(r["num_matched_videos"] for r in valid) / len(valid),
        "avg_gt_videos": sum(r["num_gt_videos"] for r in valid) / len(valid),
        "avg_pred_videos": sum(r["num_pred_videos"] for r in valid) / len(valid),
        # MLflow logging stats
        "mlflow_logged_count": sum(1 for r in valid if r.get("logged_to_mlflow")),
    }


def print_summary(summary: dict) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("MTGS & VIDEO RECALL ANALYSIS SUMMARY")
    logger.info(sep)
    logger.info(f"Total traces:                  {summary.get('total_traces', 'N/A')}")
    logger.info(f"Matched traces (with GT):      {summary.get('matched_traces', 'N/A')}")
    logger.info(f"Logged to MLflow:              {summary.get('mlflow_logged_count', 'N/A')}")
    logger.info(sep)
    logger.info("--- VIDEO RECALL METRICS ---")
    logger.info(f"Average VideoRecall:           {summary.get('avg_video_recall_pct', 0):.2f}%")
    logger.info(f"Min VideoRecall:               {summary.get('min_video_recall_pct', 0):.2f}%")
    logger.info(f"Max VideoRecall:               {summary.get('max_video_recall_pct', 0):.2f}%")
    logger.info(f"Avg GT videos per trace:       {summary.get('avg_gt_videos', 0):.2f}")
    logger.info(f"Avg Pred videos per trace:     {summary.get('avg_pred_videos', 0):.2f}")
    logger.info(f"Avg matched videos per trace:  {summary.get('avg_matched_videos', 0):.2f}")
    recall_dist = summary.get("video_recall_distribution", {})
    logger.info("Video Recall Distribution:")
    logger.info(f"  0% (no match):     {recall_dist.get('zero', 0)}")
    logger.info(f"  1-49% (partial):   {recall_dist.get('partial_1_49', 0)}")
    logger.info(f"  50% (half):        {recall_dist.get('half_50', 0)}")
    logger.info(f"  51-74% (good):     {recall_dist.get('good_51_74', 0)}")
    logger.info(f"  75-99% (most):     {recall_dist.get('most_75_99', 0)}")
    logger.info(f"  100% (perfect):    {recall_dist.get('perfect_100', 0)}")
    logger.info(sep)
    logger.info("--- MTGS METRICS ---")
    logger.info(f"Average MTGS:                  {summary.get('avg_mtgs', 0):.4f}")
    logger.info(f"Min MTGS:                      {summary.get('min_mtgs', 0):.4f}")
    logger.info(f"Max MTGS:                      {summary.get('max_mtgs', 0):.4f}")
    mtgs_dist = summary.get("mtgs_distribution", {})
    logger.info("MTGS Distribution:")
    logger.info(f"  Zero (no match): {mtgs_dist.get('zero', 0)}")
    logger.info(f"  Low   (0-0.25):  {mtgs_dist.get('low_0_25', 0)}")
    logger.info(f"  Med  (0.25-0.5): {mtgs_dist.get('medium_25_50', 0)}")
    logger.info(f"  High (0.5-0.75): {mtgs_dist.get('high_50_75', 0)}")
    logger.info(f"  Top  (0.75-1.0): {mtgs_dist.get('perfect_75_100', 0)}")
    logger.info(sep)


def run(
    eval_records_path: Path = EVAL_RECORDS_PATH,
    predicted_segments_path: Path = PREDICTED_SEGMENTS_PATH,
    output_dir: Path = OUTPUT_DIR,
    log_to_mlflow: bool = True,
) -> dict:
    eval_records = load_json(eval_records_path)
    predicted_segments = load_json(predicted_segments_path)
    logger.info(f"Loaded {len(eval_records)} GT records, {len(predicted_segments)} predicted segment records")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    traces = cast(list[Trace], mlflow.search_traces(
        locations=[EXPERIMENT_ID],
        filter_string=TRACE_FILTER,
        return_type="list",
    ))

    logger.info(f"Found {len(traces)} traces")

    results = []
    for trace in tqdm(traces, total=len(traces), desc="Calculating & Logging MTGS/VideoRecall"):
        try:
            results.append(analyse_and_log_trace(trace, eval_records, predicted_segments, log_to_mlflow=log_to_mlflow))
        except Exception as exc:
            logger.error(f"Error processing trace {trace.info.trace_id}: {exc}")
            results.append({"trace_id": trace.info.trace_id, "error": str(exc)})

    summary = build_summary(results, total_traces=len(traces))
    output = {
        "metadata": {
            "experiment_name": EXPERIMENT_NAME,
            "experiment_id": EXPERIMENT_ID,
            "analysis_type": "MTGS & VideoRecall (logged to MLflow traces)",
            "log_to_mlflow_enabled": log_to_mlflow,
        },
        "summary": summary,
        "per_trace_results": results,
    }

    output_path = output_dir / "mtgs_results.json"
    save_json(output, output_path)
    logger.success(f"Results saved to {output_path}")
    print_summary(summary)
    return output


if __name__ == "__main__":
    logger.remove()
    logger.add(
        lambda msg: print(msg),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    run(log_to_mlflow=True)

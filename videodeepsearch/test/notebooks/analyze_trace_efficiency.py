"""
MLflow Trace Efficiency Analyzer

Analyzes traces from the vds-agent-validation experiment to understand:
1. When the agent found the answer vs total steps taken
2. Efficiency metrics (potential savings)
3. Correctness metrics (video ID match, segment overlap)
4. Agent hierarchy and tool usage patterns
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

import mlflow
import pandas as pd
from mlflow.entities import Trace
from loguru import logger


# Configuration
MLFLOW_TRACKING_URI = "http://100.113.186.28:5000"
EXPERIMENT_NAME = "vds-agent-validation"
EXPERIMENT_ID = "6"
TRACE_FILTER = 'trace.text LIKE "%I want%"'

BASE_DIR = Path("/home/tinhanhnguyen/Desktop/HK8/Capstone/CAPSTONE_PROJECT/videodeepsearch")
EVAL_RECORDS_PATH = BASE_DIR / "local/mlflow_eval_records.json"
OUTPUT_DIR = BASE_DIR / "test/notebooks/analysis_results"


class TraceAnalyzer:
    """Main analyzer for MLflow traces."""

    def __init__(self, eval_records_path: Path, output_dir: Path):
        self.eval_records = self._load_expected_responses(eval_records_path)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    def _load_expected_responses(self, path: Path) -> list:
        """Load expected responses as list."""
        with open(path, 'r') as f:
            records = json.load(f)

        logger.info(f"Loaded {len(records)} expected responses")
        return records

    def _normalize_for_lookup(self, text: str) -> str:
        """Normalize text for lookup - handle different quote types and escaped chars."""
        if not text:
            return ""
        normalized = text.strip()

        # Remove surrounding quotes (MLflow wraps in quotes)
        while normalized.startswith('"') or normalized.startswith("'"):
            normalized = normalized[1:]
        while normalized.endswith('"') or normalized.endswith("'"):
            normalized = normalized[:-1]

        # Replace fancy quotes with regular quotes
        normalized = normalized.replace("'", "'").replace("'", "'")
        normalized = normalized.replace('"', '"').replace('"', '"')

        # Handle escaped newlines
        normalized = normalized.replace('\\n', ' ')
        normalized = normalized.replace('\\n', ' ')

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        return normalized[:250]

    def fetch_all_traces(self) -> pd.DataFrame:
        """Fetch all traces matching the filter."""
        logger.info(f"Searching for traces with filter: {TRACE_FILTER}")

        traces = pd.DataFrame(mlflow.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=TRACE_FILTER,
            return_type='pandas'
        ))

        logger.info(f"Found {len(traces)} traces")
        return traces

    def find_matching_record(self, trace: Trace) -> Optional[dict]:
        """Find matching expected response for a trace."""
        # Get trace request to match with dataset
        request = trace.info.request_preview
        if not request:
            return None

        # Normalize the request
        request_normalized = self._normalize_for_lookup(request)

        # Try to find matching record by comparing user_demand
        for record in self.eval_records:
            user_demand = record.get('inputs', {}).get('user_demand', '')
            if not user_demand:
                continue

            record_normalized = self._normalize_for_lookup(user_demand)

            # Exact match on first 250 chars
            if request_normalized == record_normalized:
                return record

            # Or partial match - first 100 chars match
            if request_normalized[:100] == record_normalized[:100]:
                return record

        return None

    def extract_expected_info(self, record: dict) -> dict:
        """Extract video IDs and segments from expected response."""
        expected_response = record.get('expectations', {}).get('expected_response', '')

        # Extract video IDs
        video_ids = re.findall(r'Video ID:\s*([a-zA-Z0-9]+)', expected_response)

        # Extract segments (start_time, end_time)
        segments = re.findall(r'Segment:\s*\(Start:\s*([\d:]+),\s*End:\s*([\d:]+)\)', expected_response)

        return {
            'expected_video_ids': video_ids,
            'expected_segments': [(s[0], s[1]) for s in segments],
            'expected_response_preview': expected_response[:500]
        }

    def analyze_single_trace(self, trace: Trace) -> dict:
        """Analyze a single trace."""
        spans = trace.data.spans

        # Get matching expected response
        record = self.find_matching_trace_record(trace)
        expected_info = self.extract_expected_info(record) if record else {}

        # Analyze agent hierarchy
        agent_metrics = self._analyze_agent_hierarchy(spans)

        # Find milestones
        expected_video_ids = expected_info.get('expected_video_ids', [])
        milestones = self._find_answer_milestones(spans, expected_video_ids)

        # Calculate metrics
        total_spans = len(spans)
        tool_calls = len([s for s in spans if s.span_type == 'TOOL'])
        llm_calls = len([s for s in spans if s.span_type == 'LLM'])
        agent_calls = len([s for s in spans if s.span_type == 'AGENT'])

        core_answer_step = milestones.get('core_answer_step', total_spans)
        potential_savings = total_spans - core_answer_step
        savings_pct = (potential_savings / total_spans * 100) if total_spans > 0 else 0

        # Get actual video IDs found
        actual_video_ids = self._extract_actual_video_ids(spans)

        # Calculate correctness
        video_match = bool(set(expected_video_ids) & set(actual_video_ids)) if expected_video_ids else None

        result = {
            'trace_id': trace.info.trace_id,
            'total_spans': total_spans,
            'total_tool_calls': tool_calls,
            'total_llm_calls': llm_calls,
            'total_agent_calls': agent_calls,
            'execution_duration_ms': trace.info.execution_duration,

            # Agents
            'agents': agent_metrics['agents'],
            'workers_spawned': agent_metrics['workers'],
            'worker_details': agent_metrics['worker_details'],
            'tool_sequence': agent_metrics['tool_sequence'],
            'per_agent_metrics': agent_metrics['per_agent'],

            # Milestones
            'video_id_found_step': milestones.get('video_id_found_step'),
            'timeline_found_step': milestones.get('timeline_found_step'),
            'core_answer_step': core_answer_step,
            'final_step': total_spans,
            'potential_savings': potential_savings,
            'potential_savings_pct': round(savings_pct, 2),

            # Correctness
            'expected_video_ids': expected_video_ids,
            'actual_video_ids': actual_video_ids,
            'video_match': video_match,
            'expected_segments': expected_info.get('expected_segments', []),
            'expected_response_preview': expected_info.get('expected_response_preview', ''),
            'has_expected_record': record is not None,

            # Request/response
            'request_preview': trace.info.request_preview[:300] if trace.info.request_preview else '',
            'response_preview': trace.info.response_preview[:500] if trace.info.response_preview else '',
        }

        return result

    def find_matching_trace_record(self, trace: Trace) -> Optional[dict]:
        """Find matching record from eval_records for a trace."""
        # Use the same logic as in evaluate_trace.py
        trace_input = trace.info.request_preview
        if not trace_input:
            return None

        # Normalize
        normalized = self._normalize_for_lookup(trace_input)

        # Try to find matching record
        for record in self.eval_records:
            user_demand = record.get('inputs', {}).get('user_demand', '')
            if not user_demand:
                continue

            normalized_demand = self._normalize_for_lookup(user_demand)

            if normalized == normalized_demand:
                return record

        return None

    def _analyze_agent_hierarchy(self, spans: list) -> dict:
        """Analyze agent hierarchy and tool usage."""
        agents = []
        workers = []
        worker_details = []
        tool_sequence = []
        per_agent = {}

        # Build span dict for parent lookup
        span_dict = {s.span_id: s for s in spans}

        for i, span in enumerate(spans):
            step_num = i + 1

            # Track tool sequence
            if span.span_type == 'TOOL':
                tool_sequence.append({
                    'step': step_num,
                    'tool': span.name,
                    'agent': self._find_parent_agent(span, spans) or 'unknown'
                })

            # Track agents
            if span.span_type == 'AGENT':
                agent_name = span.name.replace('.arun', '')
                agents.append(agent_name)

                # Count tools for this agent
                child_tools = [s for s in spans if s.parent_id == span.span_id and s.span_type == 'TOOL']
                child_llms = [s for s in spans if s.parent_id == span.span_id and s.span_type == 'LLM']

                per_agent[agent_name] = {
                    'tool_calls': len(child_tools),
                    'llm_calls': len(child_llms),
                    'tools_used': [s.name for s in child_tools]
                }

                # Track worker spawns
                if 'worker' in agent_name.lower():
                    workers.append(agent_name)
                    worker_details.append({
                        'step': step_num,
                        'agent': agent_name,
                        'tools': len(child_tools)
                    })

        return {
            'agents': agents,
            'workers': len(workers),
            'worker_details': worker_details,
            'tool_sequence': tool_sequence,
            'per_agent': per_agent
        }

    def _find_parent_agent(self, span, spans: list) -> str:
        """Find the parent agent for a span."""
        parent_id = span.parent_id
        while parent_id:
            for s in spans:
                if s.span_id == parent_id:
                    if s.span_type == 'AGENT':
                        return s.name.replace('.arun', '')
                    parent_id = s.parent_id
                    break
            else:
                break
        return 'root'

    def _find_answer_milestones(self, spans: list, expected_video_ids: list) -> dict:
        """Find key milestones in the trace."""
        milestones = {
            'video_id_found_step': None,
            'timeline_found_step': None,
            'core_answer_step': len(spans)
        }

        if not expected_video_ids:
            # If no expected video IDs, estimate based on video_discovery_worker completion
            # Find when video_discovery_worker.arun completes
            for i, span in enumerate(spans):
                if span.span_type == 'AGENT' and 'video_discovery_worker' in span.name:
                    milestones['core_answer_step'] = i + 1
                    break
            return milestones

        # Find when expected video ID first appears
        for i, span in enumerate(spans):
            if span.span_type == 'TOOL' and span.outputs:
                output_str = str(span.outputs)

                # Check for video ID
                for vid in expected_video_ids:
                    if vid in output_str:
                        if milestones['video_id_found_step'] is None:
                            milestones['video_id_found_step'] = i + 1

                        # Check for timeline/segment info
                        if 'timeline' in output_str.lower() or 'segment' in output_str.lower():
                            if milestones['timeline_found_step'] is None:
                                milestones['timeline_found_step'] = i + 1

                # Check if we have enough info for core answer
                if milestones['timeline_found_step'] and milestones['core_answer_step'] == len(spans):
                    milestones['core_answer_step'] = i + 1

        return milestones

    def _extract_actual_video_ids(self, spans: list) -> list:
        """Extract video IDs found in trace outputs."""
        video_ids = set()

        for span in spans:
            if span.outputs:
                output_str = str(span.outputs)
                # Extract video IDs (patterns like 69d3d8b506491d0fe1767466)
                found = re.findall(r'\b[0-9a-f]{22,}\b', output_str)
                video_ids.update(found)

        return list(video_ids)

    def analyze_all_traces(self) -> dict:
        """Analyze all traces and return results."""
        # Fetch trace IDs
        traces_df = self.fetch_all_traces()

        results = []

        logger.info("Analyzing traces...")
        for idx, row in tqdm(traces_df.iterrows(), total=len(traces_df), desc="Analyzing"):
            trace_id = row['trace_id']

            try:
                trace = mlflow.get_trace(trace_id)
                if trace is None:
                    logger.warning(f"Trace not found: {trace_id}")
                    continue

                result = self.analyze_single_trace(trace)
                results.append(result)

            except Exception as e:
                logger.error(f"Error analyzing trace {trace_id}: {e}")
                results.append({
                    'trace_id': trace_id,
                    'error': str(e)
                })

        # Generate summary
        summary = self._generate_summary(results)

        return {
            'metadata': {
                'total_traces': len(traces_df),
                'analyzed_traces': len(results),
                'experiment_name': EXPERIMENT_NAME,
                'experiment_id': EXPERIMENT_ID,
                'analysis_date': datetime.now().isoformat()
            },
            'traces': results,
            'summary': summary
        }

    def _generate_summary(self, results: list) -> dict:
        """Generate summary statistics."""
        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            return {'error': 'No valid results to summarize'}

        # Calculate metrics
        total_steps = [r['total_spans'] for r in valid_results]
        core_answer_steps = [r['core_answer_step'] for r in valid_results]
        potential_savings = [r['potential_savings'] for r in valid_results]
        savings_pct = [r['potential_savings_pct'] for r in valid_results]

        # Video match
        video_matches = [r['video_match'] for r in valid_results if r.get('video_match') is not None]

        # Workers spawned
        workers_count = [r['workers_spawned'] for r in valid_results]

        summary = {
            'avg_total_steps': sum(total_steps) / len(total_steps),
            'avg_core_answer_step': sum(core_answer_steps) / len(core_answer_steps),
            'avg_potential_savings': sum(potential_savings) / len(potential_savings),
            'avg_savings_percentage': sum(savings_pct) / len(savings_pct),
            'min_steps': min(total_steps),
            'max_steps': max(total_steps),
            'avg_workers': sum(workers_count) / len(workers_count),
            'video_match_rate': len(video_matches) / len(valid_results) if valid_results else 0,
            'total_traces': len(valid_results)
        }

        return summary

    def export_results(self, results: dict):
        """Export results to JSON and CSV files."""
        # Save JSON
        json_path = self.output_dir / "trace_analysis_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.success(f"Saved JSON: {json_path}")

        # Save summary CSV
        summary_data = []
        for trace in results['traces']:
            if 'error' in trace:
                continue
            summary_data.append({
                'trace_id': trace['trace_id'],
                'total_steps': trace['total_spans'],
                'video_id_found_step': trace['video_id_found_step'],
                'timeline_found_step': trace['timeline_found_step'],
                'core_answer_step': trace['core_answer_step'],
                'final_step': trace['final_step'],
                'potential_savings': trace['potential_savings'],
                'savings_pct': trace['potential_savings_pct'],
                'workers_count': trace['workers_spawned'],
                'execution_duration_ms': trace['execution_duration_ms'],
                'video_match': trace['video_match'],
                'has_expected_record': trace['has_expected_record']
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / "trace_analysis_summary.csv"
            summary_df.to_csv(csv_path, index=False)
            logger.success(f"Saved CSV: {csv_path}")

        # Save agent tool usage CSV
        agent_data = []
        for trace in results['traces']:
            if 'error' in trace:
                continue
            trace_id = trace['trace_id']
            for agent_name, metrics in trace['per_agent_metrics'].items():
                agent_data.append({
                    'trace_id': trace_id,
                    'agent_name': agent_name,
                    'tool_calls': metrics['tool_calls'],
                    'llm_calls': metrics['llm_calls'],
                    'tools_used': '|'.join(metrics['tools_used'][:5])  # First 5 tools
                })

        if agent_data:
            agent_df = pd.DataFrame(agent_data)
            agent_csv_path = self.output_dir / "agent_tool_usage.csv"
            agent_df.to_csv(agent_csv_path, index=False)
            logger.success(f"Saved agent CSV: {agent_csv_path}")

        # Save correctness CSV
        correctness_data = []
        for trace in results['traces']:
            if 'error' in trace or not trace.get('has_expected_record'):
                continue
            correctness_data.append({
                'trace_id': trace['trace_id'],
                'expected_video_ids': '|'.join(trace['expected_video_ids']),
                'actual_video_ids': '|'.join(trace['actual_video_ids'][:3]),  # First 3
                'video_match': trace['video_match'],
                'expected_segments': str(trace['expected_segments'])
            })

        if correctness_data:
            correctness_df = pd.DataFrame(correctness_data)
            correctness_csv_path = self.output_dir / "correctness_analysis.csv"
            correctness_df.to_csv(correctness_csv_path, index=False)
            logger.success(f"Saved correctness CSV: {correctness_csv_path}")

        logger.info(f"Analysis complete! Results saved to {self.output_dir}")

        # Print summary
        summary = results.get('summary', {})
        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total traces analyzed: {summary.get('total_traces', 'N/A')}")
        logger.info(f"Average total steps: {summary.get('avg_total_steps', 0):.1f}")
        logger.info(f"Average core answer step: {summary.get('avg_core_answer_step', 0):.1f}")
        logger.info(f"Average potential savings: {summary.get('avg_potential_savings', 0):.1f} steps ({summary.get('avg_savings_percentage', 0):.1f}%)")
        logger.info(f"Video match rate: {summary.get('video_match_rate', 0)*100:.1f}%")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    # Setup
    logger.remove()
    logger.add(
        lambda msg: print(msg),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )

    logger.info("Starting trace efficiency analysis...")
    logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")

    # Create analyzer
    analyzer = TraceAnalyzer(EVAL_RECORDS_PATH, OUTPUT_DIR)

    # Run analysis
    results = analyzer.analyze_all_traces()

    # Export results
    analyzer.export_results(results)

    logger.success("Analysis complete!")


if __name__ == "__main__":
    main()

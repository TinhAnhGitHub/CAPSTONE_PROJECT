import mlflow
from mlflow.entities import Trace, Span, SpanType
from typing import cast
import json

mlflow.set_tracking_uri("http://100.113.186.28:5000")
mlflow.set_experiment("vds-agent-validation")

FAILED_TRACE_IDS = [
    "tr-d9fe3e99932901aa0692866904d78437",
    "tr-6b62c302513eb29d05765e31cea0a749",
    "tr-a28eb60b9d084523a6e71a4086e6cebe",
    "tr-956f710235a40ba3e1de448e3f733f4b",
    "tr-eda4ea886482f89d826dab2f834b344c",
    "tr-01ee9f794325242fd70297e4d8ee62f1",
    "tr-6fcb857dc745728a913f88034e753e49",
    "tr-faafa8257b2d8421ab3e975b4889c6b0",
    "tr-06dd73dbdc6cf4b6ab34302c84d458b9",
]

traces = cast(list, mlflow.search_traces(
    locations=['6'],
    filter_string="trace.text LIKE '%I want%'",
    return_type='list'
))

failed_traces = {t.info.trace_id: t for t in traces if t.info.trace_id in FAILED_TRACE_IDS}
print(f"Found {len(failed_traces)} failed traces out of {len(FAILED_TRACE_IDS)} requested")

def analyze_trace(trace: Trace):
    analysis = {
        "trace_id": trace.info.trace_id,
        "spans": [],
        "tool_calls": [],
        "llm_calls": [],
        "agent_calls": [],
        "final_response": None,
        "failure_reason": None,
    }
    
    for span in trace.data.spans:
        span_info = {
            "span_id": span.span_id,
            "name": span.name,
            "span_type": str(span.span_type),
            "status": str(span.status.status_code) if span.status else "UNKNOWN",
            "attributes": {},
        }
        
        attrs = span.attributes or {}
        
        if span.span_type == SpanType.TOOL:
            tool_name = attrs.get('tool.name', 'unknown')
            tool_input = attrs.get('tool.parameters', {})
            span_info["tool_name"] = tool_name
            span_info["tool_input"] = str(tool_input)[:500]
            
            if tool_name not in ['get_member_information', 'get_available_worker_tools', 'get_available_models', 'spawn_and_run_worker']:
                analysis["tool_calls"].append({
                    "tool_name": tool_name,
                    "input_summary": str(tool_input)[:500],
                    "span_id": span.span_id,
                })
        
        elif span.span_type == SpanType.LLM:
            model = attrs.get('mlflow.llm.model', 'unknown')
            usage = attrs.get('mlflow.chat.tokenUsage', {})
            span_info["model"] = model
            span_info["input_tokens"] = usage.get('input_tokens', 0)
            span_info["output_tokens"] = usage.get('output_tokens', 0)
            analysis["llm_calls"].append({
                "model": model,
                "input_tokens": usage.get('input_tokens', 0),
                "output_tokens": usage.get('output_tokens', 0),
            })
        
        elif span.span_type == SpanType.AGENT:
            agent_name = attrs.get('graph.node.name', 'unknown')
            span_info["agent_name"] = agent_name
            analysis["agent_calls"].append(agent_name)
        
        analysis["spans"].append(span_info)
    
    trace_text = ""
    if hasattr(trace.data, 'request') and trace.data.request:
        trace_text = str(trace.data.request)[:1000]
    analysis["trace_text"] = trace_text
    
    return analysis

for trace_id in FAILED_TRACE_IDS:
    if trace_id not in failed_traces:
        print(f"\n{'='*80}")
        print(f"TRACE NOT FOUND: {trace_id}")
        continue
    
    trace = failed_traces[trace_id]
    print(f"\n{'='*80}")
    print(f"TRACE: {trace_id}")
    print(f"{'='*80}")
    
    analysis = analyze_trace(trace)
    
    print(f"\nTotal spans: {len(analysis['spans'])}")
    print(f"Tool calls: {len(analysis['tool_calls'])}")
    print(f"LLM calls: {len(analysis['llm_calls'])}")
    print(f"Agent calls: {analysis['agent_calls']}")
    
    print(f"\n--- TOOLS USED ---")
    tools_used = [t['tool_name'] for t in analysis['tool_calls']]
    from collections import Counter
    tool_counts = Counter(tools_used)
    for tool, count in tool_counts.items():
        print(f"  {tool}: {count} times")
    
    print(f"\n--- LLM CALLS ---")
    total_input = sum(c['input_tokens'] for c in analysis['llm_calls'])
    total_output = sum(c['output_tokens'] for c in analysis['llm_calls'])
    print(f"  Total input tokens: {total_input}")
    print(f"  Total output tokens: {total_output}")
    
    print(f"\n--- SPAN DETAILS (first 10) ---")
    for i, span in enumerate(analysis['spans'][:10]):
        print(f"  [{i}] {span['span_type']} - {span['name']}")
        if 'tool_name' in span:
            print(f"      Tool: {span['tool_name']}")
        if 'agent_name' in span:
            print(f"      Agent: {span['agent_name']}")
    
    print(f"\n--- LAST SPANS (last 5) ---")
    for i, span in enumerate(analysis['spans'][-5:]):
        print(f"  [{len(analysis['spans'])-5+i}] {span['span_type']} - {span['name']}")
        if 'tool_name' in span:
            print(f"      Tool: {span['tool_name']}")
        if 'agent_name' in span:
            print(f"      Agent: {span['agent_name']}")

print("\n\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
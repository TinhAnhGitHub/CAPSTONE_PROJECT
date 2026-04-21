#!/usr/bin/env python3
"""Analyze result.json from AI Studio LLM evaluations"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_results(result_file: Path):
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total = len(data)
    found = sum(1 for d in data if d.get("answer_found", {}).get("step_index", -1) >= 0)
    not_found = total - found
    
    print("=" * 60)
    print("RESULT.JSON ANALYSIS")
    print("=" * 60)
    
    print(f"\n📊 OVERALL SUMMARY")
    print(f"   Total traces: {total}")
    print(f"   Answers found: {found} ({found/total*100:.1f}%)")
    print(f"   Answers NOT found: {not_found} ({not_found/total*100:.1f}%)")
    
    step_indices = [d.get("answer_found", {}).get("step_index", -1) for d in data]
    valid_steps = [s for s in step_indices if s >= 0]
    
    if valid_steps:
        print(f"\n📈 STEP INDEX DISTRIBUTION (when answer was found)")
        print(f"   Min step: {min(valid_steps)}")
        print(f"   Max step: {max(valid_steps)}")
        print(f"   Avg step: {sum(valid_steps)/len(valid_steps):.1f}")
        
        step_counts = Counter(valid_steps)
        print(f"\n   Top 5 most common step indices:")
        for step, count in step_counts.most_common(5):
            print(f"      Step {step}: {count} traces ({count/len(valid_steps)*100:.1f}%)")
    
    workers = [d.get("answer_found", {}).get("worker", "Unknown") for d in data]
    worker_counts = Counter(workers)
    
    print(f"\n🔧 WORKER DISTRIBUTION")
    for worker, count in worker_counts.most_common(10):
        print(f"   {worker}: {count}")
    
    all_tools = []
    for d in data:
        tools = d.get("answer_found", {}).get("tools_involved", [])
        for t in tools:
            all_tools.append(t.get("tool", "unknown"))
    
    tool_counts = Counter(all_tools)
    print(f"\n🛠️ TOP 10 MOST USED TOOLS")
    for tool, count in tool_counts.most_common(10):
        print(f"   {tool}: {count}")
    
    traces_not_found = [d["trace_id"] for d in data if d.get("answer_found", {}).get("step_index", -1) < 0]
    print(f"\n❌ TRACES WHERE ANSWER NOT FOUND ({len(traces_not_found)})")
    for tid in traces_not_found:
        evidence = next((d.get("answer_found", {}).get("evidence", "")[:100] for d in data if d["trace_id"] == tid), "")
        print(f"   {tid}")
        print(f"      Reason: {evidence}...")
    
    early_findings = sorted([d for d in data if d.get("answer_found", {}).get("step_index", 0) <= 25 and d.get("answer_found", {}).get("step_index", -1) >= 0],
                           key=lambda x: x.get("answer_found", {}).get("step_index", 0))
    
    print(f"\n⚡ EARLY FINDINGS (step <= 25, {len(early_findings)} traces)")
    for d in early_findings[:10]:
        step = d.get("answer_found", {}).get("step_index", 0)
        print(f"   {d['trace_id']}: Step {step}")
    
    late_findings = sorted([d for d in data if d.get("answer_found", {}).get("step_index", 0) > 40],
                           key=lambda x: x.get("answer_found", {}).get("step_index", 0), reverse=True)
    
    print(f"\n🐢 LATE FINDINGS (step > 40, {len(late_findings)} traces)")
    for d in late_findings[:10]:
        step = d.get("answer_found", {}).get("step_index", 0)
        print(f"   {d['trace_id']}: Step {step}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    result_file = Path(__file__).parent / "result.json"
    analyze_results(result_file)
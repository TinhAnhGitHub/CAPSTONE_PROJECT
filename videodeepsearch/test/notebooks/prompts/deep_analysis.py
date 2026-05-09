#!/usr/bin/env python3
"""
Deep Analysis of result.json - AI Studio LLM Evaluation Results
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import re

def load_data():
    with open('result.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def deep_analysis():
    data = load_data()
    
    print("=" * 80)
    print("DEEP ANALYSIS OF RESULT.JSON")
    print("=" * 80)
    print()
    
    # =========================================
    # 1. BASIC STATISTICS
    # =========================================
    total = len(data)
    found_traces = [d for d in data if d.get('answer_found', {}).get('step_index', -1) >= 0]
    not_found_traces = [d for d in data if d.get('answer_found', {}).get('step_index', -1) < 0]
    
    print("### 1. OVERALL SUCCESS METRICS ###")
    print("-" * 40)
    print(f"Total Traces Evaluated: {total}")
    print(f"Successful (Answer Found): {len(found_traces)} ({len(found_traces)/total*100:.1f}%)")
    print(f"Failed (Answer Not Found): {len(not_found_traces)} ({len(not_found_traces)/total*100:.1f}%)")
    print()
    
    # =========================================
    # 2. STEP INDEX DEEP ANALYSIS
    # =========================================
    steps = [d['answer_found']['step_index'] for d in found_traces]
    
    print("### 2. STEP INDEX DISTRIBUTION ###")
    print("-" * 40)
    
    # Quartile analysis
    sorted_steps = sorted(steps)
    q1 = sorted_steps[int(len(sorted_steps)*0.25)]
    q2 = sorted_steps[int(len(sorted_steps)*0.50)]
    q3 = sorted_steps[int(len(sorted_steps)*0.75)]
    
    print(f"Min Step: {min(steps)}")
    print(f"Max Step: {max(steps)}")
    print(f"Mean: {sum(steps)/len(steps):.1f}")
    print(f"Median (Q2): {q2}")
    print(f"Q1 (25%): {q1}")
    print(f"Q3 (75%): {q3}")
    print(f"IQR (Q3-Q1): {q3-q1}")
    print()
    
    # Performance buckets
    buckets = {
        'Very Fast (≤15)': len([s for s in steps if s <= 15]),
        'Fast (16-25)': len([s for s in steps if 16 <= s <= 25]),
        'Normal (26-35)': len([s for s in steps if 26 <= s <= 35]),
        'Slow (36-45)': len([s for s in steps if 36 <= s <= 45]),
        'Very Slow (46-60)': len([s for s in steps if 46 <= s <= 60]),
        'Extremely Slow (>60)': len([s for s in steps if s > 60]),
    }
    
    print("Performance Bucket Distribution:")
    for bucket, count in buckets.items():
        pct = count/len(steps)*100
        print(f"  {bucket}: {count} ({pct:.1f}%)")
    print()
    
    # =========================================
    # 3. WORKER ANALYSIS
    # =========================================
    print("### 3. WORKER PERFORMANCE ANALYSIS ###")
    print("-" * 40)
    
    worker_stats = defaultdict(lambda: {'found': 0, 'not_found': 0, 'steps': []})
    
    for d in data:
        worker = d['answer_found'].get('worker', 'Unknown')
        step = d['answer_found'].get('step_index', -1)
        if step >= 0:
            worker_stats[worker]['found'] += 1
            worker_stats[worker]['steps'].append(step)
        else:
            worker_stats[worker]['not_found'] += 1
    
    # Sort by total count
    worker_summary = []
    for worker, stats in worker_stats.items():
        total_w = stats['found'] + stats['not_found']
        avg_step = sum(stats['steps'])/len(stats['steps']) if stats['steps'] else 0
        success_rate = stats['found']/total_w*100 if total_w > 0 else 0
        worker_summary.append({
            'worker': worker,
            'total': total_w,
            'found': stats['found'],
            'not_found': stats['not_found'],
            'success_rate': success_rate,
            'avg_step': avg_step,
            'min_step': min(stats['steps']) if stats['steps'] else 0,
            'max_step': max(stats['steps']) if stats['steps'] else 0,
        })
    
    worker_summary.sort(key=lambda x: x['total'], reverse=True)
    
    print("\nTop 15 Workers by Activity:")
    print(f"{'Worker':<40} {'Total':>6} {'Found':>6} {'Fail':>5} {'Rate%':>7} {'AvgStep':>8}")
    print("-" * 80)
    for w in worker_summary[:15]:
        print(f"{w['worker']:<40} {w['total']:>6} {w['found']:>6} {w['not_found']:>5} {w['success_rate']:>6.1f}% {w['avg_step']:>7.1f}")
    print()
    
    # =========================================
    # 4. TOOL USAGE ANALYSIS
    # =========================================
    print("### 4. TOOL EFFECTIVENESS ANALYSIS ###")
    print("-" * 40)
    
    tool_stats = defaultdict(lambda: {'found': 0, 'total': 0, 'steps_when_used': []})
    
    for d in data:
        tools = d['answer_found'].get('tools_involved', [])
        step = d['answer_found'].get('step_index', -1)
        found = step >= 0
        
        for t in tools:
            tool_name = t.get('tool', 'unknown')
            tool_stats[tool_name]['total'] += 1
            if found:
                tool_stats[tool_name]['found'] += 1
                tool_stats[tool_name]['steps_when_used'].append(step)
    
    tool_summary = []
    for tool, stats in tool_stats.items():
        success_rate = stats['found']/stats['total']*100 if stats['total'] > 0 else 0
        avg_step = sum(stats['steps_when_used'])/len(stats['steps_when_used']) if stats['steps_when_used'] else 0
        tool_summary.append({
            'tool': tool,
            'total': stats['total'],
            'found': stats['found'],
            'success_rate': success_rate,
            'avg_step': avg_step,
        })
    
    tool_summary.sort(key=lambda x: x['total'], reverse=True)
    
    print("\nTop 20 Tools by Usage:")
    print(f"{'Tool':<45} {'Uses':>6} {'InFound':>8} {'Rate%':>7} {'AvgStep':>8}")
    print("-" * 85)
    for t in tool_summary[:20]:
        print(f"{t['tool']:<45} {t['total']:>6} {t['found']:>8} {t['success_rate']:>6.1f}% {t['avg_step']:>7.1f}")
    print()
    
    # =========================================
    # 5. TOOL SEQUENCE ANALYSIS
    # =========================================
    print("### 5. TOOL SEQUENCE PATTERNS ###")
    print("-" * 40)
    
    # Find most common tool combinations
    tool_combos = Counter()
    for d in found_traces:
        tools = d['answer_found'].get('tools_involved', [])
        tool_names = tuple(sorted([t.get('tool', '') for t in tools]))
        if tool_names:
            tool_combos[tool_names] += 1
    
    print("\nTop 10 Tool Combinations (Successful Traces):")
    for combo, count in tool_combos.most_common(10):
        print(f"  {count} traces used: {list(combo)}")
    print()
    
    # =========================================
    # 6. FAILURE ROOT CAUSE ANALYSIS
    # =========================================
    print("### 6. FAILURE ROOT CAUSE ANALYSIS ###")
    print("-" * 40)
    print(f"\nTotal Failed Traces: {len(not_found_traces)}")
    
    # Analyze failure reasons
    failure_themes = defaultdict(list)
    
    for d in not_found_traces:
        evidence = d['answer_found'].get('evidence', '').lower()
        trace_id = d['trace_id']
        worker = d['answer_found'].get('worker', 'Unknown')
        
        # Categorize failure
        if 'no relevant' in evidence or 'not found' in evidence or 'no content' in evidence:
            failure_themes['Missing Content in Corpus'].append(trace_id)
        elif 'weather' in evidence and ('engineering' in evidence or 'material' in evidence or 'fatigue' in evidence):
            failure_themes['Wrong Domain (Weather vs Engineering)'].append(trace_id)
        elif 'tornado' in evidence and 'structural' in evidence:
            failure_themes['Domain Mismatch - Meteorology Expected'].append(trace_id)
        elif 'cooking' in evidence and 'speech' in evidence:
            failure_themes['Cross-domain Confusion'].append(trace_id)
        elif 'no direct match' in evidence or 'could not' in evidence:
            failure_themes['Search Failed - No Match'].append(trace_id)
        else:
            failure_themes['Other/Unknown'].append(trace_id)
    
    print("\nFailure Categories:")
    for theme, traces in failure_themes.items():
        print(f"\n  {theme}: {len(traces)} traces")
        for tid in traces[:5]:
            print(f"    - {tid}")
        if len(traces) > 5:
            print(f"    ... and {len(traces)-5} more")
    
    # Extract keywords from failure evidence
    print("\n\nKeyword Analysis in Failure Evidence:")
    failure_keywords = Counter()
    for d in not_found_traces:
        evidence = d['answer_found'].get('evidence', '')
        # Extract significant words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', evidence.lower())
        for w in words:
            if w not in ['the', 'that', 'this', 'with', 'from', 'have', 'been', 'were', 'was', 'are', 'is', 'not', 'but', 'for', 'agent', 'video', 'output', 'search', 'tool', 'trace', 'expected', 'response']:
                failure_keywords[w] += 1
    
    print("Top 20 Keywords in Failure Evidence:")
    for word, count in failure_keywords.most_common(20):
        print(f"  {word}: {count}")
    print()
    
    # =========================================
    # 7. DOMAIN/CONTENT TYPE ANALYSIS
    # =========================================
    print("### 7. DOMAIN CATEGORY ANALYSIS ###")
    print("-" * 40)
    
    # Infer domain from evidence
    domain_stats = defaultdict(lambda: {'found': 0, 'not_found': 0, 'avg_step': 0, 'steps': []})
    
    cooking_keywords = ['pasta', 'sauce', 'chicken', 'steak', 'salmon', 'rice', 'cooking', 'kitchen', 'recipe', 'pan', 'fry', 'sear', 'onion', 'vegetable', 'carbonara', 'butter']
    speaking_keywords = ['speech', 'voice', 'vocal', 'breathing', 'public speaking', 'presentation', 'gestures', 'pace', 'tone', 'audience', 'pause', 'rambling', 'articulation']
    weather_keywords = ['tornado', 'hurricane', 'storm', 'weather', 'wind', 'damage', 'pressure', 'atmospheric', 'structural', 'debris', 'ef scale']
    
    for d in data:
        evidence = d['answer_found'].get('evidence', '').lower()
        judgement = d.get('final_judgement', '').lower()
        combined = evidence + ' ' + judgement
        
        step = d['answer_found'].get('step_index', -1)
        
        # Determine domain
        domain = 'Unknown'
        if any(kw in combined for kw in cooking_keywords):
            domain = 'Cooking/Food'
        elif any(kw in combined for kw in speaking_keywords):
            domain = 'Public Speaking'
        elif any(kw in combined for kw in weather_keywords):
            domain = 'Weather/News'
        
        if step >= 0:
            domain_stats[domain]['found'] += 1
            domain_stats[domain]['steps'].append(step)
        else:
            domain_stats[domain]['not_found'] += 1
    
    print("\nDomain Performance Summary:")
    print(f"{'Domain':<20} {'Found':>6} {'Failed':>7} {'Rate%':>7} {'AvgStep':>8}")
    print("-" * 55)
    for domain, stats in sorted(domain_stats.items(), key=lambda x: x[1]['found']+x[1]['not_found'], reverse=True):
        total = stats['found'] + stats['not_found']
        rate = stats['found']/total*100 if total > 0 else 0
        avg = sum(stats['steps'])/len(stats['steps']) if stats['steps'] else 0
        print(f"{domain:<20} {stats['found']:>6} {stats['not_found']:>7} {rate:>6.1f}% {avg:>7.1f}")
    print()
    
    # =========================================
    # 8. EFFICIENCY ANALYSIS
    # =========================================
    print("### 8. EFFICIENCY INSIGHTS ###")
    print("-" * 40)
    
    # Most efficient traces
    efficient = sorted(found_traces, key=lambda x: x['answer_found']['step_index'])[:10]
    
    print("\nMost Efficient Traces (Earliest Step Found):")
    for d in efficient:
        step = d['answer_found']['step_index']
        worker = d['answer_found'].get('worker', 'Unknown')
        tools = [t.get('tool', '') for t in d['answer_found'].get('tools_involved', [])]
        print(f"  {d['trace_id']}: Step {step} | Worker: {worker}")
        print(f"    Tools: {tools}")
    
    # Least efficient traces
    inefficient = sorted(found_traces, key=lambda x: x['answer_found']['step_index'], reverse=True)[:10]
    
    print("\nLeast Efficient Traces (Latest Step Found):")
    for d in inefficient:
        step = d['answer_found']['step_index']
        worker = d['answer_found'].get('worker', 'Unknown')
        tools = [t.get('tool', '') for t in d['answer_found'].get('tools_involved', [])]
        print(f"  {d['trace_id']}: Step {step} | Worker: {worker}")
        print(f"    Tools: {tools}")
    print()
    
    # =========================================
    # 9. CRITICAL INSIGHTS
    # =========================================
    print("### 9. CRITICAL INSIGHTS & RECOMMENDATIONS ###")
    print("-" * 40)
    
    # Insight 1: Tool efficiency correlation
    early_tools = Counter()
    late_tools = Counter()
    
    for d in found_traces:
        step = d['answer_found']['step_index']
        tools = [t.get('tool', '') for t in d['answer_found'].get('tools_involved', [])]
        if step <= 25:
            for t in tools:
                early_tools[t] += 1
        else:
            for t in tools:
                late_tools[t] += 1
    
    print("\n1. Tools Used in Fast vs Slow Traces:")
    print(f"   {'Tool':<40} {'Fast (≤25)':>10} {'Slow (>25)':>10}")
    print("   " + "-" * 60)
    all_tools = set(early_tools.keys()) | set(late_tools.keys())
    for tool in sorted(all_tools, key=lambda x: early_tools[x] + late_tools[x], reverse=True)[:10]:
        print(f"   {tool:<40} {early_tools[tool]:>10} {late_tools[tool]:>10}")
    
    # Insight 2: Worker specialization
    print("\n2. Worker Specialization:")
    specialized_workers = {
        'cooking': ['kg_cooking_explorer', 'chicken_pan_sauce_visual_worker', 'steak_sear_visual_worker', 'tomato_sauce_worker', 'egg_cheese_sauce_worker'],
        'speaking': ['kg_public_speaking_search', 'breathing_techniques_asr_worker', 'audio_vocal_authority_search', 'kg_public_speaking_explorer'],
        'weather': ['kg_infrastructure_search_worker', 'visual_event_search_worker', 'tornado_segment_finder'],
    }
    
    for domain, workers in specialized_workers.items():
        domain_found = sum(w['found'] for w in worker_summary if w['worker'] in workers)
        domain_total = sum(w['total'] for w in worker_summary if w['worker'] in workers)
        if domain_total > 0:
            print(f"   {domain}: {domain_found}/{domain_total} found ({domain_found/domain_total*100:.1f}%)")
    
    # Insight 3: Multi-tool traces
    multi_tool_count = len([d for d in found_traces if len(d['answer_found'].get('tools_involved', [])) >= 3])
    single_tool_count = len([d for d in found_traces if len(d['answer_found'].get('tools_involved', [])) == 1])
    
    print(f"\n3. Multi-Tool vs Single-Tool Traces:")
    print(f"   Single-tool traces: {single_tool_count} ({single_tool_count/len(found_traces)*100:.1f}%)")
    print(f"   Multi-tool (≥3) traces: {multi_tool_count} ({multi_tool_count/len(found_traces)*100:.1f}%)")
    
    avg_step_single = sum(d['answer_found']['step_index'] for d in found_traces if len(d['answer_found'].get('tools_involved', [])) == 1) / single_tool_count if single_tool_count > 0 else 0
    avg_step_multi = sum(d['answer_found']['step_index'] for d in found_traces if len(d['answer_found'].get('tools_involved', [])) >= 3) / multi_tool_count if multi_tool_count > 0 else 0
    
    print(f"   Avg step (single-tool): {avg_step_single:.1f}")
    print(f"   Avg step (multi-tool): {avg_step_multi:.1f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    deep_analysis()
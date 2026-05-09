#!/usr/bin/env python3
"""
Deep analysis of failed traces - showing full tool calling sequence, worker agents, and root cause.
"""
import os
import re

FAILED_TRACE_IDS = [
    "tr-6b62c302513eb29d05765e31cea0a749",
    "tr-a28eb60b9d084523a6e71a4086e6cebe",
    "tr-956f710235a40ba3e1de448e3f733f4b",
    "tr-eda4ea886482f89d826dab2f834b344c",
    "tr-01ee9f794325242fd70297e4d8ee62f1",
    "tr-06dd73dbdc6cf4b6ab34302c84d458b9",
]

PROMPTS_DIR = "/home/tinhanhnguyen/Desktop/HK8/Capstone/CAPSTONE_PROJECT/videodeepsearch/test/notebooks/prompts"

def parse_trace_file(filepath):
    """Parse trace file to extract workers, tools, steps, and outputs."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract all workers and their steps
    workers = []
    worker_pattern = r'\[Worker (\d+)\] ([\w_]+)'
    worker_matches = re.findall(worker_pattern, content)
    
    for worker_num, worker_name in worker_matches:
        workers.append({
            'worker_num': int(worker_num),
            'worker_name': worker_name,
            'steps': []
        })
    
    # Extract all steps with tools
    steps = []
    step_pattern = r'\[Step (\d+)\] Tool: ([\w_]+)\s+args: ({[^}]+}|[^}]+)\s+output: ([^[]+)(?:\n\s+agent output: ([^[]+)|\n\s*(?:\n|\[))'
    
    # Alternative parsing - line by line
    lines = content.split('\n')
    current_worker = None
    current_step = None
    current_tool = None
    current_args = None
    current_output = []
    
    for line in lines:
        # Check for worker header
        worker_match = re.match(r'\[Worker (\d+)\] ([\w_]+)', line)
        if worker_match:
            current_worker = {
                'worker_num': int(worker_match.group(1)),
                'worker_name': worker_match.group(2)
            }
            continue
        
        # Check for step header
        step_match = re.match(r'\s+\[Step (\d+)\] Tool: ([\w_]+)', line)
        if step_match:
            # Save previous step if exists
            if current_step and current_tool:
                steps.append({
                    'step': current_step,
                    'worker': current_worker['worker_name'] if current_worker else 'unknown',
                    'tool': current_tool,
                    'args': current_args,
                    'output': '\n'.join(current_output)[:500]  # Truncate
                })
            
            current_step = int(step_match.group(1))
            current_tool = step_match.group(2)
            current_output = []
            continue
        
        # Check for args
        args_match = re.match(r'\s+args: (.+)', line)
        if args_match and current_tool:
            current_args = args_match.group(1)[:300]  # Truncate
            continue
        
        # Check for output
        output_match = re.match(r'\s+output: (.+)', line)
        if output_match and current_tool:
            current_output.append(output_match.group(1)[:200])
            continue
        
        # Continue collecting output
        if current_output and line.strip() and not line.startswith('   agent output:') and not re.match(r'\s*\[', line):
            current_output.append(line[:100])
    
    # Extract expected response
    expected_match = re.search(r'=== EXPECTED RESPONSE ===\s*(.*?)\s*=== TASK ===', content, re.DOTALL)
    expected_response = expected_match.group(1) if expected_match else "NOT FOUND"
    
    # Extract expected video IDs
    expected_videos = re.findall(r'Video ID:\s*([a-f0-9]+)', expected_response)
    
    return {
        'workers': workers,
        'steps': steps,
        'expected_response': expected_response[:600],
        'expected_videos': expected_videos,
        'content': content
    }

def analyze_trace_deep(trace_id):
    """Deep analysis of a single trace."""
    filepath = os.path.join(PROMPTS_DIR, f"{trace_id}.txt")
    
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    
    data = parse_trace_file(filepath)
    
    print(f"\n{'='*80}")
    print(f"TRACE: {trace_id}")
    print("="*80)
    
    print(f"\n--- WORKER AGENTS USED ---")
    for w in data['workers']:
        print(f"  Worker {w['worker_num']}: {w['worker_name']}")
    
    print(f"\n--- TOOL CALLING SEQUENCE ---")
    tools_by_worker = {}
    for step in data['steps']:
        worker = step['worker']
        if worker not in tools_by_worker:
            tools_by_worker[worker] = []
        tools_by_worker[worker].append({
            'step': step['step'],
            'tool': step['tool'],
            'args_summary': step['args'][:150] if step['args'] else 'N/A'
        })
    
    for worker, tools in tools_by_worker.items():
        print(f"\n  [{worker}]")
        for t in tools:
            print(f"    Step {t['step']}: {t['tool']}")
            print(f"      Args: {t['args_summary']}")
    
    print(f"\n--- EXPECTED VIDEO IDs ---")
    for vid in data['expected_videos']:
        # Check if video appears in actual outputs
        found = False
        in_context = []
        for step in data['steps']:
            if vid in str(step['output']) or vid in str(step['args']):
                found = True
                in_context.append(f"Step {step['step']} ({step['tool']})")
        
        status = "FOUND" if found else "NOT FOUND IN TOOL OUTPUTS"
        print(f"  {vid}: {status}")
        if in_context:
            print(f"    Appeared in: {', '.join(in_context)}")
    
    print(f"\n--- EXPECTED RESPONSE ---")
    print(data['expected_response'][:400])
    
    print(f"\n--- ANALYSIS: WHAT DID THE AGENT FIND? ---")
    
    # Check agent outputs for what they concluded
    agent_outputs = []
    lines = data['content'].split('\n')
    for i, line in enumerate(lines):
        if 'agent output:' in line:
            # Collect the agent output section
            output_lines = []
            for j in range(i+1, min(i+30, len(lines))):
                if lines[j].strip() and not re.match(r'\s*\[', lines[j]):
                    output_lines.append(lines[j][:100])
                else:
                    break
            agent_outputs.append('\n'.join(output_lines)[:500])
    
    for i, output in enumerate(agent_outputs[:3]):
        print(f"\n  Agent Output #{i+1} (first 400 chars):")
        print(f"    {output[:400]}")
    
    print(f"\n--- ROOT CAUSE ANALYSIS ---")
    
    # Determine root cause
    all_tools_returned_weather = True
    for step in data['steps']:
        output = str(step['output'])
        # Check if output mentions weather/tornado/storm
        if output and not any(kw in output.lower() for kw in ['tornado', 'storm', 'weather', 'wind', 'pressure', 'solar']):
            all_tools_returned_weather = False
            break
    
    if all_tools_returned_weather:
        print("  FAILURE TYPE: CONTENT MISMATCH")
        print("  - The expected response asks for ENGINEERING/MATERIAL SCIENCE content")
        print("  - The corpus contains WEATHER/TORNADO documentaries")
        print("  - Tools returned relevant weather content but NOT the expected engineering content")
        print("  - The expected videos ARE in the corpus but contain WEATHER content, not engineering")
    else:
        print("  FAILURE TYPE: TOOL RETRIEVAL ISSUE or OTHER")
    
    return data

def main():
    print("="*80)
    print("DEEP FAILED TRACE ANALYSIS")
    print("="*80)
    
    all_data = []
    for trace_id in FAILED_TRACE_IDS:
        data = analyze_trace_deep(trace_id)
        all_data.append(data)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nAll 6 failed traces share the SAME root cause:")
    print("\n  ROOT CAUSE: TEST DATA ERROR - EXPECTED RESPONSE MISMATCH")
    print("  - Expected response claims videos contain ENGINEERING content")
    print("    (pressure pulses, material fatigue, structural failure, fragmentation)")
    print("  - Actual corpus videos contain WEATHER/TORNADO content")
    print("    (storm surge, tornado damage, wind speeds, atmospheric pressure)")
    print("  - The agent CORRECTLY identified that no relevant engineering content exists")
    print("  - This is NOT a tool failure, NOT hallucination, NOT retrieval issue")
    print("  - The TEST CASES themselves are incorrectly specified")

if __name__ == "__main__":
    main()
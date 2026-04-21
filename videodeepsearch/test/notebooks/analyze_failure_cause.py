#!/usr/bin/env python3
"""
Analyze failed traces to determine root cause of failure.
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

def extract_expected_response(content):
    """Extract the expected response section from trace file."""
    match = re.search(r'=== EXPECTED RESPONSE ===\s*(.*?)\s*=== TASK ===', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_expected_video_ids(expected_response):
    """Extract video IDs from expected response."""
    video_ids = re.findall(r'Video ID:\s*([a-f0-9]+)', expected_response)
    return video_ids

def check_video_in_trace(content, video_id):
    """Check if a video ID appears in tool outputs (not agent output)."""
    # Find tool outputs section (before agent output)
    lines = content.split('\n')
    found_in_tool = False
    found_in_agent = False
    
    for i, line in enumerate(lines):
        if video_id in line:
            # Check if it's in a tool output section (has "output:" before it)
            context_start = max(0, i - 10)
            context = '\n'.join(lines[context_start:i+1])
            
            if 'output:' in context or 'Tool:' in context:
                # Check if it's actually in tool output vs agent summary
                if 'agent output:' in context or '## Step' in context or '===' in context:
                    found_in_agent = True
                else:
                    found_in_tool = True
            else:
                found_in_agent = True
    
    return found_in_tool, found_in_agent

def extract_tool_outputs(content):
    """Extract all tool outputs from the trace."""
    tool_outputs = []
    lines = content.split('\n')
    
    current_tool = None
    current_output = []
    in_output = False
    
    for line in lines:
        if re.match(r'\s*\[Step \d+\] Tool:', line):
            if current_tool and current_output:
                tool_outputs.append({
                    'tool': current_tool,
                    'output': '\n'.join(current_output)
                })
            current_tool = re.search(r'Tool:\s*(\w+)', line).group(1) if re.search(r'Tool:\s*(\w+)', line) else 'unknown'
            current_output = []
            in_output = False
        elif 'output:' in line and current_tool:
            in_output = True
        elif in_output and current_tool:
            if line.strip() and not line.startswith('   agent output:') and not re.match(r'\s*\[Step', line):
                current_output.append(line)
            else:
                in_output = False
                if current_output:
                    tool_outputs.append({
                        'tool': current_tool,
                        'output': '\n'.join(current_output)
                    })
                    current_output = []
    
    return tool_outputs

def analyze_trace(trace_id):
    """Analyze a single trace."""
    filepath = os.path.join(PROMPTS_DIR, f"{trace_id}.txt")
    
    if not os.path.exists(filepath):
        return {"error": f"Trace file not found: {filepath}"}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    expected_response = extract_expected_response(content)
    expected_video_ids = extract_expected_video_ids(expected_response or "")
    
    tool_outputs = extract_tool_outputs(content)
    
    # Check if expected video IDs appear in tool outputs
    video_analysis = {}
    for video_id in expected_video_ids:
        in_tool, in_agent = check_video_in_trace(content, video_id)
        video_analysis[video_id] = {
            'found_in_tool_output': in_tool,
            'found_in_agent_summary': in_agent
        }
    
    # Check if expected response content keywords appear in tool outputs
    all_tool_outputs_text = '\n'.join([t['output'] for t in tool_outputs])
    
    # Extract key phrases from expected response
    key_phrases = []
    if expected_response:
        # Extract numbered points
        points = re.findall(r'\d\)\s*([^.]+\.?)', expected_response)
        key_phrases.extend(points)
    
    phrase_analysis = {}
    for phrase in key_phrases[:5]:  # Check first 5 key phrases
        phrase_clean = phrase.strip()[:50]  # First 50 chars
        found = phrase_clean.lower() in all_tool_outputs_text.lower()
        phrase_analysis[phrase_clean] = found
    
    # Determine failure type
    all_videos_in_tool = all([v['found_in_tool_output'] for v in video_analysis.values()])
    all_videos_in_agent = all([v['found_in_agent_summary'] for v in video_analysis.values()])
    
    if not expected_video_ids:
        failure_type = "NO_EXPECTED_VIDEO_ID"
    elif all_videos_in_tool:
        failure_type = "FOUND_IN_TOOL_BUT_NOT_EXTRACTED_CORRECTLY"
    elif all_videos_in_agent:
        failure_type = "FOUND_IN_AGENT_SUMMARY_BUT_NOT_IN_TOOL_OUTPUT"
    else:
        failure_type = "VIDEO_NOT_IN_CORPUS_OR_WRONG_QUERY"
    
    return {
        'trace_id': trace_id,
        'expected_response': expected_response[:500] if expected_response else None,
        'expected_video_ids': expected_video_ids,
        'video_analysis': video_analysis,
        'key_phrases_analysis': phrase_analysis,
        'total_tool_calls': len(tool_outputs),
        'tools_used': list(set([t['tool'] for t in tool_outputs])),
        'failure_type': failure_type,
    }

def main():
    print("="*80)
    print("FAILED TRACE ANALYSIS")
    print("="*80)
    
    for trace_id in FAILED_TRACE_IDS:
        print(f"\n{'='*80}")
        print(f"TRACE: {trace_id}")
        print("="*80)
        
        analysis = analyze_trace(trace_id)
        
        if 'error' in analysis:
            print(f"ERROR: {analysis['error']}")
            continue
        
        print(f"\nEXPECTED VIDEO IDs: {analysis['expected_video_ids']}")
        print(f"\nVIDEO ANALYSIS:")
        for vid, status in analysis['video_analysis'].items():
            print(f"  {vid}:")
            print(f"    Found in tool output: {status['found_in_tool_output']}")
            print(f"    Found in agent summary: {status['found_in_agent_summary']}")
        
        print(f"\nKEY PHRASES FROM EXPECTED RESPONSE:")
        for phrase, found in analysis['key_phrases_analysis'].items():
            print(f"  '{phrase[:40]}...': Found={found}")
        
        print(f"\nTOOLS USED ({analysis['total_tool_calls']} calls): {analysis['tools_used']}")
        
        print(f"\nFAILURE TYPE: {analysis['failure_type']}")
        
        print(f"\nEXPECTED RESPONSE (first 500 chars):")
        print(f"  {analysis['expected_response']}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    failure_types = {}
    for trace_id in FAILED_TRACE_IDS:
        analysis = analyze_trace(trace_id)
        if 'error' not in analysis:
            ft = analysis['failure_type']
            failure_types[ft] = failure_types.get(ft, 0) + 1
    
    for ft, count in failure_types.items():
        print(f"  {ft}: {count} traces")

if __name__ == "__main__":
    main()
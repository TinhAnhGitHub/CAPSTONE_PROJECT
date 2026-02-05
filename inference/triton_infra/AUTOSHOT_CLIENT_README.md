# Autoshot Triton Client Documentation

This directory contains multiple Triton client implementations for the **autoshot** model (shot boundary detection using TransNetV2).

## Model Information

- **Model Name**: autoshot
- **Platform**: ONNX Runtime
- **Input**:
  - Name: `input_frames`
  - Shape: `[1, 100, 27, 48, 3]`
  - Type: UINT8
  - Description: 100 video frames resized to 27x48 pixels, RGB format

- **Output**:
  - Name: `shot_boundary_prob`
  - Shape: `[1, 100, 1]`
  - Type: FP32
  - Description: Shot boundary probability for each of the 100 frames

## Client Implementations

### 1. Full-Featured HTTP Client (`autoshot_client.py`)

Comprehensive client with preprocessing, video file support, and health checks.

**Features**:
- Server health checking
- Model metadata retrieval
- Frame preprocessing (resizing, padding)
- Video file processing with OpenCV
- Shot boundary detection with configurable threshold

**Usage**:
```python
from autoshot_client import AutoshotTritonClient
import numpy as np

# Initialize client
client = AutoshotTritonClient(
    triton_url="localhost:8000",
    model_name="autoshot",
    model_version="1"
)

# Check server health
if client.check_server_health():
    # Option 1: Infer with numpy array
    frames = np.random.randint(0, 255, (100, 720, 1280, 3), dtype=np.uint8)
    probs = client.infer(frames)

    # Option 2: Infer from video file
    probs, shot_boundaries = client.infer_from_video_file(
        video_path="/path/to/video.mp4",
        start_frame=0,
        threshold=0.5
    )
    print(f"Shot boundaries: {shot_boundaries}")
```

### 2. Simple HTTP Client (`autoshot_simple_example.py`)

Minimal example for quick one-shot inference.

**Usage**:
```python
from autoshot_simple_example import simple_autoshot_inference
import numpy as np

# Prepare frames (already preprocessed to correct size)
frames = np.random.randint(0, 255, (1, 100, 27, 48, 3), dtype=np.uint8)

# Run inference
result = simple_autoshot_inference(
    frames,
    triton_url="localhost:8000",
    model_name="autoshot"
)

# Find shot boundaries
threshold = 0.5
shot_indices = np.where(result[0, :, 0] > threshold)[0]
print(f"Shot boundaries at frames: {shot_indices}")
```

### 3. gRPC Client (`autoshot_grpc_client.py`)

High-performance gRPC client with async support for production use.

**Features**:
- gRPC protocol (faster than HTTP)
- Synchronous and asynchronous inference
- Batch async inference for processing multiple requests concurrently

**Synchronous Usage**:
```python
from autoshot_grpc_client import AutoshotGRPCClient
import numpy as np

client = AutoshotGRPCClient(
    triton_url="localhost:8001",  # Note: gRPC uses port 8001
    model_name="autoshot"
)

try:
    frames = np.random.randint(0, 255, (1, 100, 27, 48, 3), dtype=np.uint8)
    result = client.infer(frames)
    print(f"Result shape: {result.shape}")
finally:
    client.close()
```

**Asynchronous Usage**:
```python
import asyncio
from autoshot_grpc_client import AutoshotGRPCClient
import numpy as np

async def main():
    client = AutoshotGRPCClient(triton_url="localhost:8001")

    try:
        frames = np.random.randint(0, 255, (1, 100, 27, 48, 3), dtype=np.uint8)
        result = await client.async_infer(frames)
        print(f"Result: {result}")
    finally:
        client.close()

asyncio.run(main())
```

**Batch Async Inference**:
```python
import asyncio
from autoshot_grpc_client import AutoshotGRPCClient
import numpy as np

async def main():
    client = AutoshotGRPCClient(triton_url="localhost:8001")

    try:
        # Create batch of requests
        batch_frames = [
            np.random.randint(0, 255, (1, 100, 27, 48, 3), dtype=np.uint8)
            for _ in range(10)
        ]

        # Process all concurrently
        results = await client.batch_async_infer(batch_frames)
        print(f"Processed {len(results)} requests")
    finally:
        client.close()

asyncio.run(main())
```

## Installation

Install required dependencies:

```bash
# For HTTP clients
pip install tritonclient[http] numpy opencv-python

# For gRPC client (in addition to above)
pip install tritonclient[grpc]
```

## Running the Examples

1. Start Triton server:
```bash
docker-compose up
```

2. Run the examples:
```bash
# Simple HTTP example
python autoshot_simple_example.py

# Full-featured HTTP client
python autoshot_client.py

# gRPC client (sync and async)
python autoshot_grpc_client.py
```

## Performance Considerations

- **HTTP Client**: Good for development and testing, easier to debug
- **gRPC Client**:
  - ~20-30% faster than HTTP for large payloads
  - Lower latency
  - Better for production
  - Supports async operations for concurrent requests

## Input Preprocessing

The autoshot model expects frames in a specific format:
- Shape: `(1, 100, 27, 48, 3)`
- Type: UINT8 (values 0-255)
- 100 consecutive frames
- Each frame resized to 27x48 pixels
- RGB color format

The `AutoshotTritonClient` includes preprocessing that:
1. Resizes frames to 27x48
2. Pads/truncates to exactly 100 frames
3. Converts to UINT8 format
4. Adds batch dimension

## Output Interpretation

The model outputs shot boundary probabilities:
- Shape: `(1, 100, 1)`
- Values: 0.0 to 1.0 (higher = more likely to be shot boundary)
- Common threshold: 0.5

Example:
```python
# Get predictions
probs = client.infer(frames)  # Shape: (1, 100, 1)

# Find shot boundaries
threshold = 0.5
shot_frames = []
for i, prob in enumerate(probs[0]):
    if prob[0] > threshold:
        shot_frames.append(i)
        print(f"Shot boundary at frame {i}: {prob[0]:.3f}")
```

## Troubleshooting

1. **Server not ready**: Ensure Triton is running on correct port
   - HTTP: port 8000
   - gRPC: port 8001

2. **Model not loaded**: Check model repository path in docker-compose.yaml

3. **Shape mismatch**: Ensure frames are preprocessed to `(1, 100, 27, 48, 3)`

4. **Memory issues**: For long videos, process in batches of 100 frames

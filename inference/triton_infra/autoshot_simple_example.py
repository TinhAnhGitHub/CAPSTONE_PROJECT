"""
Simple example of using Triton client for autoshot inference
"""

import numpy as np
import tritonclient.http as httpclient


def simple_autoshot_inference(
    frames: np.ndarray,
    triton_url: str = "localhost:8000",
    model_name: str = "autoshot"
) -> np.ndarray:
    """
    Simple one-shot inference for autoshot model

    Args:
        frames: Input frames of shape (1, 100, 27, 48, 3) in UINT8
        triton_url: Triton server URL
        model_name: Model name

    Returns:
        Shot boundary probabilities of shape (1, 100, 1)
    """
    # Create client
    client = httpclient.InferenceServerClient(url=triton_url)

    # Prepare input
    inputs = [
        httpclient.InferInput("input_frames", frames.shape, "UINT8")
    ]
    inputs[0].set_data_from_numpy(frames)

    # Prepare output
    outputs = [
        httpclient.InferRequestedOutput("shot_boundary_prob")
    ]

    # Run inference
    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )

    # Get result
    return response.as_numpy("shot_boundary_prob")


if __name__ == "__main__":
    # Create dummy input (1, 100, 27, 48, 3)
    dummy_frames = np.random.randint(0, 255, (1, 100, 27, 48, 3), dtype=np.uint8)

    print("Sending inference request to Triton...")
    print(f"Input shape: {dummy_frames.shape}, dtype: {dummy_frames.dtype}")

    # Run inference
    result = simple_autoshot_inference(dummy_frames)

    print(f"\nInference complete!")
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    print(f"\nShot boundary probabilities (first 10 frames):")
    print(result[0][:10].flatten())

    # Find shot boundaries with threshold
    threshold = 0.5
    shot_indices = np.where(result[0, :, 0] > threshold)[0]
    print(f"\nShot boundaries detected at frames (threshold={threshold}):")
    print(shot_indices)

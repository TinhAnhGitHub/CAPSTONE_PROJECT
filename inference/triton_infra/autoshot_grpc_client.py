"""
Async gRPC Triton Client for Autoshot Model
Higher performance alternative using gRPC protocol
"""

import numpy as np
import tritonclient.grpc as grpcclient
from typing import Optional
import asyncio


class AutoshotGRPCClient:
    def __init__(
        self,
        triton_url: str = "localhost:8001",  # gRPC default port
        model_name: str = "autoshot",
        model_version: str = "1",
    ):
        """
        Initialize gRPC Triton client for autoshot model

        Args:
            triton_url: Triton server gRPC URL (default: localhost:8001)
            model_name: Model name in Triton (default: autoshot)
            model_version: Model version (default: 1)
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.client = grpcclient.InferenceServerClient(url=triton_url)

    def check_server_health(self) -> bool:
        """Check if Triton server is alive and model is ready"""
        try:
            if not self.client.is_server_live():
                print("Triton server is not live")
                return False

            if not self.client.is_server_ready():
                print("Triton server is not ready")
                return False

            if not self.client.is_model_ready(self.model_name, self.model_version):
                print(f"Model {self.model_name} version {self.model_version} is not ready")
                return False

            print(f"Server and model {self.model_name} are ready")
            return True
        except Exception as e:
            print(f"Error checking server health: {e}")
            return False

    def infer(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """
        Run synchronous inference on video frames

        Args:
            frames: Input frames of shape (1, 100, 27, 48, 3) in UINT8

        Returns:
            Shot boundary probabilities of shape (1, 100, 1) or None if failed
        """
        try:
            # Prepare input
            inputs = [
                grpcclient.InferInput(
                    "input_frames",
                    frames.shape,
                    "UINT8"
                )
            ]
            inputs[0].set_data_from_numpy(frames)

            # Prepare output
            outputs = [
                grpcclient.InferRequestedOutput("shot_boundary_prob")
            ]

            # Send inference request
            response = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs
            )

            # Get output
            shot_probs = response.as_numpy("shot_boundary_prob")

            return shot_probs

        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    async def async_infer(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """
        Run asynchronous inference on video frames

        Args:
            frames: Input frames of shape (1, 100, 27, 48, 3) in UINT8

        Returns:
            Shot boundary probabilities of shape (1, 100, 1) or None if failed
        """
        try:
            # Prepare input
            inputs = [
                grpcclient.InferInput(
                    "input_frames",
                    frames.shape,
                    "UINT8"
                )
            ]
            inputs[0].set_data_from_numpy(frames)

            # Prepare output
            outputs = [
                grpcclient.InferRequestedOutput("shot_boundary_prob")
            ]

            # Send async inference request
            response = await self.client.async_infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs
            )

            # Get output
            shot_probs = response.as_numpy("shot_boundary_prob")

            return shot_probs

        except Exception as e:
            print(f"Error during async inference: {e}")
            return None

    async def batch_async_infer(self, batch_frames: list) -> list:
        """
        Run multiple inferences concurrently

        Args:
            batch_frames: List of frame arrays, each of shape (1, 100, 27, 48, 3)

        Returns:
            List of shot boundary probabilities
        """
        tasks = [self.async_infer(frames) for frames in batch_frames]
        results = await asyncio.gather(*tasks)
        return results

    def close(self):
        """Close the client connection"""
        self.client.close()


async def main_async():
    """Example async usage"""
    # Initialize client
    client = AutoshotGRPCClient(
        triton_url="localhost:8001",
        model_name="autoshot",
        model_version="1"
    )

    try:
        # Check server health
        if not client.check_server_health():
            print("Server is not ready")
            return

        # Example: Single async inference
        print("\n=== Single async inference ===")
        frames = np.random.randint(0, 255, (1, 100, 27, 48, 3), dtype=np.uint8)
        result = await client.async_infer(frames)

        if result is not None:
            print(f"Output shape: {result.shape}")
            print(f"First 5 probabilities: {result[0][:5].flatten()}")

        # Example: Batch async inference
        print("\n=== Batch async inference ===")
        batch_size = 5
        batch_frames = [
            np.random.randint(0, 255, (1, 100, 27, 48, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        results = await client.batch_async_infer(batch_frames)
        print(f"Processed {len(results)} requests concurrently")

        for i, result in enumerate(results):
            if result is not None:
                shot_count = np.sum(result[0] > 0.5)
                print(f"Request {i+1}: Detected {shot_count} shot boundaries")

    finally:
        client.close()


def main_sync():
    """Example sync usage"""
    # Initialize client
    client = AutoshotGRPCClient(
        triton_url="localhost:8001",
        model_name="autoshot",
        model_version="1"
    )

    try:
        # Check server health
        if not client.check_server_health():
            print("Server is not ready")
            return

        # Synchronous inference
        print("\n=== Synchronous inference ===")
        frames = np.random.randint(0, 255, (1, 100, 27, 48, 3), dtype=np.uint8)
        result = client.infer(frames)

        if result is not None:
            print(f"Output shape: {result.shape}")
            print(f"First 5 probabilities: {result[0][:5].flatten()}")

            # Find shot boundaries
            threshold = 0.5
            shot_indices = np.where(result[0, :, 0] > threshold)[0]
            print(f"Shot boundaries at frames: {shot_indices}")

    finally:
        client.close()


if __name__ == "__main__":
    # Run sync example
    print("Running synchronous example...")
    main_sync()

    # Run async example
    print("\n" + "="*50)
    print("Running asynchronous example...")
    asyncio.run(main_async())

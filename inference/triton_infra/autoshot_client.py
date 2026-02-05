"""
Triton Inference Client for Autoshot Model
Performs shot boundary detection on video frames
"""

import numpy as np
import tritonclient.http as httpclient
from typing import Optional, Tuple
import cv2


class AutoshotTritonClient:
    def __init__(
        self,
        triton_url: str = "localhost:8000",
        model_name: str = "autoshot",
        model_version: str = "1",
    ):
        """
        Initialize Triton client for autoshot model

        Args:
            triton_url: Triton server URL (default: localhost:8000)
            model_name: Model name in Triton (default: autoshot)
            model_version: Model version (default: 1)
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.client = httpclient.InferenceServerClient(url=triton_url)

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

    def get_model_metadata(self):
        """Get model metadata from Triton server"""
        try:
            metadata = self.client.get_model_metadata(self.model_name, self.model_version)
            print(f"Model metadata: {metadata}")
            return metadata
        except Exception as e:
            print(f"Error getting model metadata: {e}")
            return None

    def preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Preprocess video frames for autoshot model

        Args:
            frames: numpy array of shape (num_frames, height, width, channels)
                   Expected to be 100 frames of RGB images

        Returns:
            Preprocessed frames ready for inference (1, 100, 27, 48, 3) in UINT8
        """
        # Expected shape: (1, 100, 27, 48, 3)
        target_height, target_width = 27, 48
        num_frames = 100

        # Ensure we have the right number of frames
        if frames.shape[0] != num_frames:
            print(f"Warning: Expected {num_frames} frames, got {frames.shape[0]}")
            # Pad or truncate
            if frames.shape[0] < num_frames:
                # Pad with last frame
                padding = np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)
                frames = np.concatenate([frames, padding], axis=0)
            else:
                # Truncate
                frames = frames[:num_frames]

        # Resize frames to target size
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (target_width, target_height))
            resized_frames.append(resized)

        resized_frames = np.array(resized_frames, dtype=np.uint8)

        # Add batch dimension: (100, 27, 48, 3) -> (1, 100, 27, 48, 3)
        resized_frames = np.expand_dims(resized_frames, axis=0)

        return resized_frames

    def infer(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference on video frames

        Args:
            frames: Input frames as numpy array
                   Should be shape (100, height, width, 3) or already preprocessed (1, 100, 27, 48, 3)

        Returns:
            Shot boundary probabilities of shape (1, 100, 1) or None if failed
        """
        try:
            # Preprocess if needed
            if frames.shape != (1, 100, 27, 48, 3):
                print(f"Preprocessing frames from shape {frames.shape}")
                frames = self.preprocess_frames(frames)

            # Prepare input
            inputs = [
                httpclient.InferInput(
                    "input_frames",
                    frames.shape,
                    "UINT8"
                )
            ]
            inputs[0].set_data_from_numpy(frames)

            # Prepare output
            outputs = [
                httpclient.InferRequestedOutput("shot_boundary_prob")
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

    def infer_from_video_file(
        self,
        video_path: str,
        start_frame: int = 0,
        threshold: float = 0.5
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        Run inference on video file

        Args:
            video_path: Path to video file
            start_frame: Starting frame index (default: 0)
            threshold: Threshold for shot boundary detection (default: 0.5)

        Returns:
            Tuple of (probabilities, shot_boundaries)
            - probabilities: Shot boundary probabilities (1, 100, 1)
            - shot_boundaries: List of frame indices where shots are detected
        """
        try:
            # Read video frames
            cap = cv2.VideoCapture(video_path)

            # Skip to start frame
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            for _ in range(100):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                print("No frames read from video")
                return None, None

            frames = np.array(frames)
            print(f"Read {len(frames)} frames from video")

            # Run inference
            probs = self.infer(frames)

            if probs is None:
                return None, None

            # Find shot boundaries
            shot_boundaries = []
            for i, prob in enumerate(probs[0]):
                if prob[0] > threshold:
                    shot_boundaries.append(start_frame + i)

            print(f"Detected {len(shot_boundaries)} shot boundaries")

            return probs, shot_boundaries

        except Exception as e:
            print(f"Error processing video file: {e}")
            return None, None


def main():
    """Example usage of AutoshotTritonClient"""

    # Initialize client
    client = AutoshotTritonClient(
        triton_url="localhost:8000",
        model_name="autoshot",
        model_version="1"
    )

    # Check server health
    if not client.check_server_health():
        print("Server is not ready")
        return

    # Get model metadata
    client.get_model_metadata()

    # Example 1: Inference with random frames (for testing)
    print("\n=== Example 1: Random frames ===")
    random_frames = np.random.randint(0, 255, (100, 720, 1280, 3), dtype=np.uint8)
    probs = client.infer(random_frames)

    if probs is not None:
        print(f"Output shape: {probs.shape}")
        print(f"Shot boundary probabilities (first 10): {probs[0][:10].flatten()}")

        # Find shot boundaries (threshold = 0.5)
        threshold = 0.5
        shot_boundaries = [i for i, p in enumerate(probs[0]) if p[0] > threshold]
        print(f"Shot boundaries detected at frames: {shot_boundaries}")

    # Example 2: Inference from video file (uncomment and provide video path)
    # print("\n=== Example 2: Video file ===")
    # video_path = "/path/to/your/video.mp4"
    # probs, shot_boundaries = client.infer_from_video_file(
    #     video_path=video_path,
    #     start_frame=0,
    #     threshold=0.5
    # )
    #
    # if probs is not None:
    #     print(f"Shot boundaries: {shot_boundaries}")


if __name__ == "__main__":
    main()

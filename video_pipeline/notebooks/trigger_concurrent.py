import asyncio
import uuid
from bson import ObjectId
from prefect.deployments import run_deployment

DEPLOYMENT_NAME = "Single Video Processing Flow/local-deployment"


async def trigger_video_pipeline(
    video_s3_path: str,
    additional_flow_description: str = "",
    timeout: int = 300,
):
    video_id = str(ObjectId())
    user_id = str(uuid.uuid4())

    result = await run_deployment(  # type: ignore
        name=DEPLOYMENT_NAME,
        parameters={
            "video_id": video_id,
            "user_id": user_id,
            "video_file_path": video_s3_path,
            "additional_flow_description": additional_flow_description,
        },
        timeout=timeout,
    )

    return {
        "video_id": video_id,
        "user_id": user_id,
        "flow_run": result,
    }


async def trigger_concurrent(video_paths: list[str], timeout: int = 300):
    print(f"Triggering {len(video_paths)} deployments concurrently...\n")

    results = await asyncio.gather(
        *[
            trigger_video_pipeline(path, timeout=timeout)
            for path in video_paths
        ],
        return_exceptions=True,
    )

    for i, (path, result) in enumerate(zip(video_paths, results)):
        print(f"[{i+1}] {path}")
        if isinstance(result, Exception):
            print(f"     FAILED: {result}")
        else:
            run = result["flow_run"]
            print(f"     video_id : {result['video_id']}")
            print(f"     user_id  : {result['user_id']}")
            print(f"     run name : {run.name}")
            print(f"     state    : {run.state_name}")
        print()

    return results


if __name__ == "__main__":
    video_paths = [
        "s3://video/veratasium1.mp4",
        # "s3://video/veratasium2.mp4",
        # "s3://video/veratasium3.mp4",
        # "s3://video/veratasium1.mp4",
        # "s3://video/veratasium2.mp4",
        # "s3://video/veratasium3.mp4",
    ]

    asyncio.run(trigger_concurrent(video_paths))

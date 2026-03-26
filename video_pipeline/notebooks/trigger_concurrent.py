import asyncio
import uuid
from bson import ObjectId
from prefect.deployments import run_deployment

DEPLOYMENT_NAME = "Single Video Processing Flow/local-deployment"

async def trigger_video_pipeline(
    video_id: str,
    video_s3_path: str,
    timeout: int = 300,
):
    user_id = "tinhanhuser"

    result = await run_deployment(  # type: ignore
        name=DEPLOYMENT_NAME,
        parameters={
            "video_id": video_id,
            "user_id": user_id,
            "video_file_path": video_s3_path,
        },
        timeout=timeout,
    )

    return {
        "video_id": video_id,
        "user_id": user_id,
        "flow_run": result,
    }


async def trigger_concurrent(video_paths: list[tuple[str, str]], timeout: int = 300):
    print(f"Triggering {len(video_paths)} deployments concurrently...\n")

    results = await asyncio.gather(
        *[
            trigger_video_pipeline(path[0], path[1], timeout=timeout)
            for path in video_paths
        ],
        return_exceptions=True,
    )

    for i, (path, result) in enumerate(zip(video_paths, results)):
        print(f"[{i+1}] {path}")
        if isinstance(result, Exception):
            print(f"     FAILED: {result}")
        else:
            run = result["flow_run"] #type:ignore
            print(f"     video_id : {result['video_id']}") #type:ignore
            print(f"     user_id  : {result['user_id']}") #type:ignore
            print(f"     run name : {run.name}") #type:ignore
            print(f"     state    : {run.state_name}") #type:ignore
        print()

    return results


if __name__ == "__main__":
    video_paths = [
        # ('0e64f1c0da591ca67f07b7f9', 's3://video/veritasium_00.mp4'),
        # ('c98019fd17ff4420ea47eee7', 's3://video/veritasium_01.mp4'),
        # ('eee3534844edab3ebb4d6ceb', 's3://video/veritasium_02.mp4'),
        # ('f784df5c6c00eb123f9820dc', 's3://video/veritasium_03.mp4'),
        ('c510fac771767405c891bf64', 's3://video/veritasium_04.mp4'),
        # ('4a081d6f16c83d089f67161b', 's3://video/veritasium_05.mp4'),
        # ('92ba4b2e27f460945fded9e5', 's3://video/veritasium_06.mp4'),
        # ('833d951b499a8e12d53803b1', 's3://video/veritasium_07.mp4'),
        # ('533914541945c2060c128da3', 's3://video/veritasium_08.mp4'),
        # ('bb3bd8f2c7edb86b032f53f3', 's3://video/veritasium_09.mp4'),
        # ('a735acb2a4ae02367b8e3e4c', 's3://video/veritasium_10.mp4'),
        # ('44c4d7ca191aa6b53b5eb4c1', 's3://video/veritasium_11.mp4'),
        # ('384209bac6311285477382d8', 's3://video/veritasium_12.mp4'),
        # ('946330031ead69b21354d038', 's3://video/veritasium_13.mp4'),
        # ('3636d10a2ad4787733c9700d', 's3://video/veritasium_14.mp4'),
        # ('9b17f473300a5436f0a053be', 's3://video/veritasium_15.mp4'),
        # ('1e1d300356360ed84020821c', 's3://video/veritasium_16.mp4'),
        # ('02d242459a690605ee3a8ddf', 's3://video/veritasium_17.mp4'),
        # ('c096e9a1462d83d89826cb83', 's3://video/veritasium_18.mp4'),
        # ('c45a3309ea7a5a05dcbd1ffc', 's3://video/veritasium_19.mp4'),
        # ('77f9300537d25f8fbab5bfb3', 's3://video/veritasium_20.mp4'),
        # ('0f48acd4ac783dfbdee85468', 's3://video/veritasium_21.mp4'),
        # ('b1abb34af1bc67cb712d5ffb', 's3://video/veritasium_22.mp4'),
    ]

    asyncio.run(trigger_concurrent(video_paths))

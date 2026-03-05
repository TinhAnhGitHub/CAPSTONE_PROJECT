import asyncio
from datetime import timedelta
import io
import os
from io import BytesIO
import tempfile
from moviepy import VideoFileClip
from PIL import Image
from minio import Minio

from app.model.video import Video


class MinioService:
    def __init__(self, minio_client: Minio = None):
        self.minio_client = minio_client
        self._ensure_buckets()

    def _ensure_buckets(self):
        buckets = ["avatars", "videos", "thumbnails"]
        for bucket_name in buckets:
            if not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)

                # # Set different policies per bucket
                # if bucket == "thumbnails":
                #     # Make thumbnails publicly readable
                #     policy = {
                #         "Version": "2012-10-17",
                #         "Statement": [
                #             {
                #                 "Effect": "Allow",
                #                 "Principal": {"AWS": "*"},
                #                 "Action": ["s3:GetObject"],
                #                 "Resource": [f"arn:aws:s3:::{bucket}/*"],
                #             }
                #         ],
                #     }
                #     self.client.set_bucket_policy(bucket, json.dumps(policy))

    def add_user_avatar(self, idinfo, response) -> str:
        picture_data = response.content
        picture_stream = io.BytesIO(picture_data)
        # ko có thì fallback image/jpeg
        content_type = response.headers.get("Content-Type", "image/jpeg")

        # save to minio
        self.minio_client.put_object(
            bucket_name="avatars",
            object_name=f"{idinfo['sub']}.jpg",
            data=picture_stream,
            length=len(picture_data),
            content_type=content_type,
        )
        url = self.minio_client.presigned_get_object(
            bucket_name="avatars",
            object_name=f"{idinfo['sub']}.jpg",
            expires=timedelta(days=7),  # 7 days
        )

        return url

    def save_videos(self, video_ids, files):
        """
        Saves videos & thumbnails. Returns list of dicts like:
        [
          {"video_id": "...", "video_url": "...", "thumbnail_url": "..."},
          ...
        ]
        """
        results = []

        for video_id, file in zip(video_ids, files):
            object_name = f"{video_id}.mp4"

            video_bytes = file.file.read()
            file.file.seek(0)

            # --- upload video ---
            self.minio_client.put_object(
                bucket_name="videos",
                object_name=object_name,
                data=BytesIO(video_bytes),
                length=len(video_bytes),
                part_size=10 * 1024 * 1024,
                content_type="video/mp4",
            )
            video_url = self.minio_client.presigned_get_object(
                bucket_name="videos",
                object_name=object_name,
                expires=timedelta(days=7),
            )
            # video_url = f"s3://videos/{object_name}"
            thumbnail_url, length, fps = self.generate_thumbnail(video_bytes, video_id)

            results.append(
                (
                    video_id,  # This is already a PydanticObjectId
                    video_url,
                    thumbnail_url,
                    length,
                    fps,
                )
            )

        return results

    def get_video(self, video_id):
        object_name = f"{video_id}.mp4"
        return self.minio_client.get_object("videos", object_name)

    def delete_videos(self, video_ids):
        for video_id in video_ids:
            video_object = f"{video_id}.mp4"
            thumb_object = f"{video_id}.jpg"

            self.minio_client.remove_object("videos", video_object)
            self.minio_client.remove_object("thumbnails", thumb_object)

    def generate_thumbnail(
        self, video_bytes, video_id, time=5, size=(320, 320), frame_index=None
    ):
        tmp_video = None
        fps = None
        length = None

        try:
            # Save video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp_video = tmp.name
                tmp.write(video_bytes)

            # Extract frame
            with VideoFileClip(tmp_video) as clip:
                fps = clip.fps
                length = clip.duration

                if frame_index is not None:
                    # Convert frame index to timestamp
                    timestamp = frame_index / fps
                    timestamp = min(timestamp, length)
                else:
                    frame_index = int(time * fps)
                    timestamp = min(time, length / 2)

                frame = clip.get_frame(timestamp)

            # Convert to PIL Image and resize
            image = Image.fromarray(frame)
            image.thumbnail(size, Image.Resampling.LANCZOS)

            # Save to memory
            thumb_bytes = BytesIO()
            image.save(thumb_bytes, format="JPEG", quality=85)
            thumb_bytes.seek(0)

            # Upload
            thumb_object = f"{video_id}_{frame_index}.jpg"
            self.minio_client.put_object(
                bucket_name="thumbnails",
                object_name=thumb_object,
                data=thumb_bytes,
                length=thumb_bytes.getbuffer().nbytes,
                content_type="image/jpeg",
            )

            return self.generate_thumbnail_link(video_id, frame_index), length, fps

            # return (
            #     self.minio_client.presigned_get_object(
            #         bucket_name="thumbnails",
            #         object_name=thumb_object,
            #         expires=timedelta(days=7),
            #     ),
            #     length,
            #     fps,
            # )

        finally:
            if tmp_video and os.path.exists(tmp_video):
                os.remove(tmp_video)

    def _get_thumbnail_frames(self, frame_index, video_fps, video_duration):
        """Compute 5 frame indices: -5s, -2.5s, center, +2.5s, +5s (in frame space)."""
        max_frame = int(video_duration * video_fps)
        offsets = [-5, -2.5, 0, 2.5, 5]  # seconds
        frames = []
        for offset in offsets:
            f = int(frame_index + offset * video_fps)
            f = max(0, min(f, max_frame))
            frames.append(f)
        return frames

    def _generate_timeline_thumbnail_sync(
        self, video_id, frame_index, video_duration, video_fps, size=(320, 320)
    ):
        video_object = f"{video_id}.mp4"
        response = self.minio_client.get_object("videos", video_object)
        video_bytes = response.read()

        frames = self._get_thumbnail_frames(frame_index, video_fps, video_duration)
        thumbnail_urls = [
            self.generate_thumbnail(video_bytes, video_id, size=size, frame_index=f)[0]
            for f in frames
        ]
        return thumbnail_urls

    async def generate_timeline_thumbnails(
        self, video_id, frame_index, size=(320, 320)
    ):
        # get file from minio
        # video_object = f"{video_id}.mp4"
        # response = self.minio_client.get_object("videos", video_object)
        # video_bytes = response.read()
        # get video metadata from mongodb so dont need to read video file to get duration
        video = await Video.get(video_id)
        video_duration = video.length
        video_fps = video.fps

        # offload into blocking into thread
        return await asyncio.to_thread(
            self._generate_timeline_thumbnail_sync,
            video_id,
            frame_index,
            video_duration,
            video_fps,
            size,
        )

    async def generate_thumbnail_links(self, video_id, frame_index):
        video = await Video.get(video_id)
        video_fps = video.fps
        video_duration = video.length

        frames = self._get_thumbnail_frames(frame_index, video_fps, video_duration)
        return [self.generate_thumbnail_link(video_id, f) for f in frames]

    def generate_thumbnail_link(self, video_id, frame_index):
        return f"http://100.113.186.28:9000/thumbnails/{video_id}_{frame_index}.jpg"

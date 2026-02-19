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
            expires=timedelta(days=7)  # 7 days
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

            # --- upload video ---
            self.minio_client.put_object(
                bucket_name="videos",
                object_name=object_name,
                data=file.file,
                length=file.size,
                part_size=10 * 1024 * 1024,
                content_type="video/mp4",
            )
            video_url = self.minio_client.presigned_get_object(
                bucket_name="videos",
                object_name=object_name,
                expires=timedelta(days=7),
            )
            # video_url = f"s3://videos/{object_name}"

            thumbnail_url, length, fps = self.generate_thumbnail(file, video_id)

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

    def generate_thumbnail(self, file, video_id, time=5, size=(320, 320)):
        tmp_video = None
        fps = None
        length = None 

        try:
            # Save video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp_video = tmp.name
                file.file.seek(0)
                tmp.write(file.file.read())

            # Extract frame at 5 seconds
            with VideoFileClip(tmp_video) as clip:
                # Get frame at 5 seconds (or middle if shorter)
                fps = clip.fps
                length = clip.duration
                timestamp = min(time, clip.duration / 2)
                frame = clip.get_frame(timestamp)

            # Convert to PIL Image and resize
            image = Image.fromarray(frame)
            image.thumbnail(size, Image.Resampling.LANCZOS)

            # Save to memory
            thumb_bytes = BytesIO()
            image.save(thumb_bytes, format="JPEG", quality=85)
            thumb_bytes.seek(0)

            # Upload
            thumb_object = f"{video_id}.jpg"
            self.minio_client.put_object(
                bucket_name="thumbnails",
                object_name=thumb_object,
                data=thumb_bytes,
                length=thumb_bytes.getbuffer().nbytes,
                content_type="image/jpeg",
            )

            return self.minio_client.presigned_get_object(
                bucket_name="thumbnails",
                object_name=thumb_object,
                expires=timedelta(days=7)
            ), length, fps

        finally:
            if tmp_video and os.path.exists(tmp_video):
                os.remove(tmp_video)

    def delete_videos(self, video_ids):
        for video_id in video_ids:
            video_object = f"{video_id}.mp4"
            thumb_object = f"{video_id}.jpg"

            self.minio_client.remove_object("videos", video_object)
            self.minio_client.remove_object("thumbnails", thumb_object)

    async def generate_timeline_thumbnails(self, video_id, frame_index, size=(320, 320)):
        video_duration = None
        # get file from minio
        video_object = f"{video_id}.mp4"
        file = self.minio_client.get_object("videos", video_object)
        # get video metadata from mongodb so dont need to read video file to get duration
        video = await Video.get(video_id)
        video_duration = video.length
        video_fps = video.fps

        # generate 5
        time = frame_index / video_fps
        timestamp_3 = time
        timestamp_2 = max(time - 1, 0)
        timestamp_1 = max(time-2, 0)
        timestamp_4 = min(time+1, video_duration)
        timestamp_5 = min(time+2, video_duration)
        timestamps = [timestamp_1, timestamp_2, timestamp_3, timestamp_4, timestamp_5]
        thumbnail_urls = [self.generate_thumbnail(file, video_id, t, size)[0] for t in timestamps]
        return thumbnail_urls

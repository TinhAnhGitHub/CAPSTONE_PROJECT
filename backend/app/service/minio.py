from datetime import timedelta
import io
import os
from io import BytesIO
import tempfile
from moviepy import VideoFileClip
from PIL import Image


class Minio:
    def __init__(self, minio_client=None):
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

            thumbnail_url = self.generate_thumbnail(file, video_id)

            results.append(
                (
                    video_id,  # This is already a PydanticObjectId
                    video_url,
                    thumbnail_url,
                    f"s3://videos/{object_name}",
                )
            )

        return results

    def generate_thumbnail(self, file, video_id):
        tmp_video = None
        try:
            # Save video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp_video = tmp.name
                file.file.seek(0)
                tmp.write(file.file.read())

            # Extract frame at 5 seconds
            with VideoFileClip(tmp_video) as clip:
                # Get frame at 5 seconds (or middle if shorter)
                timestamp = min(5, clip.duration / 2)
                frame = clip.get_frame(timestamp)

            # Convert to PIL Image and resize
            image = Image.fromarray(frame)
            image.thumbnail((320, 320), Image.Resampling.LANCZOS)

            # Save to memory
            thumb_bytes = BytesIO()
            image.save(thumb_bytes, format="JPEG", quality=85)
            thumb_bytes.seek(0)

            # Upload
            thumb_object = f"{video_id}.jpg"
            self.minio_client.put_object(
                "thumbnails",
                thumb_object,
                thumb_bytes,
                thumb_bytes.getbuffer().nbytes,
                content_type="image/jpeg",
            )

            return self.minio_client.presigned_get_object(
                "thumbnails", thumb_object, expires=timedelta(days=7)
            )

        finally:
            if tmp_video and os.path.exists(tmp_video):
                os.remove(tmp_video)

    def delete_videos(self, video_ids):
        for video_id in video_ids:
            video_object = f"{video_id}.mp4"
            thumb_object = f"{video_id}.jpg"

            self.minio_client.remove_object("videos", video_object)
            self.minio_client.remove_object("thumbnails", thumb_object)

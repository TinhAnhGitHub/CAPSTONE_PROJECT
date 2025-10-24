from datetime import timedelta
import io
import os
import subprocess
import tempfile

class Minio():
    def __init__(self, minio_client=None):
        self.minio_client = minio_client

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

    async def save_videos(self, video_ids: list[str], files) -> list[dict]:
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

            # --- generate thumbnail using ffmpeg ---
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
                # rewind file, write to temp path (or save file temporarily)
                file.file.seek(0)
                tmp.write(file.file.read())
            thumbnail_path = tmp_path + "_thumb.jpg"

            # Example ffmpeg command: capture frame at 5 seconds
            subprocess.run(
                [
                    "ffmpeg",
                    "-ss",
                    "00:00:05",
                    "-i",
                    tmp_path,
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",  # quality
                    thumbnail_path,
                ],
                check=True,
            )

            # Upload thumbnail
            thumb_object = f"{video_id}.jpg"
            with open(thumbnail_path, "rb") as thumb_file:
                self.minio_client.put_object(
                    bucket_name="videos",
                    object_name=thumb_object,
                    data=thumb_file,
                    length=os.path.getsize(thumbnail_path),
                    content_type="image/jpeg",
                )
            thumbnail_url = self.minio_client.presigned_get_object(
                bucket_name="videos",
                object_name=thumb_object,
                expires=timedelta(days=7),
            )

            # Clean up temp file
            os.remove(tmp_path)
            os.remove(thumbnail_path)

            results.append(
                {
                    "video_id": video_id,
                    "video_url": video_url,
                    "thumbnail_url": thumbnail_url,
                }
            )

        return results

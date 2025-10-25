from datetime import timedelta
import io
import os
import subprocess
import tempfile
import cv2
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

    async def save_videos(self, video_ids: list[str], files):
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
                {
                    "video_id": video_id,
                    "video_url": video_url,
                    "thumbnail_url": thumbnail_url,
                    "video_s3_url": "s3://videos/" + object_name,
                }
            )

        return results

    def generate_thumbnail(self, file, video_id):
        # Save to a temp file so OpenCV can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name
            file.file.seek(0)
            tmp.write(file.file.read())

        # Open the video
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("❌ Cannot open video file")

        # Get total frame count and FPS to choose middle frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        target_time = min(5, duration / 2)  # 5s or middle of the video

        # Seek to the target frame
        cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)

        success, frame = cap.read()
        if not success:
            raise RuntimeError("❌ Failed to read frame from video")

        # Save frame as JPEG
        thumbnail_path = tmp_path + "_thumb.jpg"
        cv2.imwrite(thumbnail_path, frame)

        cap.release()

        # Upload to MinIO
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

        # Cleanup
        os.remove(tmp_path)
        os.remove(thumbnail_path)

        return thumbnail_url

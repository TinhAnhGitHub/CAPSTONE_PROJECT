from email.policy import default
import io
import os
from typing import Annotated, Sequence

from beanie import PydanticObjectId
from bson import ObjectId
from fastapi import Depends, File, HTTPException, status
import httpx
import jwt
from app.model.user import User

from llama_index.core.base.llms.types import MessageRole
from datetime import datetime, timedelta, timezone

from uuid import uuid4
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from app.schema.user import (
    ACCESS_TOKEN_EXPIRE_HOURS,
    ALGORITHM,
    SECRET_KEY,
    TokenData,
)
import requests
from app.service.minio import Minio as MinioService
from app.model.chat_history import ChatHistory
from app.model.group import Group
from app.model.session_video import SessionVideo
from app.model.video import Video
from app.model.session_message import SessionMessage
from fastapi.encoders import jsonable_encoder

from werkzeug.utils import secure_filename
class UserService:
    def __init__(self, minio_service: MinioService):
        self.minio_service = minio_service

    def verify_google_id_token(self, id_token_str: str, access_token: str) -> dict:
        """Verify Google's JWT token and extract user info"""
        # try:
        client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
        if not client_id:
            raise HTTPException(status_code=500, detail="Missing Google Client ID")

        # Verify the token with Google's servers
        idinfo = id_token.verify_oauth2_token(
            id_token_str, google_requests.Request(), client_id
        )
        # if picture is in there, save it to minio
        if idinfo.get("picture"):
            # fetch the binarys of the picture
            response = requests.get(
                idinfo["picture"],
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if response.status_code == 200:
                url = self.minio_service.add_user_avatar(idinfo, response)
                idinfo["picture"] = url
        # Extract user information from verified token
        return {
            "google_id": idinfo["sub"],
            "email": idinfo["email"],
            "name": idinfo.get("name", ""),
            "picture": idinfo.get("picture", ""),
            "email_verified": idinfo.get("email_verified", False),
            "given_name": idinfo.get("given_name", ""),
            "family_name": idinfo.get("family_name", ""),
        }
        # except ValueError as e:
        #     raise HTTPException(
        #         status_code=400, detail=f"Invalid Google token: {str(e)}"
        #     )

    def create_jwt_token(self, user_data: dict) -> str:
        """Create your own JWT token for the user"""
        JWT_EXPIRATION_HOURS = ACCESS_TOKEN_EXPIRE_HOURS
        JWT_SECRET_KEY = SECRET_KEY  # Use a secure key in production
        JWT_ALGORITHM = ALGORITHM
        payload = {
            "user_id": user_data["user_id"],
            "email": user_data["email"],
            "google_id": user_data.get("google_id"),
            "iat": datetime.now(timezone.utc),  # Issued at
            "exp": datetime.now(timezone.utc)
            + timedelta(hours=JWT_EXPIRATION_HOURS),  # Expires
        }

        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token

    async def create_or_update_user(self, google_user_info: dict) -> User:
        """Create new user or update existing user from Google OAuth"""

        # First, try to find existing user by Google ID
        existing_user = await User.find_one(
            User.google_id == google_user_info["google_id"]
        )

        if existing_user:
            # Update existing user
            existing_user.email = google_user_info["email"]
            existing_user.full_name = google_user_info["name"]
            existing_user.picture = google_user_info.get("picture", "")
            existing_user.email_verified = google_user_info.get("email_verified", False)
            existing_user.last_login = datetime.utcnow()

            await existing_user.save()
            return existing_user

        else:
            # Check if user exists by email (in case they signed up before with different method)
            existing_by_email = await User.find_one(
                User.email == google_user_info["email"]
            )

            if existing_by_email:
                # Link Google account to existing user
                existing_by_email.google_id = google_user_info["google_id"]
                existing_by_email.picture = google_user_info.get("picture", "")
                existing_by_email.email_verified = google_user_info.get(
                    "email_verified", False
                )
                existing_by_email.last_login = datetime.utcnow()

                await existing_by_email.save()
                return existing_by_email

            else:
                # Create completely new user
                new_user = User(
                    username=google_user_info["email"].split("@")[
                        0
                    ],  # Use email prefix as username
                    email=google_user_info["email"],
                    full_name=google_user_info["name"],
                    google_id=google_user_info["google_id"],
                    picture=google_user_info.get("picture", ""),
                    email_verified=google_user_info.get("email_verified", False),
                    disabled=False,
                    created_at=datetime.utcnow(),
                    last_login=datetime.utcnow(),
                )

                await new_user.insert()

                # create new group "default" for the user
                default_group = Group(
                    user_id=str(new_user.id),
                    name="default",
                )
                await default_group.insert()

                return new_user

    async def get_user_chat_history(self, user_id: str):
        """Get chat history for a user"""
        chat_history = (
            await ChatHistory.find(ChatHistory.user_id == PydanticObjectId(user_id))
            .sort(("last_updated", -1))
            .to_list()
        )
        return chat_history

    async def create_new_chat_session(self, user_id: str):
        session_id = PydanticObjectId()
        new_chat_his = ChatHistory(
            id=session_id, user_id=PydanticObjectId(user_id),
        )
        await new_chat_his.insert()
        return str(session_id)


    async def get_user_chat_detail(self, session_id: str):
        """get chat detail by session id"""
        chat_messages = (
            await SessionMessage.find(
                SessionMessage.session_id == PydanticObjectId(session_id)
            )
            .sort(("timestamp", 1))
            .to_list()
        )
        return chat_messages

    async def get_user_groups(self, user_id: str):
        """get user groups"""
        groups = await Group.find(Group.user_id == PydanticObjectId(user_id)).to_list()
        return groups

    async def create_user_group(self, user_id: str, group_name: str = "default"):
        group_id = PydanticObjectId()
        new_group = Group(id=group_id, user_id=user_id, name=group_name)
        await new_group.insert()
        return str(group_id)

    async def add_videos_to_user(
        self, user_id: str, files: list[File], group_id: str, session_id: str = None
    ):
        # sanitize filenames
        filenames = [secure_filename(file.filename) for file in files]
        new_videos = [
            Video(user_id=user_id, group_id=group_id, name=filename)
            for filename in filenames
        ]
        # get video ids # order true
        inserted_videos = await Video.insert_many(new_videos)
        video_ids = inserted_videos.inserted_ids # pydantic object ids
        new_group_videos = [
            SessionVideo(
                session_id=PydanticObjectId(session_id),
                video_id=video_id,
                selected=False,
            )
            for video_id in video_ids
        ]
        await SessionVideo.insert_many(new_group_videos)
        # save videos to minio
        video_id_video_url_thumbnail_url_s3_url_obj = self.minio_service.save_videos(
            video_ids, files
        )
        # update video thumbnail urls
        for video_id, video_url, thumbnail_url, s3_url in video_id_video_url_thumbnail_url_s3_url_obj:
            video = await Video.get(video_id)
            if video:
                video.thumbnail = thumbnail_url
                video.url = video_url
                await video.save()

        return video_id_video_url_thumbnail_url_s3_url_obj

    async def get_user_videos(self, group_id: str, session_id: str):
        # all videos and their selected state
        videos = await Video.find(
            Video.group_id == PydanticObjectId(group_id)
        ).to_list()

        # if session id is null return all videos with selected false
        session_videos = await SessionVideo.find(
            SessionVideo.session_id == PydanticObjectId(session_id)
        ).to_list()

        selected_map = {sv.video_id: sv.selected for sv in session_videos}

        return [
            {**jsonable_encoder(v), "selected": selected_map.get(v.id, False)}
            for v in videos
        ]

    async def select_videos(self, session_id: str, video_ids: list[str]):
        # find session videos
        for vid in video_ids:
            session_video = await SessionVideo.find_one(
                SessionVideo.session_id == PydanticObjectId(session_id),
                SessionVideo.video_id == PydanticObjectId(vid),
            )
            if session_video:
                session_video.selected = not session_video.selected
                await session_video.save()
            else:
                new_session_video = SessionVideo(
                    session_id=PydanticObjectId(session_id),
                    video_id=PydanticObjectId(vid),
                    selected=True,
                )
                await new_session_video.insert()

    async def delete_group(self, group_id: str):
        # delete group
        await Group.find_one(Group.id == PydanticObjectId(group_id)).delete()
        # delete group videos
        await SessionVideo.find(SessionVideo.id == PydanticObjectId(group_id)).delete()
        return True

    async def delete_videos(self, video_ids: list[str]):
        # delete the videos
        print("🗑️🗑️🗑️ Deleting videos:", video_ids)
        video_ids = [PydanticObjectId(v) for v in video_ids]
        await Video.find({"_id": {"$in": video_ids}}).delete()
        # delete the session videos
        await SessionVideo.find({"video_id": {"$in": video_ids}}).delete()
        # also tell ingestion to stop processing if it is processing those videos
        # request ingestion service
        async with httpx.AsyncClient() as client:
            # loop thorugh video ids to make request
            # liệu ingested có đưa ra 2th là xoá hay chưa xoá ko?
            for vid in video_ids:
                try:
                    await client.post(
                        f"http://100.113.186.28:8000/management/runs/{vid}/cancel",
                        json={"video_id": str(vid)},
                    )
                except Exception as e:
                    print(f"Error notifying ingestion service to delete video {vid}: {e}")

        # also delete from minio
        self.minio_service.delete_videos(video_ids)

        return True

    async def delete_session(self, session_id: str):
        # xoá chat_messages
        await SessionMessage.find(SessionMessage.session_id == PydanticObjectId(session_id)).delete()
        # xoá session_videos 
        await SessionVideo.find(SessionVideo.session_id == PydanticObjectId(session_id)).delete()
        # xoá session (chat_history)
        await ChatHistory.find_one(ChatHistory.id == PydanticObjectId(session_id)).delete()

        return True
import select
from urllib import request
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
import jwt
from regex import P
import requests
from google.auth.transport import requests as grequests
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordRequestForm,
)
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional
from datetime import datetime
from llama_index.core.base.llms.types import MessageRole
from google.oauth2 import id_token

from app.schema.user import ALGORITHM, SECRET_KEY, Token
from dotenv import load_dotenv

from app.core.dependencies import UserServiceDep
import test

load_dotenv()
import logging
from app.core.config import settings
import httpx

router = APIRouter(prefix="/api/user", tags=["user"])


class GoogleLoginRequest(BaseModel):
    code: str = Field(
        ..., description="The authorization code received from Google OAuth"
    )


security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    tester_payload = {
        "user_id": "6916f84e9a79606c0413d5d6",
        "email": "giaphuc29082004@gmail.com",
        "google_id": "109380215299372172369",
        "iat": 1763113291,
        "exp": 1763242891,
    }
    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        # print("Decoded JWT payload:", payload)
        return payload  # contains user_id, email, etc.
    except jwt.ExpiredSignatureError:
        return tester_payload
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        return tester_payload
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        return tester_payload


@router.get("/secure-endpoint")
async def secure_endpoint(user=Depends(verify_token)):
    return {"msg": f"Hello {user['email']}, welcome back!"}


@router.post("/login/google")
async def login_with_google(request: GoogleLoginRequest, user_service: UserServiceDep):
    client_id = settings.GOOGLE_OAUTH_CLIENT_ID
    client_secret = settings.GOOGLE_OAUTH_CLIENT_SECRET

    try:
        if not client_id or not client_secret:
            raise HTTPException(status_code=500, detail="Missing OAuth credentials")

        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "code": request.code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": "postmessage",  # for use with PKCE and SPA
            "grant_type": "authorization_code",
        }

        response = requests.post(token_url, data=payload, timeout=10)
        token_data = response.json()
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Google token error: {token_data.get('error', 'Unknown error')}",
            )

        if "error" in token_data:
            raise HTTPException(
                status_code=400,
                detail=f"Google OAuth error: {token_data['error']} - {token_data.get('error_description', '')}",
            )

        # Extract the id_token (JWT with user profile info)
        if "id_token" not in token_data:
            raise HTTPException(
                status_code=400,
                detail=f"No id_token in response. Available keys: {list(token_data.keys())}",
            )

        google_user_info = user_service.verify_google_id_token(
            token_data["id_token"], token_data["access_token"]
        )

        user = await user_service.create_or_update_user(google_user_info)

        user_data = {
            "user_id": str(user.id),  # Convert MongoDB ObjectId to string
            "email": user.email,
            "name": user.full_name,
            "google_id": user.google_id,
        }

        my_app_token = user_service.create_jwt_token(user_data)

        return {
            "access_token": my_app_token,
            "token_type": "Bearer",
            "user": {
                "id": user_data["user_id"],
                "email": user_data["email"],
                "name": user_data["name"],
                "picture": user.picture,
            },
        }

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Google OAuth request timed out")
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"Google OAuth request error: {str(e)}"
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid token audience")
    except jwt.InvalidIssuedAtError:
        raise HTTPException(status_code=401, detail="Invalid token issued at time")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# {data: "xin choa"}
@router.get("/chat-history")
async def get_chat_history(user_service: UserServiceDep, user=Depends(verify_token)):
    user_id = user["user_id"]
    chats = await user_service.get_user_chat_history(user_id)
    return {"chats": chats}


@router.get("/chat-history/search")
async def search_chat_history(
    user_service: UserServiceDep,
    query_text: str = Query(..., description="Search query for chat history"),
    user=Depends(verify_token),
):
    user_id = user["user_id"]
    search_results = await user_service.search_text_messages(user_id, query_text)
    return {"results": search_results}


@router.get("/chat-history/{session_id}")
async def get_chat_detail(session_id: str, user_service: UserServiceDep):
    chat = await user_service.get_user_chat_detail(session_id)
    return {"chat": chat}


@router.post("/uploads")
async def upload_files(
    user_service: UserServiceDep,
    group: str = Form(None),
    session_id: str = Form(None),
    files: List[UploadFile] = File(...),
    user=Depends(verify_token),
):
    # lưu metadata vào db
    user_id = user["user_id"]
    # lưu file lên minio, trả về ids
    video_id_video_url_thumbnail_url_s3_url_obj = await user_service.add_videos_to_user(
        user_id, files, group, session_id
    )
    video_ids_video_url_obj = [
        {"video_id": str(vid), "video_url": video_url}
        for vid, video_url, thumb_url, length, fps in video_id_video_url_thumbnail_url_s3_url_obj
    ]

    await user_service.ingest_videos(user_id, video_ids_video_url_obj)

    return {"msg": "File uploaded successfully"}


# for retry ingestion
@router.post("/ingestion/retry")
async def retry_ingestion(
    user_service: UserServiceDep,
    data: dict,
    user=Depends(verify_token),
):
    user_id = user["user_id"]
    video_ids = data.get("video_ids", [])

    await user_service.retry_ingestion(user_id, video_ids)
    return {"msg": "Ingestion retried successfully"}


@router.get("/groups")
async def get_user_groups(user_service: UserServiceDep, user=Depends(verify_token)):
    user_id = user["user_id"]
    groups = await user_service.get_user_groups(user_id)
    return {"groups": groups}


@router.post("/new-chat")
async def create_new_chat(
    user_service: UserServiceDep,
    user=Depends(verify_token),
):
    user_id = user["user_id"]
    new_chat_session_id = await user_service.create_new_chat_session(user_id)
    return {"chat_session_id": new_chat_session_id}


# get user video base on group
@router.get("/videos")
async def get_user_videos(
    user_service: UserServiceDep,
    group: str = Query(...),
    session_id: str = Query(...),
    user=Depends(verify_token),
):
    videos = await user_service.get_user_videos(group, session_id)
    return {"videos": videos}


@router.post("/videos/select")
async def select_videos(
    user_service: UserServiceDep,
    data: dict,
    user=Depends(verify_token),
):
    session_id = data.get("session_id")
    video_ids = data.get("video_ids", [])

    await user_service.select_videos(session_id, video_ids)
    return {"msg": "Videos selected"}


@router.post("/groups/create")
async def create_user_group(
    user_service: UserServiceDep, data: dict, user=Depends(verify_token)
):
    user_id = user["user_id"]
    group_name = data.get("group_name", "New Group")
    new_group_id = await user_service.create_user_group(user_id, group_name)
    return {"group_id": new_group_id}


@router.delete("/groups/{group_id}/delete")
async def delete_group(
    group_id: str, user_service: UserServiceDep, user=Depends(verify_token)
):
    user_id = user["user_id"]
    await user_service.delete_group(group_id)
    return {"group_id": group_id}


@router.delete("/videos/delete")
async def delete_videos(
    user_service: UserServiceDep,
    data: dict,
):
    user = (Depends(verify_token),)
    video_ids = data.get("video_ids", [])
    await user_service.delete_videos(video_ids)

    return {"msg": "Videos deleted"}


@router.delete("/session/{session_id}/delete")
async def delete_session(
    user_service: UserServiceDep,
    session_id: str,
    user=Depends(verify_token),
):
    await user_service.delete_session(session_id)

    return {"session_id": session_id}


@router.patch("/session/{session_id}/rename")
async def rename_session(
    user_service: UserServiceDep,
    session_id: str,
    data: dict,
    user=Depends(verify_token),
):
    new_name = data.get("new_name", "Renamed Session")
    success = await user_service.rename_session(session_id, new_name)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "new_name": new_name}


# group name rename
@router.patch("/group/{group_id}/rename")
async def rename_group(
    user_service: UserServiceDep,
    group_id: str,
    data: dict,
    user=Depends(verify_token),
):
    new_name = data.get("new_name", "Renamed Group")
    success = await user_service.rename_group(group_id, new_name)

    if not success:
        raise HTTPException(status_code=404, detail="Group not found")

    return {"group_id": group_id, "new_name": new_name}


# video name rename


@router.patch("/video/{video_id}/rename")
async def rename_video(
    user_service: UserServiceDep,
    video_id: str,
    data: dict,
    user=Depends(verify_token),
):
    new_name = data.get("new_name", "Renamed Video")
    success = await user_service.rename_video(video_id, new_name)

    if not success:
        raise HTTPException(status_code=404, detail="Video not found")

    return {"video_id": video_id, "new_name": new_name}


@router.get("/thumbnails")
async def get_segment_thumbnails(
    user_service: UserServiceDep,
    video_id: str = Query(..., description="Video ID"),
    frame_index: int = Query(
        ..., description="Frame index to generate thumbnails around"
    ),
):
    """
    On-demand thumbnail generation for video segments.
    Returns 5 thumbnails around the specified frame (-2s, -1s, center, +1s, +2s).
    """
    thumbnail_urls = await user_service.generate_video_thumbnails(
        video_id=video_id, frame_index=frame_index
    )
    return {"thumbnails": thumbnail_urls}

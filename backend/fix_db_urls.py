import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

async def main():
    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    
    old_prefix = "http://100.113.186.28:9000/"
    new_prefix = getattr(settings, 'MEDIA_URL_BASE', "https://api.departmentofcodingknight.site/media/")
    
    # Update video URLs
    video_collection = db[settings.VIDEO_COLLECTION_NAME]
    count = 0
    async for video in video_collection.find({"url": {"$regex": f"^{old_prefix}"}}):
        new_url = video.get("url", "").replace(old_prefix, new_prefix)
        new_thumb = video.get("thumbnail", "").replace(old_prefix, new_prefix)
        
        await video_collection.update_one(
            {"_id": video["_id"]},
            {"$set": {"url": new_url, "thumbnail": new_thumb}}
        )
        count += 1
    print(f"Updated {count} video documents.")
    
    # Update chat_messages collection blocks
    message_collection = db[settings.CHAT_MESSAGE_COLLECTION_NAME]
    msg_count = 0
    async for msg in message_collection.find({"blocks": {"$elemMatch": {"$or": [{"url": {"$regex": f"^{old_prefix}"}}, {"thumbnail": {"$regex": f"^{old_prefix}"}}]}}}):
        blocks = msg.get("blocks", [])
        updated = False
        for block in blocks:
            if "url" in block and isinstance(block["url"], str) and block["url"].startswith(old_prefix):
                block["url"] = block["url"].replace(old_prefix, new_prefix)
                updated = True
            if "url" in block and isinstance(block["url"], list):
                new_urls = []
                for u in block["url"]:
                    if isinstance(u, str) and u.startswith(old_prefix):
                        new_urls.append(u.replace(old_prefix, new_prefix))
                        updated = True
                    else:
                        new_urls.append(u)
                block["url"] = new_urls
            if "thumbnail" in block and isinstance(block["thumbnail"], str) and block["thumbnail"].startswith(old_prefix):
                block["thumbnail"] = block["thumbnail"].replace(old_prefix, new_prefix)
                updated = True
            # Also check preview_images in VideoSegment inside VideoBlock
            if "segments" in block and isinstance(block["segments"], list):
                for seg in block["segments"]:
                    if "preview_images" in seg and isinstance(seg["preview_images"], list):
                        new_previews = []
                        for u in seg["preview_images"]:
                            if isinstance(u, str) and u.startswith(old_prefix):
                                new_previews.append(u.replace(old_prefix, new_prefix))
                                updated = True
                            else:
                                new_previews.append(u)
                        seg["preview_images"] = new_previews
        if updated:
            await message_collection.update_one(
                {"_id": msg["_id"]},
                {"$set": {"blocks": blocks}}
            )
            msg_count += 1
            
    print(f"Updated {msg_count} chat messages containing old URLs.")

if __name__ == "__main__":
    asyncio.run(main())

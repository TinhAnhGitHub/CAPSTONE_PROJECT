import { useEffect, useState } from "react";
import VideoPlayer from "../VideoPlayer";
import socket from "@/api/socket";
export default function VideoBlock({ block }) {
    const [cacheBuster, setCacheBuster] = useState(0);

    useEffect(() => {
        function handleThumbnailsGenerated(data) {
            if (data.video_id === block.video_id) {
                // Bump cache buster so img URLs get a fresh query param
                setCacheBuster(n => n + 1);
            }
        }
        socket.on("thumbnails_generated", handleThumbnailsGenerated)
        return () => {
            socket.off("thumbnails_generated", handleThumbnailsGenerated)
        }

    }, [block.video_id])

    // Append cache buster to preview_images so browser re-fetches after upload
    const segments = cacheBuster > 0
        ? block.segments?.map(seg => ({
            ...seg,
            preview_images: seg.preview_images?.map(url => `${url}?v=${cacheBuster}`) ?? []
        }))
        : block.segments;

    return (
        <div className="grid grid-cols-3 gap-2 py-2 px-4">
            <VideoPlayer
                video={{
                    video_id: block.video_id,
                    url: block.url,
                    title: 'Video',
                    segments: segments,
                    fps: block.fps,
                    thumbnail: block.thumbnail
                }}
            />
        </div>
    );
}

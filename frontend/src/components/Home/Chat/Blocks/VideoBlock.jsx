import VideoPlayer from "../VideoPlayer";

export default function VideoBlock({ block }) {
    return (
        <div className="grid grid-cols-3 gap-2 py-2 px-4">
            <VideoPlayer
                video={{
                    video_id: block.video_id,
                    url: block.url,
                    title: 'Video',
                    segments: block.segments,
                    fps: block.fps
                }}
            />
        </div>
    );
}

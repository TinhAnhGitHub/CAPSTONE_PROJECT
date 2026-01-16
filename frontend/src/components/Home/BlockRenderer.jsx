import clsx from "clsx";
import { useMemo, memo } from "react";
import ImageViewer from "./Chat/ImageViewer";
import VideoPlayer from "./Chat/VideoPlayer";
import Markdown from "react-markdown";
import { ClipboardDocumentIcon } from "@heroicons/react/24/solid";
import toast from "react-hot-toast";
function handleCopyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text);
    } else {
        // Fallback for non-secure contexts
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    }
    toast.success("Copied to clipboard!");
}

export default memo(function BlockRenderer({ block, role }) {
    // Hooks must be called unconditionally at the top level
    // Stringify to get stable dependency for useMemo (array reference changes on each render)
    const imageUrlsKey = block.block_type === 'image' ? JSON.stringify(block.url) : '[]';

    const allImages = useMemo(() => {
        if (block.block_type !== 'image') return [];
        return block.url.map((url, i) => ({
            url: url,
            title: `Image ${i + 1}`
        }));
    }, [imageUrlsKey, block.block_type, block.url]);

    if (block.block_type === 'text') {
        //  check if user or assistant
        // .role === 'user' or 'assistant'
        return <div
            className={clsx(
                'max-w-[75%] px-4 py-2 my-2 rounded-lg text-sm whitespace-pre-wrap break-words hyphens-auto',
                "backdrop-blur-md shadow-lg border border-white/10",
                role === 'user'
                    ? 'bg-indigo-500/70 border-indigo-300/40 text-white self-end' // user: right
                    : 'bg-slate-800/60 border-white/10 text-slate-50 self-start' // bot: left
            )}
        >
            <Markdown key={block.text}>{block.text || ''}</Markdown>
            {role === 'assistant' &&
                <div onClick={() => handleCopyText(block?.text)}>
                    <ClipboardDocumentIcon className="w-5 h-5 mt-2 text-gray-400 hover:text-gray-200 cursor-pointer" />
                </div>}
        </div>
    }

    if (block.block_type === 'image') {
        return (
            <div className="max-w-[75%] self-start py-2">
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
                    {allImages.map((img, i) => (
                        <ImageViewer
                            key={`${i}-${img.url}`}
                            image={img}
                            images={allImages}
                            startIndex={i}
                        />
                    ))}
                </div>
            </div>
        );
    }

    if (block.block_type === 'video') {
        //     class VideoBlock(BaseModel):
        //     """A representation of video data to directly pass to/from the LLM."""

        //     block_type: Literal["video"] = "video"
        //     video_id: str | None = None
        //     url: AnyUrl | str | None = None
        //     video_mimetype: str | None = None
        //     fps: int | None = None
        //     segments: list[VideoSegment] | None = None

        // later will use videojs to add video segments to video, right now don't need yet
        // block_type: 'video',
        // url: '',
        // segments: [],
        // fps: 30,
        return (
            <div className="grid grid-cols-3 gap-2 py-2">
                <VideoPlayer video={{ url: block.url, title: `Video`, segments: block.segments, fps: block.fps }} />
            </div>
        );
    }



    return null;
}, (prevProps, nextProps) => {
    // Custom comparison - only re-render if block content actually changed
    if (prevProps.role !== nextProps.role) return false;
    if (prevProps.block.block_type !== nextProps.block.block_type) return false;

    const prev = prevProps.block;
    const next = nextProps.block;

    if (prev.block_type === 'text') {
        return prev.text === next.text;
    }
    if (prev.block_type === 'image') {
        return JSON.stringify(prev.url) === JSON.stringify(next.url);
    }
    if (prev.block_type === 'video') {
        return prev.url === next.url &&
            JSON.stringify(prev.segments) === JSON.stringify(next.segments) &&
            prev.fps === next.fps;
    }
    return false;
})

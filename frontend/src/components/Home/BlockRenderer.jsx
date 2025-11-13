import clsx from "clsx";
import ImageViewer from "./Chat/ImageViewer";
import VideoPlayer from "./Chat/VideoPlayer";

export default function BlockRenderer({ block, role }) {
    if (block.block_type === 'text') {
        //  check if user or assistant
        // .role === 'user' or 'assistant'
        return <div
            className={clsx(
                'max-w-[75%] px-4 py-2 my-2 rounded-lg text-sm whitespace-pre-wrap break-all',
                role === 'user'
                    ? 'bg-gray-700 text-white self-end' // user: right
                    : 'bg-gray-200 text-black self-start' // bot: left
            )}
        >
            {block?.text}
        </div>
    }

    if (block.block_type === 'image') {
        return (
            <div className="grid grid-cols-3 gap-2">
                {block.url.map((url, i) => (
                    <ImageViewer key={i} image={{ url: url, title: `Image ${i + 1}` }} />
                ))}
            </div>
        );
    }

    if (block.block_type === 'video') {
        return (
            <div className="grid grid-cols-3 gap-2">
                {block.url.map((url, i) => (
                    <VideoPlayer key={i} video={{ url: url, title: `Video ${i + 1}` }} />
                ))}
            </div>
        );
    }

    return null;
}

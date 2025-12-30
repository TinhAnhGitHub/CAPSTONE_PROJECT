import clsx from "clsx";
import ImageViewer from "./Chat/ImageViewer";
import VideoPlayer from "./Chat/VideoPlayer";
import Markdown from "react-markdown";

function handleCopyText(text){
    navigator.clipboard.writeText(text);
}

export default function BlockRenderer({ block, role }) {
    if (block.block_type === 'text') {
        //  check if user or assistant
        // .role === 'user' or 'assistant'
        return <div
            className={clsx(
                'max-w-[75%] px-4 py-2 my-2 rounded-lg text-sm whitespace-pre-wrap break-all',
                "backdrop-blur-md shadow-lg border border-white/10",
                role === 'user'
                    ? 'bg-indigo-500/70 border-indigo-300/40 text-white self-end' // user: right
                    : 'bg-slate-800/60 border-white/10 text-slate-50 self-start' // bot: left
            )}
        >
            <Markdown>{block?.text}</Markdown>
            {role === 'assistant' && <div onClick={() => handleCopyText(block?.text)}> copy icon </div>}
        </div>
    }

    if (block.block_type === 'image') {
        return (
            <div className="grid grid-cols-3 gap-2 py-2">
                {block.url.map((url, i) => (
                    <ImageViewer key={i} image={{ url: url, title: `Image ${i + 1}` }} />
                ))}
            </div>
        );
    }

    if (block.block_type === 'video') {
        return (
            <div className="grid grid-cols-3 gap-2 py-2">
                {block.url.map((link, i) => (
                    <VideoPlayer key={i} video={{ url: link, title: `Video ${i + 1}` }} />
                ))}
            </div>
        );
    }

    return null;
}

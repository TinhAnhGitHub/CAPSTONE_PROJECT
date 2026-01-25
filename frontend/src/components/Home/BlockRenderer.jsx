import { useMemo, memo } from "react";
import { TextBlock, ImageGallery, VideoBlock } from "./Chat/Blocks";
import Thinking from "./Chat/Thinking";
import Tools from "./Chat/Tools";

const PREVIEW_COUNT = 6;

// now add thinking block and tools block

export default memo(function BlockRenderer({ block, role }) {
    const imageUrlsKey = block.block_type === 'image' ? JSON.stringify(block.url) : '[]';

    const allImages = useMemo(() => {
        if (block.block_type !== 'image') return [];
        return block.url.map((url, i) => ({
            url: url,
            title: `Image ${i + 1}`
        }));
    }, [imageUrlsKey, block.block_type, block.url]);

    if (block.block_type === 'text') {
        return <TextBlock block={block} role={role} />;
    }

    if (block.block_type === 'image') {
        return (
            <ImageGallery
                allImages={allImages}
                imageCount={allImages.length}
                previewCount={PREVIEW_COUNT}
            />
        );
    }

    if (block.block_type === 'video') {
        return <VideoBlock block={block} />;
    }

    if (block.block_type === 'thinking') {
        return <Thinking block={block.steps} />;
    }
    if (block.block_type === 'tools') {
        return <Tools block={block.steps} />;
    }

    return null;
}, (prevProps, nextProps) => {
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
    if (prev.block_type === 'thinking') {
        return JSON.stringify(prev.steps) === JSON.stringify(next.steps);
    }
    if (prev.block_type === 'tools') {
        return JSON.stringify(prev.steps) === JSON.stringify(next.steps);
    }
    return false;
});

export default function parseChunkToBlock(msg_type, chunk) {
    if (msg_type === 'text') {
        return { block_type: 'text', text_content: chunk || " " };
    }
    if (msg_type === 'image') {
        return { block_type: 'image', image_urls: chunk };
    }
    if (msg_type === 'video') {
        return { block_type: 'video', video_urls: chunk };
    }
    return null;
}
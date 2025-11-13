export default function parseChunkToBlock(msg_type, chunk) {
    if (msg_type === 'text') {
        return { block_type: 'text', text: chunk || " " };
    }
    if (msg_type === 'image') {
        return { block_type: 'image', url: chunk };
    }
    if (msg_type === 'video') {
        return { block_type: 'video', url: chunk };
    }
    return null;
}
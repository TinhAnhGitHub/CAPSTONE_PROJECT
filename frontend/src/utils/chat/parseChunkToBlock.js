export default function parseChunkToBlock(msg_type, chunk) {
    if (msg_type === 'text') {
        return { block_type: 'text', text: chunk || " " };
    }
    if (msg_type === 'image') {
        // results // flatten this
        const media_url = chunk.map(item => item.url).flat();
        // many url
        return [{ block_type: 'image', url: media_url }];
    }
    if (msg_type === 'video') {
        return chunk;
    }
    if (msg_type === 'thinking') {
        const thinking_block = {
            block_type: 'thinking',
            steps: [{...chunk}], // many thinking steps
        }
        return thinking_block;
    }
    if (msg_type === 'tool_call') {
        const tool_block = {
            block_type: 'tool_call',
            steps: [{
                status: 'pending',
                ...chunk,
            }]
        }
        return tool_block;
    }
    return null;
}
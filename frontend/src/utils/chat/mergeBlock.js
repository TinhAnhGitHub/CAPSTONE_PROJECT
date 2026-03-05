export default function mergeBlock(lastBlock, newBlock) {
    if (!lastBlock || !newBlock) return null;

    if (lastBlock.block_type === 'text' && newBlock.block_type === 'text') {
        // Return a NEW object instead of mutating
        return { ...lastBlock, text: lastBlock.text + newBlock.text };
    }
    if (lastBlock.block_type === 'image' && newBlock.block_type === 'image') {
        return { ...lastBlock, url: lastBlock.url.concat(newBlock.url) };
    }
    if (lastBlock.block_type === 'video' && newBlock.block_type === 'video') {
        // Only merge if same video_id (same video, different segments)
        if (lastBlock.video_id && newBlock.video_id && lastBlock.video_id !== newBlock.video_id) {
            return null; // Different videos — don't merge, append as separate block
        }
        return { ...lastBlock, segments: [...(lastBlock.segments || []), ...(newBlock.segments || [])] };
    }
    if (lastBlock.block_type === 'tool_call' && newBlock.block_type === 'tool_call') {
        return {
            ...lastBlock,
            steps: [...lastBlock.steps, ...newBlock.steps],
        };
    }
    if (lastBlock.block_type === 'thinking' && newBlock.block_type === 'thinking') {
        return {
            ...lastBlock,
            steps: [...lastBlock.steps, ...newBlock.steps],
        };
    }
    return null;
}
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
        return { ...lastBlock, url: lastBlock.url.concat(newBlock.url) };
    }
    return null;
}
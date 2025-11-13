export default function mergeBlock(lastBlock, newBlock) {
    if (!lastBlock || !newBlock) return false;
    
    if (lastBlock.block_type === 'text' && newBlock.block_type === 'text') {
        lastBlock.text += newBlock.text;
        return true;
    }
    if (lastBlock.block_type === 'image' && newBlock.block_type === 'image') {
        lastBlock.url = lastBlock.image_urls.concat(newBlock.url);
        return true;
    }
    if (lastBlock.block_type === 'video' && newBlock.block_type === 'video') {
        lastBlock.url = lastBlock.video_urls.concat(newBlock.url);
        return true;
    }
    return false;
}
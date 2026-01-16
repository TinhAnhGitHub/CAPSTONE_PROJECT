export default function parseChunkToBlock(msg_type, chunk) {
    if (msg_type === 'text') {
        return { block_type: 'text', text: chunk || " " };
    }
    if (msg_type === 'image') {
        // results // flatten this
        const media_url = chunk.map(item => item.url).flat();
        return [{ block_type: 'image', url: media_url }];
    }
    if (msg_type === 'video') {
    //     class VideoBlock(BaseModel):
    //     """A representation of video data to directly pass to/from the LLM."""

    //     block_type: Literal["video"] = "video"
    //     video_id: str | None = None
    //     url: AnyUrl | str | None = None
    //     video_mimetype: str | None = None
    //     fps: int | None = None
    //     segments: list[VideoSegment] | None = None

        // chunk is [VideoBlock]
        
        return chunk;
    }
    return null;
}
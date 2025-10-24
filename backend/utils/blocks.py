from app.model.session_message import ImageBlock, TextBlock, VideoBlock


def parseFullResponseToBlocks(responses: list):
    """
    Convert streamed (msg_type, chunk) pairs into merged, high-level message blocks.
    Consecutive text chunks concatenate; image/video lists extend.
    """
    blocks = []
    for item in responses:
        t = item["msg_type"]
        c = item["chunk"]

        if not c:
            continue  # skip empty chunks

        # if same type as previous → merge
        if blocks:
            last = blocks[-1]
            if t == "text" and isinstance(last, TextBlock):
                last.text_content += c
                continue
            if t == "image" and isinstance(last, ImageBlock):
                last.image_urls.extend(c if isinstance(c, list) else [c])
                continue
            if t == "video" and isinstance(last, VideoBlock):
                last.video_urls.extend(c if isinstance(c, list) else [c])
                continue

        # otherwise start new block
        if t == "text":
            blocks.append(TextBlock(text_content=c))
        elif t == "image":
            blocks.append(ImageBlock(image_urls=c if isinstance(c, list) else [c]))
        elif t == "video":
            blocks.append(VideoBlock(video_urls=c if isinstance(c, list) else [c]))

    return blocks

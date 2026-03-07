from PIL import Image
import io
from pathlib import Path
import tempfile

def to_jpeg(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((640, 640))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


def create_image_tmp_file(image_bytes: bytes) -> Path:
    image_bytes_jpeg = to_jpeg(image_bytes)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
    tmp.write(image_bytes_jpeg)
    tmp.close()
    return Path(tmp.name)
import base64
from urllib.parse import urlparse
from pathlib import Path

def parse_s3_url(s3_url: str) -> tuple[str, str]:
    parsed = urlparse(s3_url)
    return parsed.netloc, parsed.path.lstrip("/")

def encode_image_base64(
    image_path: str
):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(path, "rb") as f:
        image_bytes = f.read()

    encoded_str = base64.b64encode(image_bytes).decode("utf-8")
    return encoded_str
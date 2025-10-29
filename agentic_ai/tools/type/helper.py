


from urllib.parse import urlparse
def extract_s3_minio_url(s3_link:str) -> tuple[str,str]:
    parsed = urlparse(s3_link)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key

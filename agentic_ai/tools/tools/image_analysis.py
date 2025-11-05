import io
import base64
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from minio import Minio
from pydantic import BaseModel
from google.ai.generativelanguage_v1beta.types import Tool

class ImageInput(BaseModel):
    image_base64: str
    image_type: str

def _IWI_impl(image: ImageInput):
    """
    Embed input image and retrieve top 10 most similar images from MinIO
    """
    img_data = base64.b64decode(image.image_base64)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    model = SentenceTransformer("clip-ViT-B-32")
    query_emb = model.encode(np.array(img), convert_to_tensor=True)
    client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
    results = []
    for obj in client.list_objects("images", recursive=True):
        data = client.get_object("images", obj.object_name).read()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        emb = model.encode(np.array(im), convert_to_tensor=True)
        sim = util.cos_sim(query_emb, emb).item()
        results.append((obj.object_name, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:10]]

IwI = Tool(
    function=_IWI_impl,
    name="IwI",
    description="Embed an image and retrieve top 10 most similar images from MinIO.",
)

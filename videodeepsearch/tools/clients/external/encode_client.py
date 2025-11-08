
from .base import ImageEmbeddingClient, TextEmbeddingClient, ImageEmbeddingSettings, TextEmbeddingSettings
from ingestion.prefect_agent.service_image_embedding.schema import ImageEmbeddingRequest, ImageEmbeddingResponse
from ingestion.prefect_agent.service_text_embedding.schema import TextEmbeddingRequest, TextEmbeddingResponse

class ExternalEncodeClient:
    def __init__(
            self, 
            img_text_client: ImageEmbeddingClient,
            img_text_settings: ImageEmbeddingSettings, 
            txt_settings: TextEmbeddingSettings,
            txt_client: TextEmbeddingClient,    
        ):
        self.img_text_client = img_text_client
        self.img_text_settings = img_text_settings

        self.txt_client = txt_client
        self.txt_settings = txt_settings

    async def connect(self):
        await self.img_text_client.connect()
        await self.txt_client.connect()

        await self.img_text_client.load_model(model_name=self.img_text_settings.model_name, device=self.img_text_settings.device)
        await self.txt_client.load_model(model_name=self.txt_settings.model_name, device=self.txt_settings.device)

    async def disconnect(self):
        await self.img_text_client.unload_model()
        await self.txt_client.unload_model()   

        await self.img_text_client.close()
        await self.txt_client.close()

        
    async def encode_visual_text(
        self,
        request: ImageEmbeddingRequest,
    ) -> ImageEmbeddingResponse:
        
        await self.img_text_client.load_model(
            model_name=self.img_text_settings.model_name,
            device=self.img_text_settings.device
        )
        response = await self.img_text_client.make_request(
            method='POST',
            endpoint=self.img_text_client.inference_endpoint,
            request_data=request
        )  
        parsed = ImageEmbeddingResponse.model_validate(response)

        return parsed
    
    async def encode_text(
        self,
        request: TextEmbeddingRequest,
    )->TextEmbeddingResponse:
        await self.txt_client.load_model(
            model_name=self.txt_settings.model_name,
            device=self.txt_settings.device
        )
        response = await self.txt_client.make_request(
            method='POST',
            endpoint=self.txt_client.inference_endpoint,
            request_data=request
        )  
        parsed = TextEmbeddingResponse.model_validate(response)

        return parsed

    
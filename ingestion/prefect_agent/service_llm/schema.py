from typing import Any

from pydantic import BaseModel, Field



class LLMSingleRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text provided to the model")
    image_base64: list[str] | None = Field(None, description="Optional list of image paths")

class LLMRequest(BaseModel):
    llm_requests: list[LLMSingleRequest] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMSingleResponse(BaseModel):
    answer: str = Field("", description="Model-generated answer for a single request")
    input_tokens: int | None = Field(None, description="Input token count for this request")
    output_tokens: int | None = Field(None, description="Output token count for this request")
    status: str = Field("success", description="Status for this specific inference")
    error: str | None = Field(None, description="Error message if the inference failed")


class LLMResponse(BaseModel):
    responses: list[LLMSingleResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_name: str = Field(..., description="Model identifier")
    status: str = Field("success", description="Overall status for the batch")
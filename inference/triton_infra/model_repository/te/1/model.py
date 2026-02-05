import triton_python_backend_utils as pb_utils #type:ignore
import numpy as np
import torch
import json
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

class TritonPythonModel:
    def initialize(self, args):
        """Load model and tokenizer."""
        self.model_config = json.loads(args["model_config"])
        
        params = self.model_config.get("parameters", {})
        model_name = params["mmbert_model_name"]["string_value"]
        self.max_length = int(params["mmbert_max_length"]["string_value"])
       

        
        self.device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        
  
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        torch.set_float32_matmul_precision('high')
        
        self.embedding_dim = self.model.config.hidden_size

    def execute(self, requests):
        """Process batch of requests."""
        responses = []
        
        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "texts")
                texts = [t.decode("utf-8") for t in input_tensor.as_numpy()]
                
                if not texts:
                    raise ValueError("Empty text input")

                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = embeddings.cpu().numpy().astype(np.float32)

                # Create output tensor
                # Shape: [batch_size, embedding_dim]
                output_tensor = pb_utils.Tensor(
                    "embeddings",
                    embeddings
                )
                
                responses.append(pb_utils.InferenceResponse([output_tensor]))
                
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=str(e)
                ))

        return responses

    def finalize(self):
        """Cleanup."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


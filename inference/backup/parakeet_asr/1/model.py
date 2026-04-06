import os
# Force CPU usage before importing torch/nemo
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import io
import json
import numpy as np
import torch
import soundfile as sf
import triton_python_backend_utils as pb_utils
import nemo.collections.asr as nemo_asr

class TritonPythonModel:
    def initialize(self, args):
        """
        Load NeMo ASR model once when Triton starts.
        """
        self.device = torch.device("cpu")
        
        # Load the RNNT/TDT model
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            map_location='cpu'
        )
        
        self.model.eval()
        self.model.to(self.device)

    def execute(self, requests):
        responses = []

        for request in requests:
            audio_input = pb_utils.get_input_tensor_by_name(
                request, "AUDIO"
            ).as_numpy()

           
    

            results = []            
            for i in range(audio_input.shape[0]):
                try:
                    
                    raw_audio_bytes = audio_input[i][0]
                    byte_io = io.BytesIO(raw_audio_bytes)
                    audio, sr = sf.read(byte_io)
                    audio_tensor = torch.tensor(audio, dtype=torch.float32).to(self.device)
                
                    with torch.no_grad():
                        encoder_outputs, encoded_lengths = self.model.forward(
                            input_signal=audio_tensor.unsqueeze(0),
                            input_signal_length=torch.tensor([audio_tensor.shape[0]])
                        )

                        hypotheses = self.model.decoding.rnnt_decoder_predictions_tensor(
                            encoder_outputs,
                            encoded_lengths
                        )

                    results.append({
                        "text": hypotheses[0].text
                    })
                except Exception as e:
                    results.append({"text": "", "error": str(e)})

            
            output_tensor = pb_utils.Tensor(
                "OUTPUT",
                np.array(
                    [json.dumps(results).encode("utf-8")],
                    dtype=object
                )
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
            )

        return responses
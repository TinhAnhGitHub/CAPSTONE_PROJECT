import numpy as np
import triton_python_backend_utils as pb_utils #type:ignore
from fastembed import SparseTextEmbedding
import pickle

MODEL_NAME = "prithivida/Splade_PP_en_v1"

class TritonPythonModel:

    def initialize(self, args):
        self.model = SparseTextEmbedding(model_name=MODEL_NAME)

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            raw_texts = input_tensor.as_numpy().flatten()

            texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in raw_texts]

            embeddings = list(self.model.embed(texts))

            indices_batch = []
            values_batch = []

            for emb in embeddings:
                indices_batch.append(pickle.dumps(np.array(emb.indices, dtype=np.int32)))
                values_batch.append(pickle.dumps(np.array(emb.values, dtype=np.float32)))

            out_indices = pb_utils.Tensor(
                "INDICES", np.array(indices_batch, dtype=object)
            )

            out_values = pb_utils.Tensor(
                "VALUES", np.array(values_batch, dtype=object)
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[out_indices, out_values]
                )
            )

        return responses
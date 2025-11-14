from transformers import BertTokenizer, BertModel

import torch
import numpy as np
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def embedding(text : str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def sim_search(data, query: str, top_k=3):

    query = embedding(query)

    ques = [embedding(q["question"]) for q in data]

    sims = [cosine_similarity(q, query) for q in ques]

    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "question": data[idx]["question"],
            "score": float(sims[idx])
        })
    return results



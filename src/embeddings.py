import google.generativeai as genai
import numpy as np

class GeminiEmbedder:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = "models/embedding-001"

    def embed_text(self, text):
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="semantic_similarity"
        )
        return np.array(response["embedding"], dtype=np.float32)

def get_embedding_fn(api_key):
    return GeminiEmbedder(api_key)

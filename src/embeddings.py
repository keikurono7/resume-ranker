# src/embeddings.py

import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
import os

MODEL_DIR = "models/mpnet"

class MPNetEmbedder:
    def __init__(self):
        self.tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        self.session = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"))
        self.input_names = [i.name for i in self.session.get_inputs()]

    def embed_text(self, text):
        enc = self.tokenizer.encode(text)

        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        outputs = self.session.run(None, feed)

        # Model output: token embeddings (1, seq, hidden)
        token_embeddings = np.array(outputs[0])

        # attention mask: (1, seq) -> expanded (1, seq, 1)
        mask = attention_mask[:, :, None]

        # Apply mask-aware mean pooling
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)

        sentence_embedding = summed / counts  # (1, hidden)

        # Remove batch dimension → (hidden,)
        sentence_embedding = sentence_embedding.squeeze()

        # Normalize
        sentence_embedding = sentence_embedding / np.linalg.norm(sentence_embedding)

        return sentence_embedding.tolist()



def get_embedding_fn():
    return MPNetEmbedder()

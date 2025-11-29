import numpy as np

class VectorStore:
    def __init__(self, dim=768):
        self.dim = dim
        self.embeddings = []
        self.texts = []
        self.ids = []

    def add(self, resume_id, text, embedding):
        embedding = np.array(embedding, dtype=np.float32)
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.ids.append(resume_id)

    def search(self, query_emb, top_k=5):
        if len(self.embeddings) == 0:
            return []

        query_emb = np.array(query_emb, dtype=np.float32)
        X = np.stack(self.embeddings)

        dot = np.dot(X, query_emb)
        X_norm = np.linalg.norm(X, axis=1)
        q_norm = np.linalg.norm(query_emb) + 1e-9
        sims = dot / (X_norm * q_norm)

        idxs = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in idxs:
            results.append({
                "resume_id": self.ids[idx],
                "text": self.texts[idx],
                "similarity": float(sims[idx])
            })

        return results

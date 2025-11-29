import numpy as np
from sklearn.neighbors import NearestNeighbors

class VectorStore:
    def __init__(self, dim=768):
        self.dim = dim
        self.embeddings = []
        self.texts = []
        self.ids = []
        self.nn = None

    def add(self, resume_id, text, embedding):
        embedding = np.array(embedding, dtype=np.float32)
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.ids.append(resume_id)

    def build(self):
        if len(self.embeddings) == 0:
            return
        X = np.stack(self.embeddings)
        k = min(5, len(X))
        self.nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self.nn.fit(X)

    def search(self, query_emb, top_k=5):
        if self.nn is None:
            raise ValueError("Index not built. Call build() after adding resumes.")

        total = len(self.embeddings)
        k = min(top_k, total)

        query_emb = np.array(query_emb).reshape(1, -1)
        distances, indices = self.nn.kneighbors(query_emb, n_neighbors=k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = float(1 - dist)
            results.append({
                "resume_id": self.ids[idx],
                "text": self.texts[idx],
                "similarity": similarity
            })

        return results

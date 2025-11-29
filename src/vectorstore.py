# src/vectorstore.py

import chromadb

class SimpleChromaStore:
    def __init__(self, path="/tmp/chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        try:
            self.collection = self.client.get_collection("resumes")
        except:
            self.collection = self.client.create_collection(
                name="resumes",
                metadata={"hnsw:space": "cosine"}
            )

    def reset(self):
        try:
            self.client.delete_collection("resumes")
        except:
            pass

        self.collection = self.client.create_collection(
            name="resumes",
            metadata={"hnsw:space": "cosine"}
        )

    def add_resume(self, rid, text, emb):
        self.collection.add(
            ids=[rid],
            documents=[text],
            embeddings=[emb]
        )

    def search(self, emb, top_k):
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=top_k
        )

        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "resume_id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "distance": float(res["distances"][0][i])
            })

        return out


def create_chroma_store(path="/tmp/chroma_db"):
    return SimpleChromaStore(path)

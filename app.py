import streamlit as st
import os
import sys
sys.path.append("src")

from src.parser import parse_resume
from src.embeddings import get_embedding_fn
from src.vectorstore import VectorStore
from src.ranker import score_resume

st.title("AI Resume Screening Agent")

api_key = st.text_input("Gemini API Key", type="password")

if not api_key:
    st.warning("Enter your API key to continue.")
    st.stop()

job_desc = st.text_area("Job Description")

uploaded = st.file_uploader("Upload resumes (PDF/DOCX/TXT)", accept_multiple_files=True)

if "store" not in st.session_state:
    st.session_state.store = VectorStore(dim=768)

if st.button("Ingest Resumes"):
    embedder = get_embedding_fn(api_key)

    for file in uploaded:
        os.makedirs("uploaded", exist_ok=True)
        path = f"uploaded/{file.name}"
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        text = parse_resume(path)
        emb = embedder.embed_text(text)

        st.session_state.store.add(file.name, text, emb)

    st.session_state.store.build()
    print("NEIGHBORS SET TO:", st.session_state.store.nn.n_neighbors)

    st.success("Resumes ingested successfully!")

if st.button("Screen Candidates"):
    if not job_desc.strip():
        st.error("Enter job description")
        st.stop()

    embedder = get_embedding_fn(api_key)
    jd_emb = embedder.embed_text(job_desc)

    results = st.session_state.store.search(jd_emb, top_k=5)

    st.subheader("Ranking Results")
    for r in results:
        st.write("PROCESSING:", r['resume_id'])
        st.write(f"### {r['resume_id']} — Similarity: {r['similarity']:.3f}")

        scores = score_resume(api_key, job_desc, r["text"])
        st.json(scores)
        st.write("---")


import streamlit as st
import os
import sys

sys.path.append("src")

from src.embeddings import get_embedding_fn
from src.vectorstore import create_chroma_store
from src.parser import parse_resume
from src.ranker import score_resume, hybrid_rank

st.title("AI Resume Screening Agent")

if "embedder" not in st.session_state:
    st.session_state.embedder = get_embedding_fn()

if "store" not in st.session_state:
    st.session_state.store = create_chroma_store()

if "resumes" not in st.session_state:
    st.session_state.resumes = {}

job_desc = st.text_area("Job Description")
files = st.file_uploader("Upload resumes", accept_multiple_files=True)
top_k = st.sidebar.slider("Top K", 1, 10, 5)

if st.button("Ingest Resumes"):
    st.session_state.store.reset()
    st.session_state.resumes = {}

    for f in files:
        path = f"/tmp/{f.name}"
        with open(path, "wb") as o:
            o.write(f.getbuffer())

        text = parse_resume(path)
        emb = st.session_state.embedder.embed_text(text)

        st.session_state.resumes[f.name] = text
        st.session_state.store.add_resume(f.name, text, emb)

    st.success("Ingested successfully!")

if st.button("Screen JD"):
    jd_emb = st.session_state.embedder.embed_text(job_desc)
    results = st.session_state.store.search(jd_emb, top_k)

    final = []
    for r in results:
        sim = 1 - r["distance"]
        rs_emb = st.session_state.embedder.embed_text(r["text"])

        scores = score_resume(job_desc, r["text"], jd_emb, rs_emb)
        rank = hybrid_rank(scores)

        final.append((rank, r, scores))

    final.sort(reverse=True, key=lambda x: x[0])

    for rank, r, scores in final:
        st.subheader(f"{r['resume_id']} — Rank: {rank:.3f}")
        st.json(scores)

# src/parser.py

import pdfplumber
from docx import Document

def parse_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def parse_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def parse_resume(path):
    if path.lower().endswith(".pdf"):
        return parse_pdf(path)

    if path.lower().endswith(".docx"):
        return parse_docx(path)

    return open(path, "r", encoding="utf8", errors="ignore").read()

import docx
from pypdf import PdfReader

def parse_resume(path):
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if path.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

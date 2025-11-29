# src/ranker.py

import numpy as np
import re
from datetime import datetime

def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

MONTHS = {
    "jan":1,"january":1, "feb":2,"february":2, "mar":3,"march":3,
    "apr":4,"april":4, "may":5,
    "jun":6,"june":6, "jul":7,"july":7,
    "aug":8,"august":8, "sep":9,"september":9,
    "oct":10,"october":10, "nov":11,"november":11, "dec":12,"december":12
}

def extract_experience(text):
    text = text.lower()

    # 1. "2 years", "3+ years", "1.5 years"
    year_matches = re.findall(r"(\d+\.?\d*)\s*\+?\s*years?", text)
    month_matches = re.findall(r"(\d+)\s*months?", text)

    total_years = 0.0
    for y in year_matches:
        total_years += float(y)

    for m in month_matches:
        total_years += float(m) / 12

    # 2. Year ranges "2018–2020"
    date_ranges = re.findall(r"(19|20)\d{2}\s*[-–to]+\s*(19|20)\d{2}", text)
    for start, end in date_ranges:
        total_years += (int(end) - int(start))

    # 3. Month-year ranges "Aug 2017 – Dec 2018"
    month_year_pattern = re.findall(
        r"(" + "|".join(MONTHS.keys()) + r")\s*(19|20)\d{2}\s*[-–to]+\s*(" + "|".join(MONTHS.keys()) + r")\s*(19|20)\d{2}",
        text
    )

    for m1, y1, m2, y2 in month_year_pattern:
        start = datetime(int(y1), MONTHS[m1], 1)
        end = datetime(int(y2), MONTHS[m2], 1)
        diff_years = (end - start).days / 365
        total_years += diff_years

    # If nothing is found, assume at least 1 year experience
    if total_years == 0:
        return 1

    return round(total_years, 2)


def score_resume(job_desc, resume, jd_emb, rs_emb):
    # TRUE semantic similarity
    semantic_sim = cosine(jd_emb, rs_emb)

    # Experience
    years = extract_experience(resume)
    exp_score = min(years * 10, 100)

    # Communication (length heuristic)
    comm_score = min(len(resume.split()) // 20, 100)

    return {
        "semantic_similarity": semantic_sim,
        "experience": exp_score,
        "communication": comm_score,
        "summary": f"Semantic similarity={semantic_sim:.3f}, Exp={years} yrs"
    }


def hybrid_rank(scores):
    return (
        0.6 * scores["semantic_similarity"] +
        0.25 * (scores["experience"] / 100) +
        0.15 * (scores["communication"] / 100)
    )

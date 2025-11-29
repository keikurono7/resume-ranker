import google.generativeai as genai
import re
import json

def score_resume(api_key, job_desc, resume_text):
    genai.configure(api_key=api_key)

    prompt = f"""
You are an HR expert. Score this resume for the job description below.
Return output in JSON with keys: technical, experience, communication, summary.

Job Description:
{job_desc}

Resume:
{resume_text}
"""

    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(prompt)

    text = response.text

    # 1. Extract json inside ```json ... ``` if present
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        # fallback: detect first {...}
        json_match = re.search(r"{[\s\S]*}", text)
        cleaned = json_match.group(0) if json_match else None

    # 2. If still nothing, return fallback
    if not cleaned:
        return {
            "technical": 0,
            "experience": 0,
            "communication": 0,
            "summary": text
        }

    # 3. Remove nested JSON inside summary
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)

    # 4. Parse safely
    try:
        parsed = json.loads(cleaned)
    except:
        # try fixing smart quotes or extra commas
        cleaned = cleaned.replace("“","\"").replace("”","\"").replace("\n"," ")
        try:
            parsed = json.loads(cleaned)
        except:
            parsed = {
                "technical": 0,
                "experience": 0,
                "communication": 0,
                "summary": text
            }

    # Ensure correct keys even if missing
    return {
        "technical": parsed.get("technical", 0),
        "experience": parsed.get("experience", 0),
        "communication": parsed.get("communication", 0),
        "summary": parsed.get("summary", "")
    }


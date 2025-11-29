import google.generativeai as genai

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

    try:
        import json
        return json.loads(response.text)
    except:
        return {
            "technical": 0,
            "experience": 0,
            "communication": 0,
            "summary": response.text
        }

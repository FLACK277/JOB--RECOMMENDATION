import os
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests

load_dotenv()

JOB_API_URL = os.getenv("JOB_API_URL", "https://example-jobs-api.com/search")
JOB_API_KEY = os.getenv("JOB_API_KEY", "")
PORT = int(os.getenv("PORT", "5000"))

app = Flask(
    __name__,
    static_folder=".",
    static_url_path="",
    template_folder="."
)
CORS(app)


# ====== ML MODEL LOADING & SCORING (REPLACE WITH YOUR REAL MODEL) ======

def load_ml_model():
    """
    Load your trained model and any vectorizers/encoders here.
    Replace this dummy implementation with your real one.
    Example:

        import pickle
        with open(\"models/model.pkl\", \"rb\") as f:
            model = pickle.load(f)
        return model
    """
    return None  # placeholder


MODEL = load_ml_model()


def compute_match_score(profile: Dict[str, Any], job: Dict[str, Any]) -> float:
    """
    Compute a match score between the user profile and a job posting.
    Replace this with your real ML logic.

    profile: {
      \"profile_skills\": \"...\",
      \"experience_years\": 2,
      \"desired_role\": \"...\",
      ...
    }

    job is a normalized job dict from the external API.
    Returns a float between 0 and 1.
    """
    # Placeholder: simple keyword overlap between profile skills and job text.
    profile_skills = (profile.get("profile_skills") or "").lower()
    job_title = (job.get("title") or "").lower()
    job_description = (job.get("description") or "").lower()

    if not profile_skills:
        return 0.0

    score = 0.0
    for token in [s.strip() for s in profile_skills.replace(",", " ").split() if s.strip()]:
        if token in job_title:
            score += 0.15
        if token in job_description:
            score += 0.08

    # crude clamp
    score = min(score, 1.0)

    # TODO: use your real ML model here instead of this heuristic
    # Example:
    # features = build_feature_vector(profile, job)
    # score = MODEL.predict_proba([features])[0, 1]

    return score


# ====== EXTERNAL JOBS API CALL ======

def call_external_jobs_api(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Call the external jobs API using the profile & preferences.
    You MUST adapt this to your chosen job API.

    Returns a list of raw job dicts from the external API.
    """
    headers = {}
    if JOB_API_KEY:
        headers["Authorization"] = f"Bearer {JOB_API_KEY}"

    params = {
        "search": query.get("desired_role") or query.get("profile_skills"),
        "location": query.get("location"),
        "remote": "true" if query.get("remote_only") else None,
        "employment_type": query.get("employment_type"),
        "min_salary": query.get("salary_min"),
        "limit": 50,
    }
    # Remove None/empty values
    params = {k: v for k, v in params.items() if v not in (None, "")}

    try:
        resp = requests.get(JOB_API_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print("Error calling external jobs API:", exc)
        return []

    # Adjust this based on your API response schema
    if isinstance(data, dict) and "jobs" in data:
        jobs = data["jobs"]
    elif isinstance(data, dict) and "results" in data:
        jobs = data["results"]
    elif isinstance(data, list):
        jobs = data
    else:
        jobs = []

    return jobs


def normalize_job(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw job posting from the external API into a consistent structure
    consumed by the frontend.
    Fill in as many fields as you can from your API's schema.
    """
    return {
        "title": raw.get("title") or raw.get("job_title"),
        "company": raw.get("company") or raw.get("company_name"),
        "location": raw.get("location") or raw.get("city"),
        "salary": raw.get("salary") or raw.get("salary_range") or raw.get("compensation"),
        "employment_type": raw.get("type") or raw.get("employment_type"),
        "skills": raw.get("skills") or raw.get("keywords") or [],
        "posted_at": raw.get("posted_at") or raw.get("date_posted") or raw.get("created_at"),
        "remote": bool(raw.get("remote") or raw.get("is_remote")),
        "url": raw.get("url") or raw.get("job_url") or raw.get("apply_url"),
        "description": raw.get("description") or raw.get("job_description") or "",
        "source": raw.get("source") or "External API",
    }


# ====== ROUTES ======

@app.route("/", methods=["GET"])
def index():
    # Serve index.html directly
    return app.send_static_file("index.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():
    if not request.is_json:
        return jsonify({"detail": "Expected JSON body"}), 400

    payload = request.get_json(force=True) or {}

    profile = {
        "profile_skills": payload.get("profile_skills", ""),
        "experience_years": payload.get("experience_years"),
        "desired_role": payload.get("desired_role"),
        "location": payload.get("location"),
        "salary_min": payload.get("salary_min"),
        "employment_type": payload.get("employment_type"),
        "remote_only": payload.get("remote_only", False),
    }

    # 1. Get raw jobs from external API
    raw_jobs = call_external_jobs_api(profile)

    if not raw_jobs:
        return jsonify({"jobs": []})

    # 2. Normalize and score via ML model (or placeholder)
    scored_jobs: List[Dict[str, Any]] = []
    for raw in raw_jobs:
        job = normalize_job(raw)
        score = compute_match_score(profile, job)
        job["score"] = float(score)
        scored_jobs.append(job)

    # 3. Sort by score descending
    scored_jobs.sort(key=lambda j: j.get("score", 0.0), reverse=True)

    return jsonify({"jobs": scored_jobs})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)

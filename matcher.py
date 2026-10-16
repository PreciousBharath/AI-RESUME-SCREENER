"""Core matching engine for TalentLens AI."""
import io
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = [
    "python", "java", "javascript", "sql", "machine learning", "deep learning", "nlp",
    "computer vision", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "matplotlib", "power bi", "tableau", "excel", "statistics", "data analysis",
    "data visualization", "generative ai", "llm", "prompt engineering", "langchain",
    "rag", "docker", "kubernetes", "aws", "azure", "gcp", "git", "flask", "django",
    "fastapi", "react", "html", "css", "mongodb", "postgresql", "spark", "airflow",
    "mlops", "blockchain", "cybersecurity", "communication", "leadership",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def extract_skills(text: str) -> set:
    t = normalize(text)
    return {s for s in SKILLS if re.search(r"(?<![a-z])" + re.escape(s) + r"(?![a-z])", t)}


def read_upload(file) -> str:
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")


def rank_resumes(jd: str, resumes: dict, w_sem: float = 0.5, w_skill: float = 0.5):
    names = list(resumes)
    docs = [normalize(jd)] + [normalize(resumes[n]) for n in names]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    tfidf = vec.fit_transform(docs)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).ravel()
    # rescale similarity so the best match is near the top of the range
    sem = sims / sims.max() if sims.max() > 0 else sims

    jd_skills = extract_skills(jd)
    rows = []
    for name, s in zip(names, sem):
        cand = extract_skills(resumes[name])
        matched = jd_skills & cand
        missing = jd_skills - cand
        skill_cov = len(matched) / len(jd_skills) if jd_skills else 0
        score = 100 * (w_sem * s + w_skill * skill_cov)
        rows.append({
            "Candidate": name.rsplit(".", 1)[0],
            "Score": round(score, 1),
            "Semantic %": round(100 * s, 1),
            "Skill %": round(100 * skill_cov, 1),
            "Matched skills": ", ".join(sorted(matched)),
            "Missing skills": ", ".join(sorted(missing)),
        })
    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df, jd_skills


SAMPLE_JD = """We are hiring a Machine Learning Engineer.
Requirements: strong Python, SQL and statistics; hands-on machine learning and deep learning
with scikit-learn, TensorFlow or PyTorch; experience with NLP, LLM and generative AI applications,
prompt engineering and RAG using LangChain; deploy models with Docker, FastAPI and AWS; Git;
good communication skills."""

SAMPLE_RESUMES = {
    "Priya_Sharma.txt": """ML Engineer with 4 years experience. Python, SQL, statistics, scikit-learn,
TensorFlow, PyTorch. Built NLP pipelines and LLM chatbots with LangChain and RAG, prompt engineering.
Deployed models using Docker, FastAPI on AWS. Git, MLOps. Strong communication.""",
    "Arjun_Kumar.txt": """Data Analyst. Excel, Power BI, Tableau, SQL, Python, pandas, data visualization,
statistics. Created executive dashboards and data analysis reports. Communication and leadership.""",
    "Meera_Iyer.txt": """Deep learning researcher: computer vision with PyTorch and TensorFlow, Python, numpy,
machine learning, statistics. Published papers. Git, Docker.""",
    "Rahul_Verma.txt": """Full stack developer: JavaScript, React, HTML, CSS, Django, Flask, PostgreSQL, MongoDB,
Docker, Git, AWS. Built REST APIs with FastAPI.""",
    "Sneha_Reddy.txt": """Generative AI developer. Python, LLM, generative AI, prompt engineering, RAG, LangChain,
NLP, FastAPI, Azure, Git. Machine learning with scikit-learn. Communication.""",
    "Karthik_Raj.txt": """Cybersecurity analyst: network security, SIEM, Python scripting, Linux, cybersecurity
audits, blockchain basics, communication.""",
}

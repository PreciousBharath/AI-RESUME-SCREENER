<div align="center">

# 🎯 TalentLens AI — Intelligent Resume Screening & Job Matching

**Rank hundreds of resumes against a job description in seconds — with transparent, explainable scores.**

![Python-3.10%2B](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit-App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Plotly-Charts](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white) ![scikit--learn-NLP](https://img.shields.io/badge/scikit--learn-NLP-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![Domain-AI%20%2F%20NLP](https://img.shields.io/badge/Domain-AI%20%2F%20NLP-8E2DE2?style=for-the-badge&logo=openai&logoColor=white)

`Artificial Intelligence · NLP`

</div>

---

![AI-RESUME-SCREENER](assets/screenshot.png)

## 📌 Problem Statement

Recruiters spend 6–8 seconds per resume and still miss strong candidates. Keyword filters in ATS tools are rigid and opaque.

## ✨ Key Features

- Upload **PDF / TXT** resumes in bulk or try the built-in sample set
- **Hybrid score** = TF-IDF semantic similarity (unigrams + bigrams) + skill-coverage ratio
- Skill extraction from a 45+ skill taxonomy with **matched vs missing skills** per candidate
- Adjustable weights and shortlist threshold from the sidebar
- Ranking chart, score-breakdown chart and **CSV report export**

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Job Description] --> C[TF-IDF Vectorizer]
    B[Resumes PDF/TXT] --> P[PDF Parser] --> C
    C --> D[Cosine Similarity]
    A --> E[Skill Extractor]
    P --> E
    D --> F[Weighted Scorer]
    E --> F --> G[Ranked Shortlist + CSV]
```

## ⚙️ How It Works

1. Text is normalised and vectorised with **TF-IDF** (`ngram_range=(1,2)`, sublinear TF).
2. **Cosine similarity** between the JD vector and every resume gives the semantic score (rescaled to the best match).
3. Regex-based **skill matcher** finds skills in the JD and in each resume → coverage = matched / required.
4. Final score = `w × semantic + (1 − w) × skill coverage`, shortlisting candidates above the threshold.

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML / Logic | pandas, scikit-learn, pypdf |
| UI & Visualisation | Streamlit, Plotly |

## 📁 Project Structure

```
AI-RESUME-SCREENER/
├── app.py             # Streamlit UI — inputs, KPIs, charts, ranked table
├── matcher.py         # Matching engine: parsing, skill extraction, TF-IDF scoring, sample data
├── assets/              # screenshots
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

```bash
git clone https://github.com/PreciousBharath/AI-RESUME-SCREENER.git
cd AI-RESUME-SCREENER
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> The project ships with a realistic built-in demo dataset, so it runs out of the box with no downloads or API keys.

## 🗺️ Roadmap

- [ ] Sentence-transformer embeddings for deeper semantic matching
- [ ] LLM-generated interview questions per candidate
- [ ] Bias audit report (name/gender-blind scoring)

## 👨‍💻 Author

**Bharath** — Software Developer · Project Engineer · Technical Trainer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-connectbharath-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/connectbharath)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://itsbharathportfolio.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-PreciousBharath-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PreciousBharath)

⭐ If this project helped you, please give it a star!

## 📄 License

Released under the [MIT License](LICENSE).

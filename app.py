"""TalentLens AI - Intelligent Resume Screening & Job Matching.

Ranks resumes against a job description using TF-IDF semantic similarity
plus skill-coverage scoring, and explains every score.
Run:  streamlit run app.py
"""
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from matcher import SAMPLE_JD, SAMPLE_RESUMES, rank_resumes, read_upload

st.set_page_config(page_title="TalentLens AI", page_icon="🎯", layout="wide")

st.markdown(
    """<h1 style='margin-bottom:0'>🎯 TalentLens AI</h1>
    <p style='color:#888;margin-top:4px'>Intelligent resume screening & job matching with explainable scores</p>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Scoring weights")
    w_sem = st.slider("Semantic similarity", 0.0, 1.0, 0.5, 0.05)
    w_skill = 1 - w_sem
    st.caption(f"Skill coverage weight: **{w_skill:.2f}**")
    shortlist = st.slider("Shortlist threshold (%)", 0, 100, 60)
    st.divider()
    st.caption("Upload .txt or .pdf resumes, or use the built-in sample set.")

left, right = st.columns([1, 1])
with left:
    jd = st.text_area("📄 Job description", SAMPLE_JD, height=230)
with right:
    files = st.file_uploader("📎 Upload resumes", type=["txt", "pdf"], accept_multiple_files=True)
    use_samples = st.checkbox("Use sample resumes", value=not files)

resumes = {f.name: read_upload(f) for f in files} if files else {}
if use_samples:
    resumes.update(SAMPLE_RESUMES)

if not resumes or not jd.strip():
    st.info("Add a job description and at least one resume to start.")
    st.stop()

df, jd_skills = rank_resumes(jd, resumes, w_sem, w_skill)
df["Status"] = df["Score"].apply(lambda s: "✅ Shortlisted" if s >= shortlist else "⏳ Review")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Candidates", len(df))
k2.metric("Shortlisted", int((df["Score"] >= shortlist).sum()))
k3.metric("Top score", f"{df['Score'].max():.1f}%")
k4.metric("Skills in JD", len(jd_skills))

c1, c2 = st.columns([3, 2])
with c1:
    fig = px.bar(
        df.sort_values("Score"), x="Score", y="Candidate", orientation="h",
        color="Score", color_continuous_scale="Tealgrn", range_x=[0, 100],
        title="Candidate ranking",
    )
    fig.add_vline(x=shortlist, line_dash="dash", line_color="#F75C7E")
    fig.update_layout(height=360, coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
with c2:
    melted = df.melt(id_vars="Candidate", value_vars=["Semantic %", "Skill %"], var_name="Component", value_name="Value")
    fig2 = px.bar(melted, x="Candidate", y="Value", color="Component", barmode="group",
                  title="Score breakdown", color_discrete_sequence=["#8E2DE2", "#00C9FF"])
    fig2.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_tickangle=-30)
    st.plotly_chart(fig2, width="stretch")

st.subheader("📋 Ranked candidates")
st.dataframe(
    df[["Rank", "Candidate", "Score", "Semantic %", "Skill %", "Matched skills", "Missing skills", "Status"]],
    width="stretch", hide_index=True,
    column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f%%")},
)

st.download_button("⬇️ Download report (CSV)", df.to_csv(index=False), "screening_report.csv", "text/csv")

with st.expander("🔍 Skills detected in the job description"):
    st.write(", ".join(sorted(jd_skills)) or "No known skills detected.")

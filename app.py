import streamlit as st
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from spacy.cli import download

# --- Load small spaCy model (en_core_web_sm) ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Define a basic skill list (can be expanded) ---
SKILL_KEYWORDS = [
    "python", "java", "sql", "excel", "communication", "leadership", "project management",
    "teamwork", "problem solving", "data analysis", "machine learning", "deep learning",
    "time management", "presentation", "cloud", "aws", "azure", "docker", "linux"
]

# --- Page settings ---
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("🔍 Resume Analyzer with AI")
st.markdown("Upload resumes and compare them to a job description. Get match scores, extract skills, highlight keywords, and more.")

# --- Sidebar inputs ---
st.sidebar.header("📂 Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Select resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

st.sidebar.header("📝 Job Description")
job_desc = st.sidebar.text_area("Paste the job description here", height=200)

# --- Helper Functions ---
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill in SKILL_KEYWORDS if skill.lower() in text_lower]

def highlight_keywords(text, keywords):
    for word in keywords:
        word_regex = re.compile(rf"(?i)\b({re.escape(word)})\b")
        text = word_regex.sub(r"**🟨\1**", text)
    return text

# --- Main Analysis ---
results = []
if uploaded_files and job_desc:
    if st.button("🚀 Analyze All Resumes"):
        with st.spinner("Analyzing resumes..."):
            for file in uploaded_files:
                resume_text = extract_text(file)
                skills = extract_skills(resume_text)

                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform([resume_text, job_desc])
                match_score = cosine_similarity(tfidf[0], tfidf[1])[0][0] * 100

                job_keywords = re.findall(r'\b\w+\b', job_desc.lower())
                highlighted_resume = highlight_keywords(resume_text, job_keywords)

                results.append({
                    "Filename": file.name,
                    "Match Score (%)": round(match_score, 1),
                    "Skills": ", ".join(skills[:10]) if skills else "N/A",
                    "Highlighted Resume": highlighted_resume,
                    "Raw Resume Text": resume_text
                })

        # --- Show Summary ---
        sorted_results = sorted(results, key=lambda x: x["Match Score (%)"], reverse=True)

        st.subheader("📊 Resume Match Summary")
        df_display = pd.DataFrame(sorted_results).drop(columns=["Highlighted Resume", "Raw Resume Text"])
        st.dataframe(df_display)

        # --- Detail View ---
        st.subheader("🧾 Detailed Resume Insights")
        for res in sorted_results:
            with st.expander(f"📄 {res['Filename']} (Score: {res['Match Score (%)']}%)"):
                st.markdown(f"**Skills:** {res['Skills']}")
                st.markdown("**Resume with Highlighted Keywords:**")
                st.markdown(res["Highlighted Resume"], unsafe_allow_html=True)

        # --- Ranking ---
        st.subheader("🏆 Resume Ranking")
        for idx, res in enumerate(sorted_results):
            st.markdown(f"**{idx+1}. {res['Filename']}** — {res['Match Score (%)']}% match")

        # --- Bar Chart ---
        st.subheader("📈 Match Score Comparison")
        chart_df = pd.DataFrame(sorted_results)
        st.bar_chart(chart_df.set_index("Filename")["Match Score (%)"])

        # --- Missing Keywords ---
        st.subheader("💡 Missing Keywords Suggestion")
        job_keywords_set = set(re.findall(r'\b\w+\b', job_desc.lower()))
        stop_words = set([
            "a", "an", "the", "in", "on", "at", "of", "to", "for", "and", "or",
            "is", "be", "are", "was", "were", "has", "have", "had", "do", "does",
            "did", "will", "would", "should", "can", "could", "may", "might", "must"
        ])

        for res in sorted_results:
            resume_words = set(re.findall(r'\b\w+\b', res["Raw Resume Text"].lower()))
            missing = job_keywords_set - resume_words - stop_words
            missing_keywords = sorted(list(missing))[:15]
            with st.expander(f"🔎 {res['Filename']} – Missing Keywords"):
                if missing_keywords:
                    st.markdown(", ".join(missing_keywords))
                else:
                    st.success("No missing keywords found 🎉")

        # --- Export CSV ---
        if st.button("💾 Export Results to CSV"):
            export_df = pd.DataFrame(sorted_results).drop(columns=["Highlighted Resume", "Raw Resume Text"])
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_data, "resume_analysis_results.csv", "text/csv")

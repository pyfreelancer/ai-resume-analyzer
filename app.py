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

# --- SpaCy Model Loading ---
# This block attempts to load the spaCy model. If it's not found (e.g., on a fresh deployment
# environment like Streamlit Sharing), it will download it programmatically.
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    st.warning("SpaCy 'en_core_web_lg' model not found. Attempting to download...")
    # Using spacy.cli.download for programmatic download
    # This might take a moment during the first run on a new environment
    with st.spinner("Downloading large spaCy model (this may take a few minutes for first-time deploy)..."):
        try:
            spacy.cli.download("en_core_web_lg")
            nlp = spacy.load("en_core_web_lg")
            st.success("SpaCy 'en_core_web_lg' model downloaded and loaded successfully!")
        except Exception as e:
            st.error(f"Failed to download spaCy model: {e}")
            st.stop() # Stop the app if model download fails critically
# Page settings
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Sidebar for inputs
st.sidebar.header("📂 Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Select resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

st.sidebar.header("📝 Job Description")
job_desc = st.sidebar.text_area("Paste the job description here", height=200)

# Main Title
st.title("🔍 Resume Analyzer with AI")
st.markdown("Upload resumes and compare them to a job description. Get match scores, extract skills, highlight keywords, and more.")

# Functions
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

def extract_skills(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ == "SKILL"))

def highlight_keywords(text, keywords):
    for word in keywords:
        word_regex = re.compile(rf"(?i)\b({re.escape(word)})\b")
        text = word_regex.sub(r"**🟨\1**", text)
    return text

# Analysis section
results = []
if uploaded_files and job_desc:
    if st.button("🚀 Analyze All Resumes"):
        with st.spinner("Analyzing resumes..."):
            for file in uploaded_files:
                resume_text = extract_text(file)
                skills = extract_skills(resume_text)

                # Match score
                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform([resume_text, job_desc])
                match_score = cosine_similarity(tfidf[0], tfidf[1])[0][0] * 100

                # Keyword Highlighting
                job_keywords = re.findall(r'\b\w+\b', job_desc.lower())
                highlighted_resume = highlight_keywords(resume_text, job_keywords)

                results.append({
                    "Filename": file.name,
                    "Match Score (%)": round(match_score, 1),
                    "Skills": ", ".join(skills[:10]),
                    "Highlighted Resume": highlighted_resume
                })

        # Display Table
        st.subheader("📊 Resume Match Summary")
        df = pd.DataFrame(results).drop(columns=["Highlighted Resume"])
        st.dataframe(df)

        # Detailed Resume View
        st.subheader("🧾 Detailed Resume Insights")
        for res in results:
            with st.expander(f"📄 {res['Filename']}"):
                st.markdown(f"**Match Score:** {res['Match Score (%)']}%")
                st.markdown(f"**Skills:** {res['Skills']}")
                st.markdown("**Resume with Highlighted Keywords:**")
                st.markdown(res["Highlighted Resume"])

        # Ranking and Bar Chart
        st.subheader("🏆 Resume Ranking")
        sorted_results = sorted(results, key=lambda x: x["Match Score (%)"], reverse=True)
        for idx, res in enumerate(sorted_results):
            st.markdown(f"**{idx+1}. {res['Filename']}** — {res['Match Score (%)']}% match")

        st.subheader("📈 Match Score Comparison")
        chart_df = pd.DataFrame(sorted_results)
        plt.figure(figsize=(10, 5))
        sns.barplot(x="Filename", y="Match Score (%)", data=chart_df, palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.title("Resume Match Scores")
        st.pyplot(plt)

        # Missing Keywords
        st.subheader("💡 Missing Keywords Suggestion")
        job_keywords_set = set(re.findall(r'\b\w+\b', job_desc.lower()))
        for res in sorted_results:
            resume_words = set(re.findall(r'\b\w+\b', res["Highlighted Resume"].lower()))
            missing = job_keywords_set - resume_words
            common_missing = sorted(list(missing))[:10]
            with st.expander(f"🔎 {res['Filename']} – Missing Keywords"):
                if common_missing:
                    st.markdown(", ".join(common_missing))
                else:
                    st.success("No missing keywords found 🎉")

        # Export to CSV
        if st.button("💾 Export Results to CSV"):
            df.to_csv("resume_analysis_results.csv", index=False)
            st.success("Results saved to resume_analysis_results.csv ✅")

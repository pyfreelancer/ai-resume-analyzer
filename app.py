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
import os # Added for path manipulation

# --- SpaCy Model Loading (Correct & Robust for Streamlit Cloud) ---
# Define a local directory within the app's space where spaCy models can be stored.
# This directory is writable on Streamlit Sharing.
SPACY_MODEL_DIR = os.path.join(os.path.dirname(__file__), ".spacy_models")
os.makedirs(SPACY_MODEL_DIR, exist_ok=True) # Ensure the directory exists

# Set the SPACY_DATA environment variable to this custom directory.
# This tells spaCy where to look for models and where to download them using spacy.cli.download.
os.environ["SPACY_DATA"] = SPACY_MODEL_DIR

# Define the specific model we want to use (preferring lg for resume analysis)
SPACY_MODEL_NAME = "en_core_web_lg" # Changed to lg for better performance

# Check if the model is already downloaded in our custom path
# spacy.util.is_model_loaded check might be too complex here, simple path check is often enough.
model_path_exists = os.path.exists(os.path.join(SPACY_MODEL_DIR, SPACY_MODEL_NAME))

if not model_path_exists:
    st.warning(f"SpaCy '{SPACY_MODEL_NAME}' model not found. Attempting to download into custom path...")
    with st.spinner(f"Downloading large spaCy model ({SPACY_MODEL_NAME}). This may take a few minutes for first-time deploy..."):
        try:
            # spacy.cli.download respects SPACY_DATA environment variable.
            # This will download the model into the SPACY_MODEL_DIR.
            spacy.cli.download(SPACY_MODEL_NAME)
            st.success(f"SpaCy '{SPACY_MODEL_NAME}' model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download spaCy model '{SPACY_MODEL_NAME}': {e}")
            st.stop() # Stop the app if model download fails critically
else:
    st.info(f"SpaCy '{SPACY_MODEL_NAME}' model found. Loading...")

try:
    # Load the model from the custom path (now that SPACY_DATA is set)
    nlp = spacy.load(SPACY_MODEL_NAME)
    st.success(f"SpaCy '{SPACY_MODEL_NAME}' model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load spaCy model '{SPACY_MODEL_NAME}' even after download attempt: {e}")
    st.stop()


# --- Page Settings ---
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# --- Sidebar for Inputs ---
st.sidebar.header("📂 Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Select resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

st.sidebar.header("📝 Job Description")
job_desc = st.sidebar.text_area("Paste the job description here", height=200)

# --- Main Title ---
st.title("🔍 Resume Analyzer with AI")
st.markdown("Upload resumes and compare them to a job description. Get match scores, extract skills, highlight keywords, and more.")

# --- Functions ---
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

def extract_skills(text):
    doc = nlp(text) # Use the loaded nlp model
    # Note: Ensure your 'SKILL' entity label is correct for the en_core_web_lg model
    # You might need a custom pipeline or different entity recognition for specific skills
    # If this doesn't extract skills as expected, consider fine-tuning or a different approach
    return list(set(ent.text for ent in doc.ents if ent.label_ == "SKILL")) # Keep SKILL or change based on your nlp model's entity recognition

def highlight_keywords(text, keywords):
    # This function uses regex to find whole words and make them bold with a yellow square emoji.
    for word in keywords:
        # (?i) makes it case-insensitive
        # \b ensures whole word match (word boundaries)
        word_regex = re.compile(rf"(?i)\b({re.escape(word)})\b")
        text = word_regex.sub(r"**🟨\1**", text) # Adds yellow square and bold
    return text

# --- Analysis Section ---
results = []
if uploaded_files and job_desc:
    if st.button("🚀 Analyze All Resumes"):
        with st.spinner("Analyzing resumes..."):
            for file in uploaded_files:
                resume_text = extract_text(file)
                skills = extract_skills(resume_text)

                # Match score calculation
                # Using max_features to limit vocabulary size for efficiency if texts are very large
                vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                # It's important to fit_transform on both texts combined to ensure same vocabulary
                tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
                
                # Calculate cosine similarity between the first (resume) and second (job_desc) vectors
                match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

                # Keyword Highlighting
                # Extract words from job description, convert to lowercase for consistent matching
                job_keywords = re.findall(r'\b\w+\b', job_desc.lower())
                highlighted_resume = highlight_keywords(resume_text, job_keywords)

                results.append({
                    "Filename": file.name,
                    "Match Score (%)": round(match_score, 1),
                    "Skills": ", ".join(skills) if skills else "N/A", # Handle cases with no skills
                    "Highlighted Resume": highlighted_resume,
                    "Original Resume Text": resume_text # Store original text for missing keywords
                })

        # --- Display Results ---

        # Sort results by match score in descending order for ranking
        sorted_results = sorted(results, key=lambda x: x["Match Score (%)"], reverse=True)

        # Display Summary Table
        st.subheader("📊 Resume Match Summary")
        df_display = pd.DataFrame(sorted_results).drop(columns=["Highlighted Resume", "Original Resume Text"])
        st.dataframe(df_display, use_container_width=True)

        # Detailed Resume View
        st.subheader("🧾 Detailed Resume Insights")
        for res in sorted_results:
            with st.expander(f"📄 {res['Filename']} (Score: {res['Match Score (%)']}%)"):
                st.markdown(f"**Match Score:** {res['Match Score (%)']}%")
                st.markdown(f"**Skills:** {res['Skills']}")
                st.markdown("---") # Separator for readability
                st.markdown("**Resume with Highlighted Keywords:**")
                st.markdown(res["Highlighted Resume"], unsafe_allow_html=True) # unsafe_allow_html for markdown bold/emoji

        # Ranking and Bar Chart
        st.subheader("🏆 Resume Ranking")
        for idx, res in enumerate(sorted_results):
            st.markdown(f"**{idx+1}. {res['Filename']}** — {res['Match Score (%)']}% match")

        st.subheader("📈 Match Score Comparison")
        chart_df = pd.DataFrame(sorted_results)
        # Use Streamlit's built-in charting for better integration and responsiveness
        st.bar_chart(chart_df.set_index("Filename")["Match Score (%)"])
        # If you prefer matplotlib, uncomment below and comment out st.bar_chart
        # plt.figure(figsize=(10, 5))
        # sns.barplot(x="Filename", y="Match Score (%)", data=chart_df, palette="viridis")
        # plt.xticks(rotation=45, ha="right")
        # plt.title("Resume Match Scores")
        # st.pyplot(plt)

        # Missing Keywords
        st.subheader("💡 Missing Keywords Suggestion")
        job_keywords_set = set(re.findall(r'\b\w+\b', job_desc.lower()))

        for res in sorted_results:
            # Use the stored original resume text for accurate missing keyword calculation
            resume_words_set = set(re.findall(r'\b\w+\b', res["Original Resume Text"].lower()))
            
            missing = job_keywords_set - resume_words_set
            
            # Filter out very common words that are unlikely to be "missing keywords"
            # You can expand this list if needed
            stop_words_for_missing = set(re.findall(r'\b\w+\b', "a an the in on at of to for and or is be are was were has have had do does did will would should can could may might must".lower()))
            missing = missing - stop_words_for_missing

            common_missing = sorted(list(missing))[:15] # Show up to 15 relevant missing words
            with st.expander(f"🔎 {res['Filename']} – Missing Keywords"):
                if common_missing:
                    st.markdown("Keywords you could add: " + ", ".join(common_missing))
                else:
                    st.success("No significant missing keywords found from job description! 🎉")

        # --- Export to CSV ---
        # Ensure 'df' contains all necessary data for export
        if st.button("💾 Export Results to CSV"):
            # Prepare a DataFrame for export, dropping "Highlighted Resume" and "Original Resume Text"
            # as these are not typically desired in a simple CSV export
            df_export = pd.DataFrame(results).drop(columns=["Highlighted Resume", "Original Resume Text"])
            
            # Convert DataFrame to CSV string
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="resume_analysis_results.csv",
                mime="text/csv",
            )
            st.success("Results ready for download ✅")
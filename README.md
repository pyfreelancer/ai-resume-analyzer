# 🧠 AI Resume Analyzer

An intelligent web application built with **Streamlit** that allows users to upload and analyze resumes against a job description using **NLP** and **machine learning**. The tool calculates match scores, extracts skills, highlights job-specific keywords, and provides insights to help recruiters and job seekers.

---

## 🚀 Features

- 📂 Upload and analyze multiple resumes (PDF or DOCX)
- 📊 Match each resume to the job description using **TF-IDF + cosine similarity**
- 🛠️ Extract relevant skills using **spaCy NLP**
- 🟨 Highlight job keywords in resumes for clarity
- 🏆 Automatically rank resumes by match score
- 📈 Visualize results with a bar chart
- 💡 Show missing job-specific keywords for each resume
- 💾 Export results as a CSV file

---

## 🧰 Tech Stack

- **Python**
- **Streamlit** (Web UI)
- **spaCy** (`en_core_web_lg` model)
- **Scikit-learn** (TF-IDF & cosine similarity)
- **Pandas**, **Matplotlib**, **Seaborn**

---

## 📦 Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/ai-resume-analyzer.git
   cd ai-resume-analyzer

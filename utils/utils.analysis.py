import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_lg")

def extract_skills(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ == "SKILL"))

def analyze_resume(resume_text, job_desc):
    skills = extract_skills(resume_text)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100
    return score, skills

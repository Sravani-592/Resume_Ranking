import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()  # Corrected method name
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes # Ensure job description is the first element
    vectorizer = TfidfVectorizer().fit(documents) # Fit on all documents at once
    vectors = vectorizer.transform(documents).toarray() # Transform all documents

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(job_description_vector.reshape(1, -1), resume_vectors).flatten() # Reshape for cosine_similarity

    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    resumes = []
    file_names = []  # Store file names for display

    for file in uploaded_files:
        try: # Handle potential PDF read errors
            text = extract_text_from_pdf(file)
            resumes.append(text)
            file_names.append(file.name)
        except Exception as e:
            st.error(f"Error reading PDF {file.name}: {e}")
            continue # Skip to the next file

    if resumes: # Only proceed if resumes were successfully extracted
        scores = rank_resumes(job_description, resumes)
        results = pd.DataFrame({"Resume": file_names, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)
        st.write(results)
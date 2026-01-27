import streamlit as st
import pickle
import re
import pdfplumber

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="centered"
)

st.markdown("<h1 style='text-align:center;'>📝 AI-Based Resume Screening</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Machine Learning powered resume analysis and skill gap detection</p>",
    unsafe_allow_html=True
)
st.divider()

# ---------------- LOAD MODEL ----------------
with open("resume_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# ---------------- SKILL DATABASE ----------------
ROLE_SKILLS = {
    "Data Scientist": ["python", "machine learning", "statistics", "pandas", "numpy"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau"],
    "Web Developer": ["html", "css", "javascript", "react"],
    "Software Engineer": ["java", "spring", "api", "oop"],
    "Cyber Security": ["network", "security", "linux", "penetration"],
    "AI Engineer": ["deep learning", "tensorflow", "pytorch", "nlp"],
    "HR": ["recruitment", "onboarding", "communication"],
    "Graphic Designer": ["photoshop", "illustrator", "design"]
}

# ---------------- FUNCTIONS ----------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    return text

def calculate_score(found, total):
    if total == 0:
        return 0
    return int((found / total) * 100)

# ---------------- INPUT UI ----------------
st.subheader("📥 Upload Resume")

uploaded_file = st.file_uploader(
    "Upload resume (PDF only)",
    type=["pdf"]
)

target_role = st.selectbox(
    "Select Target Job Role",
    list(ROLE_SKILLS.keys())
)

analyze = st.button("🔍 Analyze Resume")

# ---------------- ANALYSIS ----------------
if analyze:

    # ✅ FIXED VALIDATION
    if uploaded_file is None:
        st.warning("⚠️ Please upload a resume PDF to continue.")
        st.stop()

    with st.spinner("Analyzing resume..."):

        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_resume = clean_text(resume_text)

        # Predict job role
        vectorized = tfidf.transform([cleaned_resume])
        predicted_role = model.predict(vectorized)[0]

        # Skill analysis
        required_skills = ROLE_SKILLS[target_role]
        found_skills = [skill for skill in required_skills if skill in cleaned_resume]
        missing_skills = [skill for skill in required_skills if skill not in cleaned_resume]

        score = calculate_score(len(found_skills), len(required_skills))
        status = "🟢 Shortlisted" if score >= 60 else "🔴 Needs Improvement"

    # ---------------- OUTPUT ----------------
    st.divider()
    st.subheader("📊 Resume Screening Result")

    st.markdown(f"**🎯 Predicted Job Role:** `{predicted_role}`")
    st.markdown(f"**📈 Resume Match Score:** `{score}%`")
    st.markdown(f"**📌 Final Status:** {status}")

    st.subheader("✅ Skills Found")
    st.write(found_skills if found_skills else "No matching skills found")

    st.subheader("❌ Missing Skills")
    st.write(missing_skills if missing_skills else "None 🎉")
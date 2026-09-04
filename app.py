from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")


class Student(BaseModel):
    Age: int
    Gender: str
    CGPA: float
    Internships: int
    Projects: int
    Coding_Skills: float
    Communication_Skills: float
    Aptitude_Test_Score: float
    Soft_Skills_Rating: float
    Certifications: int
    Backlogs: int
    Branch: str
    Degree: str


@app.post("/predict")
def predict(student: Student):

    gender = 1 if student.Gender == "Male" else 0

    branch_civil = 1 if student.Branch == "Civil" else 0
    branch_ece = 1 if student.Branch == "ECE" else 0
    branch_it = 1 if student.Branch == "IT" else 0
    branch_me = 1 if student.Branch == "ME" else 0

    degree_btech = 1 if student.Degree == "B.Tech" else 0
    degree_bca = 1 if student.Degree == "BCA" else 0
    degree_mca = 1 if student.Degree == "MCA" else 0

    student_data = pd.DataFrame([{
        "Age": student.Age,
        "Gender": gender,
        "CGPA": student.CGPA,
        "Internships": student.Internships,
        "Projects": student.Projects,
        "Coding_Skills": student.Coding_Skills,
        "Communication_Skills": student.Communication_Skills,
        "Aptitude_Test_Score": student.Aptitude_Test_Score,
        "Soft_Skills_Rating": student.Soft_Skills_Rating,
        "Certifications": student.Certifications,
        "Backlogs": student.Backlogs,
        "Branch_Civil": branch_civil,
        "Branch_ECE": branch_ece,
        "Branch_IT": branch_it,
        "Branch_ME": branch_me,
        "Degree_B.Tech": degree_btech,
        "Degree_BCA": degree_bca,
        "Degree_MCA": degree_mca
    }])

    scaled_data = scaler.transform(student_data)

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)


    return {
        "prediction": "Placed" if prediction[0] == 1 else "Not Placed",
        "probability": float(probability[0][1])
    }

    # ===== Skill Recommender=====

def get_skill_recommendations(student: Student):
    recommendations = []

    if student.CGPA < 7:
        recommendations.append("Focus on improving your CGPA — aim for consistent performance above 7.0")

    if student.Coding_Skills < 7:
        recommendations.append("Practice coding daily on platforms like LeetCode or HackerRank")

    if student.Communication_Skills < 7:
        recommendations.append("Join a public speaking or communication skills workshop")

    if student.Aptitude_Test_Score < 70:
        recommendations.append("Practice aptitude tests — quantitative, logical, and verbal reasoning")

    if student.Soft_Skills_Rating < 7:
        recommendations.append("Work on teamwork and soft skills through group projects or clubs")

    if student.Internships < 1:
        recommendations.append("Apply for at least one internship to gain practical experience")

    if student.Projects < 2:
        recommendations.append("Build 2-3 solid projects to showcase on your resume")

    if student.Certifications < 1:
        recommendations.append("Earn a relevant online certification (Coursera, Udemy, etc.)")

    if student.Backlogs > 0:
        recommendations.append("Clear pending backlogs as a priority — they impact placement eligibility")

    if len(recommendations) == 0:
        recommendations.append("Your profile looks strong across all areas — keep it up!")

    return recommendations


@app.post("/recommend-skills")
def recommend_skills(student: Student):
    recommendations = get_skill_recommendations(student)
    return {"recommendations": recommendations}


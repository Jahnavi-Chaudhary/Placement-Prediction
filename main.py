from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
model = joblib.load('api/placement_rf_model.pkl')
scaler = joblib.load('api/placement_scaler.pkl')
app = FastAPI()
class StudentData(BaseModel):
    CGPA: float
    Internships: int
    Projects: int
    Coding_Skills: int
    Communication_Skills: int
    Aptitude_Test_Score: int
    Soft_Skills_Rating: int
    Certifications: int
    Backlogs: int
    Age: int
    Gender: int
    Branch_Civil: int
    Branch_ECE: int
    Branch_IT: int
    Branch_ME: int
    Degree_BTech: int
    Degree_BCA: int
    Degree_MCA: int

@app.post("/predict")
def predict_placement(student: StudentData):
    input_data = np.array([[
        student.CGPA, student.Internships, student.Projects,
        student.Coding_Skills, student.Communication_Skills,
        student.Aptitude_Test_Score, student.Soft_Skills_Rating,
        student.Certifications, student.Backlogs, student.Age,
        student.Gender, student.Branch_Civil, student.Branch_ECE,
        student.Branch_IT, student.Branch_ME, student.Degree_BTech,
        student.Degree_BCA, student.Degree_MCA
    ]])

    scaled_input = scaler.transform(input_data)
    probability = model.predict_proba(scaled_input)[0][1]

    return {"placement_probability": round(probability * 100, 2)}    
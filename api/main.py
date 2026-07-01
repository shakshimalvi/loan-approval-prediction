import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Loan Approval Prediction API")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "model", "loan_model.pkl")

model = joblib.load(model_path)

class LoanRequest(BaseModel):
    gender: int
    married: int
    dependents: int
    education: int
    self_employed: int
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_term: float
    credit_history: float
    property_area: int


@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: LoanRequest):

    input_data = np.array([[
        data.gender,
        data.married,
        data.dependents,
        data.education,
        data.self_employed,
        data.applicant_income,
        data.coapplicant_income,
        data.loan_amount,
        data.loan_term,
        data.credit_history,
        data.property_area
    ]])

    prediction = model.predict(input_data)[0]

    result = "Approved" if prediction == 1 else "Rejected"

    return {
        "loan_status": result
    }
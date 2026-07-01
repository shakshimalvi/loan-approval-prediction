# Loan Approval Prediction API

## Project Overview
This project predicts whether a loan application will be approved using a Machine Learning model deployed with FastAPI.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

## API Endpoints

### GET /
Returns API status.

### POST /predict
Predicts loan approval.

Example request:

```json
{
  "gender": 1,
  "married": 1,
  "dependents": 0,
  "education": 0,
  "self_employed": 0,
  "applicant_income": 5000,
  "coapplicant_income": 0,
  "loan_amount": 200,
  "loan_term": 360,
  "credit_history": 1,
  "property_area": 2
}
```
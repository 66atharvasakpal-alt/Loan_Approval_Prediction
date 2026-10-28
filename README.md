# Loan Approval Prediction System

Streamlit + scikit-learn + SQLite. Compares Logistic Regression, Decision Tree
and Random Forest, and serves the best (by F1-score) through a web app.

## Structure
    Loan_Approval_Prediction/
    ├── app.py             Streamlit frontend (login, predict, performance, history)
    ├── train_model.py     data loading, preprocessing, training, model selection
    ├── database.py        SQLite users + prediction history
    ├── requirements.txt
    ├── data/
    │   └── loan_data.csv  OPTIONAL real dataset (synthetic data is used if absent)
    └── models/            created automatically (loan_model.joblib, model_metrics.csv)

## Run
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python train_model.py        # optional: the app trains automatically on first run
    streamlit run app.py

## Using a real dataset
Save a CSV as `data/loan_data.csv` with columns: Gender, Married, Dependents,
Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Credit_History, Property_Area, Loan_Status (Y/N).
Extra columns such as Loan_ID are ignored. Then run `python train_model.py`.

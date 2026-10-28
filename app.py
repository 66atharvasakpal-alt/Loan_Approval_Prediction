"""
app.py
------
Streamlit frontend for the Loan Approval Prediction System.

Run with:  streamlit run app.py
"""

import json

import pandas as pd
import streamlit as st

import database as db
from train_model import CHOICES, ensure_model, predict_application

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")
db.init_db()

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1-score"]


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading model (trains automatically on first run)...")
def get_bundle():
    return ensure_model()


# --------------------------------------------------
# AUTH
# --------------------------------------------------
def auth_view():
    st.title("🏦 Loan Approval Prediction System")
    st.caption("Machine-learning based loan eligibility screening")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            user = db.authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        with st.form("register_form"):
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            created = st.form_submit_button("Create account")
        if created:
            if new_pass != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message = db.create_user(new_user, new_pass)
                (st.success if ok else st.error)(message)


# --------------------------------------------------
# PAGES
# --------------------------------------------------
def predict_page(bundle):
    st.header("Loan Application")
    st.caption(f"Active model: **{bundle['model_name']}**")

    with st.form("application_form"):
        applicant_name = st.text_input("Applicant name (optional)")

        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", CHOICES["Gender"])
        married = c2.selectbox("Married", CHOICES["Married"])
        dependents = c3.selectbox("Dependents", CHOICES["Dependents"])

        c1, c2, c3 = st.columns(3)
        education = c1.selectbox("Education", CHOICES["Education"])
        self_employed = c2.selectbox("Self employed", CHOICES["Self_Employed"])
        property_area = c3.selectbox("Property area", CHOICES["Property_Area"])

        c1, c2 = st.columns(2)
        applicant_income = c1.number_input(
            "Applicant income (monthly)", min_value=0, value=5000, step=500
        )
        coapplicant_income = c2.number_input(
            "Co-applicant income (monthly)", min_value=0, value=0, step=500
        )

        c1, c2, c3 = st.columns(3)
        loan_amount = c1.number_input(
            "Loan amount (in thousands)", min_value=1, value=130, step=10
        )
        term = c2.selectbox(
            "Loan term (months)",
            CHOICES["Loan_Amount_Term"],
            index=CHOICES["Loan_Amount_Term"].index(360),
        )
        credit = c3.selectbox(
            "Credit history",
            [1, 0],
            format_func=lambda v: "Good (meets guidelines)" if v == 1 else "Poor / no record",
        )

        submitted = st.form_submit_button("Predict loan status")

    if not submitted:
        return

    if applicant_income + coapplicant_income <= 0:
        st.error("Total household income must be greater than zero.")
        return

    inputs = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": int(applicant_income),
        "CoapplicantIncome": int(coapplicant_income),
        "LoanAmount": int(loan_amount),
        "Loan_Amount_Term": int(term),
        "Credit_History": float(credit),
        "Property_Area": property_area,
    }

    result = predict_application(bundle, inputs)

    st.subheader("Result")
    if result["approved"]:
        st.success("✅ Loan likely to be APPROVED")
    else:
        st.error("❌ Loan likely to be REJECTED")

    st.metric("Approval probability", f"{result['probability'] * 100:.2f}%")
    st.progress(min(max(result["probability"], 0.0), 1.0))

    emi = inputs["LoanAmount"] * 1000 / inputs["Loan_Amount_Term"]
    income = inputs["ApplicantIncome"] + inputs["CoapplicantIncome"]
    st.caption(
        f"Simple repayment estimate (no interest): {emi:,.2f} per month "
        f"= {emi / income * 100:.2f}% of household income."
    )
    st.caption("This is a statistical prediction, not a final lending decision.")

    db.save_prediction(
        st.session_state["user"]["id"],
        applicant_name,
        inputs,
        result["label"],
        result["probability"],
        bundle["model_name"],
    )


def performance_page(bundle):
    st.header("Model Performance")

    metrics = bundle["metrics"]
    best = metrics.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", bundle["model_name"])
    c2.metric("Accuracy", f"{best['Accuracy']:.4f}")
    c3.metric("F1-score", f"{best['F1-score']:.4f}")
    c4.metric("Dataset rows", bundle["n_samples"])

    st.subheader("Model comparison (held-out test set)")
    st.dataframe(
        metrics,
        hide_index=True,
        column_config={
            col: st.column_config.NumberColumn(col, format="%.4f")
            for col in METRIC_COLS + ["CV F1 (5-fold)"]
        },
    )
    st.bar_chart(metrics.set_index("Model")[METRIC_COLS])

    left, right = st.columns(2)
    with left:
        st.subheader(f"Confusion matrix - {bundle['model_name']}")
        cm = pd.DataFrame(
            bundle["confusion_matrix"],
            index=["Actual: Rejected", "Actual: Approved"],
            columns=["Predicted: Rejected", "Predicted: Approved"],
        )
        st.dataframe(cm)
    with right:
        st.subheader("Top features")
        fi = bundle.get("feature_importance")
        if fi is not None and len(fi):
            st.bar_chart(fi.head(10).set_index("Feature")["Importance"])

    st.caption(
        f"Data source: {bundle['data_source']}  |  "
        f"Train/Test: {bundle['n_train']}/{bundle['n_test']}  |  "
        f"Approval rate in data: {bundle['approval_rate'] * 100:.2f}%  |  "
        f"Trained: {bundle['trained_at']}"
    )

    with st.expander("Retrain model"):
        st.write("Re-runs preprocessing, training and model selection on the current dataset.")
        if st.button("Retrain now"):
            with st.spinner("Training..."):
                ensure_model(force=True)
            get_bundle.clear()
            st.rerun()


def history_page():
    st.header("Prediction History")
    rows = db.get_history(st.session_state["user"]["id"])

    if not rows:
        st.info("No predictions saved yet. Make one on the Predict page.")
        return

    records = []
    for r in rows:
        inputs = json.loads(r["inputs_json"])
        records.append(
            {
                "Date": r["created_at"],
                "Applicant": r["applicant_name"] or "-",
                "Result": r["prediction"],
                "Approval probability": r["probability"],
                "Model": r["model_name"],
                **inputs,
            }
        )
    history = pd.DataFrame(records)

    approved = int((history["Result"] == "Approved").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total predictions", len(history))
    c2.metric("Approved", approved)
    c3.metric("Rejected", len(history) - approved)

    st.dataframe(
        history,
        hide_index=True,
        column_config={
            "Approval probability": st.column_config.NumberColumn(format="%.4f")
        },
    )

    st.download_button(
        "Download as CSV",
        history.to_csv(index=False).encode("utf-8"),
        file_name="loan_prediction_history.csv",
        mime="text/csv",
    )

    with st.expander("Clear history"):
        if st.checkbox("I understand this deletes all my saved predictions"):
            if st.button("Delete history"):
                db.clear_history(st.session_state["user"]["id"])
                st.rerun()


def about_page(bundle):
    st.header("About this project")
    st.markdown(
        f"""
**Goal:** predict whether a loan application will be approved or rejected.

**Pipeline**
1. Load data (`data/loan_data.csv`, or an auto-generated dataset if absent)
2. Clean labels, impute missing values, log-scale incomes, one-hot encode categories
3. Train **Logistic Regression**, **Decision Tree** and **Random Forest**
4. Compare on accuracy, precision, recall and F1; keep the best F1 model
   (currently **{bundle['model_name']}**)
5. Serve predictions through this Streamlit app; store users and history in SQLite

**Stack:** Python, scikit-learn, pandas, Streamlit, SQLite
"""
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if "user" not in st.session_state:
        auth_view()
        return

    bundle = get_bundle()

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['user']['username']}")
        page = st.radio(
            "Navigate",
            ["Predict", "Model Performance", "History", "About"],
            label_visibility="collapsed",
        )
        if st.button("Logout"):
            del st.session_state["user"]
            st.rerun()

    if page == "Predict":
        predict_page(bundle)
    elif page == "Model Performance":
        performance_page(bundle)
    elif page == "History":
        history_page()
    else:
        about_page(bundle)


main()

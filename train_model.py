"""
train_model.py
--------------
Data loading, preprocessing, model training/comparison and inference
for the Loan Approval Prediction project.

Run directly to (re)train:   python train_model.py
Imported by app.py via:      from train_model import ensure_model
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "loan_data.csv"              # put a real dataset here (optional)
SYNTHETIC_PATH = DATA_DIR / "synthetic_loan_data.csv"  # auto-generated fallback copy
MODEL_PATH = MODEL_DIR / "loan_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.csv"

# --------------------------------------------------
# SCHEMA
# --------------------------------------------------
TARGET = "Loan_Status"

CATEGORICAL = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
LOG_NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
PLAIN_NUMERIC = ["Loan_Amount_Term", "Credit_History"]
FEATURES = CATEGORICAL + LOG_NUMERIC + PLAIN_NUMERIC

# Allowed values - used by the Streamlit form
CHOICES = {
    "Gender": ["Male", "Female"],
    "Married": ["Yes", "No"],
    "Dependents": ["0", "1", "2", "3+"],
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["No", "Yes"],
    "Property_Area": ["Urban", "Semiurban", "Rural"],
    "Loan_Amount_Term": [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
}

TARGET_MAP = {
    "Y": 1, "N": 0,
    "YES": 1, "NO": 0,
    "APPROVED": 1, "REJECTED": 0,
    "1": 1, "0": 0,
    "1.0": 1, "0.0": 0,
}


# --------------------------------------------------
# DATA
# --------------------------------------------------
def generate_synthetic_data(n_rows: int = 614, seed: int = 42) -> pd.DataFrame:
    """Create a realistic loan dataset (same columns as the public
    'Loan Prediction' dataset) so the project runs with no download."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n_rows, p=[0.80, 0.20])
    married = rng.choice(["Yes", "No"], n_rows, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], n_rows, p=[0.57, 0.17, 0.16, 0.10])
    education = rng.choice(["Graduate", "Not Graduate"], n_rows, p=[0.78, 0.22])
    self_emp = rng.choice(["No", "Yes"], n_rows, p=[0.86, 0.14])
    property_area = rng.choice(
        ["Semiurban", "Urban", "Rural"], n_rows, p=[0.38, 0.33, 0.29]
    )

    applicant = np.clip(rng.lognormal(8.4, 0.55, n_rows), 1500, 80000).round()
    has_co = rng.random(n_rows) < 0.55
    coapplicant = np.where(
        has_co, np.clip(rng.lognormal(7.4, 0.6, n_rows), 500, 40000), 0.0
    ).round()

    total_income = applicant + coapplicant
    loan_amount = np.clip(
        (total_income / 40.0) * rng.lognormal(0.0, 0.3, n_rows), 9, 600
    ).round()
    term = rng.choice(
        [360, 180, 480, 300, 240, 120, 84, 60, 36, 12],
        n_rows,
        p=[0.85, 0.07, 0.025, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005],
    ).astype(float)
    credit = rng.choice([1.0, 0.0], n_rows, p=[0.82, 0.18])

    # Hidden approval rule (credit history dominates, like real lending data)
    burden = (loan_amount * 1000.0 / term) / total_income
    score = (
        -1.9
        + 3.4 * (credit == 1.0)
        + 0.5 * (property_area == "Semiurban")
        + 0.3 * (married == "Yes")
        + 0.3 * (education == "Graduate")
        - 0.2 * (self_emp == "Yes")
        - 3.0 * burden
        + 0.00004 * total_income
    )
    prob = 1.0 / (1.0 + np.exp(-score))
    status = np.where(rng.random(n_rows) < prob, "Y", "N")

    df = pd.DataFrame(
        {
            "Loan_ID": [f"LP{1000 + i:04d}" for i in range(n_rows)],
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_emp,
            "ApplicantIncome": applicant,
            "CoapplicantIncome": coapplicant,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": term,
            "Credit_History": credit,
            "Property_Area": property_area,
            "Loan_Status": status,
        }
    )

    # Sprinkle ~3% missing values (the pipeline imputes them)
    for col in [
        "Gender", "Married", "Dependents", "Self_Employed",
        "LoanAmount", "Loan_Amount_Term", "Credit_History",
    ]:
        mask = rng.random(n_rows) < 0.03
        df[col] = df[col].astype(object)
        df.loc[mask, col] = np.nan
    for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _clean_category(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text if text else np.nan


def encode_target(series: pd.Series) -> pd.Series:
    """Map Y/N, Yes/No, Approved/Rejected, 1/0 to 1/0 (unknown -> NaN)."""
    mapped = series.astype("string").str.strip().str.upper().map(TARGET_MAP)
    return pd.to_numeric(mapped, errors="coerce")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Expected columns: {FEATURES + [TARGET]}"
        )

    for col in CATEGORICAL:
        df[col] = df[col].apply(_clean_category).astype(object)
    for col in LOG_NUMERIC + PLAIN_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = encode_target(df[TARGET])
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def load_data():
    """Return (clean_dataframe, data_source_description)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
        source = f"CSV file: data/{DATA_PATH.name}"
    else:
        raw = generate_synthetic_data()
        raw.to_csv(SYNTHETIC_PATH, index=False)
        source = "Synthetic dataset (auto-generated, 614 rows)"

    return clean_data(raw), source


# --------------------------------------------------
# MODEL
# --------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    log_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    plain_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("log", log_pipe, LOG_NUMERIC),
            ("num", plain_pipe, PLAIN_NUMERIC),
            ("cat", cat_pipe, CATEGORICAL),
        ]
    )


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    names = [
        n.split("__", 1)[-1]
        for n in pipeline.named_steps["preprocessor"].get_feature_names_out()
    ]
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    return (
        pd.DataFrame({"Feature": names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def train_and_save(model_path=None) -> dict:
    model_path = Path(model_path) if model_path else MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df, data_source = load_data()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "Training requires both target classes: approved and rejected."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fitted_models, rows, matrices = {}, [], {}

    for name, classifier in candidates.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        cv_f1 = cross_val_score(pipeline, X, y, cv=cv, scoring="f1").mean()

        fitted_models[name] = pipeline
        matrices[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-score": f1_score(y_test, y_pred, zero_division=0),
                "CV F1 (5-fold)": cv_f1,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.sort_values("F1-score", ascending=False).reset_index(drop=True)

    best_model_name = str(metrics_df.iloc[0]["Model"])
    best_model = fitted_models[best_model_name]

    bundle = {
        "pipeline": best_model,
        "model_name": best_model_name,
        "metrics": metrics_df,
        "confusion_matrix": matrices[best_model_name].tolist(),
        "feature_importance": extract_feature_importance(best_model),
        "data_source": data_source,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "approval_rate": float(y.mean()),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    joblib.dump(bundle, model_path)
    metrics_df.to_csv(METRICS_PATH, index=False)
    return bundle


_REQUIRED_KEYS = {"pipeline", "model_name", "metrics", "confusion_matrix", "sklearn_version"}


def ensure_model(model_path=None, force: bool = False) -> dict:
    """Load the saved model bundle, training a new one if needed."""
    model_path = Path(model_path) if model_path else MODEL_PATH

    if model_path.exists() and not force:
        try:
            bundle = joblib.load(model_path)
            if (
                isinstance(bundle, dict)
                and _REQUIRED_KEYS.issubset(bundle)
                and bundle["sklearn_version"] == sklearn.__version__
            ):
                return bundle
        except Exception:
            pass  # corrupt / old-format file -> retrain below

    return train_and_save(model_path)


# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
def predict_application(bundle: dict, applicant: dict) -> dict:
    row = {col: applicant.get(col) for col in FEATURES}
    row["Dependents"] = str(row["Dependents"])
    X = pd.DataFrame([row], columns=FEATURES)

    pipeline = bundle["pipeline"]
    proba_approved = float(pipeline.predict_proba(X)[0][1])
    approved = proba_approved >= 0.5

    return {
        "approved": bool(approved),
        "label": "Approved" if approved else "Rejected",
        "probability": proba_approved,
    }


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    result = train_and_save()
    pd.set_option("display.width", 140)
    pd.set_option("display.precision", 6)
    print(f"Data source : {result['data_source']}")
    print(f"Rows        : {result['n_samples']} "
          f"(train {result['n_train']} / test {result['n_test']})")
    print(f"Approval %  : {result['approval_rate']:.4f}\n")
    print(result["metrics"].to_string(index=False))
    print(f"\nBest model  : {result['model_name']}")
    print(f"Saved to    : {MODEL_PATH}")

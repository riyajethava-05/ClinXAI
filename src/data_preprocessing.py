from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_dataset(path=DATA_PATH):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md. "
            "Run `python -m src.train_model` to attempt a download."
        )
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain binary labels 0 and 1.")
    for col in ZERO_AS_MISSING:
        df.loc[df[col] == 0, col] = float("nan")
    return df

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer([("numeric", numeric_pipeline, FEATURES)], remainder="drop")

def split_dataset(df, test_size=0.2, random_state=42):
    return train_test_split(
        df[FEATURES], df[TARGET].astype(int), test_size=test_size,
        random_state=random_state, stratify=df[TARGET].astype(int)
    )

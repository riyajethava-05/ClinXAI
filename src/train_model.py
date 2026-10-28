from pathlib import Path
import json
import urllib.request
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from src.data_preprocessing import (
    PROJECT_ROOT,
    load_dataset,
    split_dataset,
    build_preprocessor,
    FEATURES,
)

DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "clin_xai_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "evaluation_metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "classification_report.txt"
DATA_URLS = [
    "https://raw.githubusercontent.com/uciml/pima-indians-diabetes-database/master/diabetes.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
]

def try_download_dataset():
    if DATA_PATH.exists():
        return True
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in DATA_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ClinXAI-educational-project"})
            with urllib.request.urlopen(req, timeout=12) as response:
                content = response.read()
            DATA_PATH.write_bytes(content)
            # Confirm the file is parseable and has expected columns.
            df = pd.read_csv(DATA_PATH, nrows=5)
            required = set(FEATURES + ["Outcome"])
            if not required.issubset(df.columns):
                DATA_PATH.unlink(missing_ok=True)
                errors.append(f"{url}: downloaded CSV is missing required columns")
                continue
            return True
        except (OSError, ValueError, urllib.error.URLError) as exc:
            DATA_PATH.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
    if errors:
        print("Dataset download attempts failed:")
        for error in errors:
            print(f"- {error}")
    return False

def main():
    if not DATA_PATH.exists():
        print("Dataset not found locally. Trying to download a public copy...")
        if not try_download_dataset():
            raise SystemExit("Dataset download failed. See data/README.md and place diabetes.csv in data/.")
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = split_dataset(df)
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    try:
        auc = float(roc_auc_score(y_test, proba))
    except ValueError:
        auc = None
    metrics = {
        "dataset_rows": int(len(df)), "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)), "features": FEATURES,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": auc, "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "random_state": 42, "model_name": "Random Forest classifier",
        "dataset_file": "data/diabetes.csv",
        "note": "Educational benchmark only; not clinically validated."
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(classification_report(y_test, pred, zero_division=0), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

# ClinXAI — Explainable Clinical AI

ClinXAI is an academic Streamlit demonstration of a diabetes-label prediction pipeline and two explainability methods, LIME and SHAP. The interface displays predictions and evaluation values produced by the saved model and saved test evaluation; it does not use mock metrics or hardcoded predictions.

> **Responsible use:** This is an educational prototype, not a medical device. It is not a diagnosis or treatment tool. Do not enter identifiable patient data or use its output to make healthcare decisions.

## What is included

- Overview with the project workflow and saved evaluation KPIs
- Risk prediction using the trained scikit-learn pipeline
- Local LIME and SHAP explanations for the selected model class
- A LIME/SHAP comparison using the same sample and prediction
- Held-out accuracy, precision, recall, F1-score, confusion matrix, and classification report
- An About & Responsible AI page

The training pipeline uses a Random Forest classifier, median imputation, and standard scaling. Its expected input columns are:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`

The training data target is `Outcome` (0 or 1). Zero values in glucose, blood pressure, skin thickness, insulin, and BMI are treated as missing in the dataset and imputed during training. The prediction form requires positive values for those measurements.

## Windows PowerShell setup

Open PowerShell and run these commands from the project directory:

```powershell
Set-Location "$HOME\Downloads\ClinXAI_Project\ClinXAI"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.10 or 3.11 and replace `-3.11` with the installed version. If script activation is blocked in the current PowerShell session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Prepare the dataset and model

This project workspace does not include the dataset or generated model artifacts by default. From the project folder, run:

```powershell
python -m src.train_model
```

Training looks for `data\diabetes.csv`. If the file is absent, the explicit training command attempts to retrieve a public copy. The app itself does **not** download datasets or retrain on page navigation. If retrieval is unavailable, use an appropriately licensed dataset with the required columns and place it at:

```text
data\diabetes.csv
```

Training validates the CSV, creates a stratified 80/20 split, trains the model, and saves:

```text
models\clin_xai_pipeline.joblib
reports\evaluation_metrics.json
reports\classification_report.txt
```

The model, metrics, and report are generated from the local dataset by the training script. If those files are absent, the app explains what is missing and does not invent performance values or prediction outputs.

## Run ClinXAI

From the same project folder and with the virtual environment active:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Suggested live demonstration

1. **Overview:** explain the measurement → preprocessing → model → explanation workflow, show the four saved test metrics, and point out the educational-use disclaimer.
2. **Risk Prediction:** enter a sample measurement set, select **Predict risk**, and show the actual model class, available model-assigned class probability, and values submitted.
3. **LIME Explanation:** generate the local surrogate explanation for that prediction; discuss bars that raise or lower the selected class output.
4. **SHAP Explanation:** show the per-prediction SHAP contributions and optionally calculate global mean-absolute SHAP importance from held-out examples.
5. **LIME vs SHAP:** calculate both methods for the same saved input and contrast their approaches and limitations.
6. **Model Performance:** review the saved confusion matrix, classification report, and held-out metrics; emphasize that they do not establish clinical validity or generalization.
7. **About & Responsible AI:** close with the privacy, uncertainty, causality, and human-oversight limitations.

For best reliability, run `python -m src.train_model` before the presentation and verify that the generated model and report files exist.

## Tests

Run the test suite from the project folder:

```powershell
python -m pytest
```

Tests that require the project dataset or saved artifacts may skip when those files have not been generated. Unit tests for preprocessing and explainability can run without committing model files.

## Project layout

```text
app.py
src\
  data_preprocessing.py
  explainability.py
  train_model.py
data\
  README.md
tests\
reports\
```

## Model, data, and interpretation limitations

- A held-out metric is a measurement on one split of this dataset, not evidence that the model generalizes.
- Positive-class probabilities are model outputs and may not be calibrated.
- LIME and SHAP describe model behavior under their own assumptions; they do not prove causation or correctness.
- Dataset representation, fairness, and clinical suitability have not been established by this prototype.
- Sample input values are kept in Streamlit session state for the current browser session and are not intentionally written to a database.

## References

- Ribeiro, Singh, and Guestrin (2016), “Why Should I Trust You?” Explaining the Predictions of Any Classifier.
- Lundberg and Lee (2017), “A Unified Approach to Interpreting Model Predictions.”
- [scikit-learn](https://scikit-learn.org/)
- [LIME](https://lime-ml.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [Streamlit](https://docs.streamlit.io/)

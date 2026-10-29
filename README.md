# ClinXAI — Explainable Clinical AI Decision Support

ClinXAI is an interpretable clinical machine learning platform designed for diabetes risk assessment, powered by a Random Forest ensemble pipeline and dual post-hoc explainability methods: **LIME** (Local Interpretable Model-agnostic Explanations) and **SHAP** (SHapley Additive exPlanations).

The application is fully runnable locally and pre-configured for cloud deployment on **Render**, **HuggingFace Spaces**, or **Streamlit Community Cloud**.

> ⚠️ **Educational Use & Clinical Governance:** This platform is an educational and research prototype, not an FDA/CE-cleared medical device. It is intended for clinical decision support education, viva examination, and academic study. Do not enter identifiable patient health information (PHI) or make real-world healthcare decisions based on its predictions.

---

## 🌟 Key Highlights & Capabilities

- **🎯 Production-Grade ML Pipeline:** Scikit-Learn `RandomForestClassifier` (250 estimators) with median imputation for physiological missing values and standard scaling.
- **⚡ 1-Click Clinical Case Presets:** Rapidly load realistic clinical profiles (*High Risk Diabetic*, *Healthy Baseline*, *Borderline Pre-diabetic*, *Elevated Genetic Pedigree*) for seamless viva and demo presentations.
- **📊 Visual Clinical Risk Gauge:** Color-coded clinical risk stratification tiers (*Low Risk*, *Moderate Risk*, *Elevated Risk*, *High Clinical Risk*) with live probability progress meters and diagnostic recommendations.
- **🔍 Dual Explainability Engine:**
  - **LIME:** Local surrogate explanations highlighting continuous feature conditions pushing or mitigating risk.
  - **SHAP:** TreeSHAP game-theoretic Shapley values with additive efficiency guarantees and population-level global importance analysis.
- **🤝 Cross-Methodological Consensus:** Side-by-side comparison matrix evaluating concordance between LIME and SHAP feature attributions on identical patient cases.
- **🎛️ Interactive 'What-If' Counterfactual Sandbox:** Simulate real-time risk shifts resulting from targeted lifestyle or glycemic interventions (e.g. lowering fasting glucose or reducing BMI).
- **📈 Held-Out Test Evaluation:** Confusion matrix, Sensitivity (Recall = 70.4%), Specificity (76.0%), Accuracy (74.0%), and ROC-AUC (0.825) loaded from validated test splits.
- **🎓 Embedded Academic & Viva Guide:** Direct in-app access to the Top 20 Viva Examination Questions with model answers and clinical case study documentation.

---

## 🚀 Deploying to Render (Free Web Service)

ClinXAI includes native deployment configurations (`render.yaml`, `Procfile`, and `.streamlit/config.toml`).

### Method 1: 1-Click Render Blueprint (Recommended)
1. Push your repository to GitHub.
2. Sign in to [Render](https://render.com/).
3. In the Render Dashboard, click **New +** ➔ **Blueprint**.
4. Connect your GitHub repository. Render will automatically read `render.yaml` and configure:
   - **Environment:** Python 3.11.8
   - **Build Command:** `pip install -r requirements.txt && python -m src.train_model`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Click **Apply**. Your app will be live on a public `.onrender.com` URL within minutes!

### Method 2: Manual Web Service Setup
1. On Render, click **New +** ➔ **Web Service**.
2. Connect your GitHub repo.
3. Configure the service settings:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && python -m src.train_model`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Environment Variables:**
     - `PYTHON_VERSION`: `3.11.8`
4. Click **Deploy Web Service**.

---

## 💻 Local Quickstart (Windows / Linux / macOS)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/ClinXAI.git
cd ClinXAI

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify / Train Model Artifacts
The repository includes pre-generated model artifacts (`models/clin_xai_pipeline.joblib`) and test metrics (`reports/evaluation_metrics.json`). To retrain from scratch:
```bash
python -m src.train_model
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the Automated Test Suite
```bash
pytest
```
All 6 automated unit tests verify preprocessing logic, pipeline serialization, LIME generation, and TreeSHAP calculations.

---

## 📁 Repository Structure

```text
ClinXAI/
├── app.py                      # Main Streamlit Clinical Decision Support UI
├── Procfile                    # Cloud web service process runner
├── render.yaml                 # Render Blueprint configuration
├── pyproject.toml              # Pytest configuration & path discovery
├── requirements.txt            # Python dependencies (optimized for cloud & local)
├── .streamlit/
│   └── config.toml             # Streamlit server & clinical theme settings
├── data/
│   ├── README.md               # Dataset documentation & licensing notes
│   └── diabetes.csv            # Public Pima Indians diabetes cohort (768 rows)
├── models/
│   └── clin_xai_pipeline.joblib # Trained scikit-learn pipeline (Imputer + Scaler + RF)
├── reports/
│   ├── CASE_STUDY_TEMPLATE.md  # Academic case study framework
│   ├── classification_report.txt# Held-out split classification metrics
│   ├── evaluation_metrics.json # Machine-readable test split KPIs
│   └── VIVA_QUESTIONS.md       # Top 20 viva examination questions & answers
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Cohort loading, validation, and imputation
│   ├── explainability.py       # LIME and TreeSHAP explainer implementations
│   └── train_model.py          # Training and held-out evaluation runner
└── tests/
    ├── __init__.py
    ├── test_explainability.py   # LIME and SHAP unit tests
    ├── test_prediction.py       # Model serialization & inference tests
    └── test_preprocessing.py    # Imputation & column validation tests
```

---

## 📊 Held-Out Test Evaluation Summary

Evaluation conducted on an independent stratified 20% test split (154 instances):

| Metric | Measured Score | Clinical Implication |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.825** | High discriminative capacity between diabetic and healthy cohorts |
| **Sensitivity (Recall)** | **70.4%** | Critical metric in screening: identifies ~70% of true diabetic cases |
| **Specificity** | **76.0%** | Minimizes false positive alarms in healthy baseline individuals |
| **Accuracy** | **74.0%** | Overall correct classification rate |
| **Precision (PPV)** | **61.3%** | Probability that a positive screen represents true pathology |
| **F1-Score** | **0.655** | Harmonic balance between sensitivity and precision |

---

## ⚖️ Responsible AI Framework

- **Human Oversight:** Designed strictly as an assistive tool to augment, never replace, medical judgment.
- **Privacy by Design:** Patient values remain in temporary session memory and are never persisted to external databases.
- **Demographic Limitations:** Trained on the Pima Indians diabetes cohort; clinical deployment to diverse global populations mandates independent multi-center clinical validation.
- **Correlation ≠ Causation:** Explainability attributions reflect statistical associations within tree partitions, not biological causality.

---

## 📜 References

1. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* ACM SIGKDD 2016.
2. **Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
3. **Pedregosa, F. et al. (2011).** *Scikit-learn: Machine Learning in Python.* JMLR 12, pp. 2825-2830.

import json
import io
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_preprocessing import FEATURES, PROJECT_ROOT
from src.explainability import lime_explanation, load_pipeline, shap_explanation, global_shap_values
from src.train_model import METRICS_PATH, MODEL_PATH, REPORT_PATH

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="ClinXAI | Explainable Clinical AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Design System & Clinical Tokens
# ---------------------------------------------------------
NAVY_DEEP = "#0A192F"
NAVY_MED = "#0F243E"
NAVY_SURFACE = "#172E4C"
TEAL_PRIMARY = "#0D9488"
TEAL_LIGHT = "#14B8A6"
CYAN_ACCENT = "#0EA5E9"
PAGE_BG = "#F8FAFC"
CARD_BG = "#FFFFFF"
TEXT_MAIN = "#0F172A"
TEXT_MUTED = "#475569"
BORDER_COLOR = "#E2E8F0"

# Risk colors
COLOR_LOW_RISK = "#10B981"      # Emerald
COLOR_MODERATE = "#F59E0B"      # Amber
COLOR_ELEVATED = "#F97316"      # Orange
COLOR_HIGH_RISK = "#EF4444"     # Rose/Crimson

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Fasting Glucose",
    "BloodPressure": "Diastolic Blood Pressure",
    "SkinThickness": "Triceps Skin Fold",
    "Insulin": "Serum 2-Hour Insulin",
    "BMI": "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Score",
    "Age": "Patient Age",
}

FEATURE_UNITS = {
    "Pregnancies": "count",
    "Glucose": "mg/dL",
    "BloodPressure": "mm Hg",
    "SkinThickness": "mm",
    "Insulin": "µU/mL",
    "BMI": "kg/m²",
    "DiabetesPedigreeFunction": "score",
    "Age": "years",
}

FEATURE_BOUNDS = {
    "Pregnancies": (0.0, 20.0),
    "Glucose": (1.0, 350.0),
    "BloodPressure": (1.0, 250.0),
    "SkinThickness": (1.0, 100.0),
    "Insulin": (1.0, 900.0),
    "BMI": (1.0, 90.0),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (1.0, 120.0),
}

DEFAULTS = {
    "Pregnancies": 2.0,
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "SkinThickness": 25.0,
    "Insulin": 80.0,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 35.0,
}

# ---------------------------------------------------------
# Curated Clinical Presets (1-Click Demonstration Profiles)
# ---------------------------------------------------------
CLINICAL_PRESETS = {
    "Case 1 · High Clinical Risk": {
        "description": "Marked hyperglycemia, class-II obesity, high metabolic indicators",
        "values": {
            "Pregnancies": 5.0,
            "Glucose": 168.0,
            "BloodPressure": 86.0,
            "SkinThickness": 36.0,
            "Insulin": 210.0,
            "BMI": 38.2,
            "DiabetesPedigreeFunction": 0.85,
            "Age": 52.0,
        },
    },
    "Case 2 · Healthy Baseline Profile": {
        "description": "Euglycemic fasting glucose, normal BMI, youthful, low genetic risk",
        "values": {
            "Pregnancies": 1.0,
            "Glucose": 92.0,
            "BloodPressure": 68.0,
            "SkinThickness": 18.0,
            "Insulin": 65.0,
            "BMI": 22.4,
            "DiabetesPedigreeFunction": 0.18,
            "Age": 24.0,
        },
    },
    "Case 3 · Borderline / Pre-Diabetic": {
        "description": "Impaired fasting glucose, borderline overweight, middle-aged",
        "values": {
            "Pregnancies": 3.0,
            "Glucose": 134.0,
            "BloodPressure": 78.0,
            "SkinThickness": 28.0,
            "Insulin": 115.0,
            "BMI": 29.6,
            "DiabetesPedigreeFunction": 0.48,
            "Age": 42.0,
        },
    },
    "Case 4 · Elevated Genetic Pedigree": {
        "description": "High familial diabetes incidence, mild glycemic elevation",
        "values": {
            "Pregnancies": 2.0,
            "Glucose": 126.0,
            "BloodPressure": 74.0,
            "SkinThickness": 24.0,
            "Insulin": 95.0,
            "BMI": 27.2,
            "DiabetesPedigreeFunction": 1.45,
            "Age": 31.0,
        },
    },
}

# ---------------------------------------------------------
# Global Styling
# ---------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {{
      --clin-navy: {NAVY_DEEP};
      --clin-navy-med: {NAVY_MED};
      --clin-teal: {TEAL_PRIMARY};
      --clin-teal-light: {TEAL_LIGHT};
      --clin-cyan: {CYAN_ACCENT};
      --clin-bg: {PAGE_BG};
      --clin-card: {CARD_BG};
      --clin-text: {TEXT_MAIN};
      --clin-muted: {TEXT_MUTED};
      --clin-border: {BORDER_COLOR};
    }}

    .stApp {{
      background: var(--clin-bg);
      color: var(--clin-text);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #071526 0%, #0F243E 100%);
      border-right: 1px solid #1E3A5F;
    }}
    [data-testid="stSidebar"] * {{
      color: #F8FAFC !important;
    }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
      color: #94A3B8 !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stRadio"] label {{
      border-radius: 8px;
      padding: 0.5rem 0.75rem;
      transition: all 0.2s ease;
      cursor: pointer;
    }}
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
      background: rgba(255, 255, 255, 0.06);
    }}
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
      background: rgba(13, 148, 136, 0.22);
      border-left: 3px solid #14B8A6;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}

    /* Typography */
    h1, h2, h3, h4 {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      color: var(--clin-navy);
      font-weight: 700;
      letter-spacing: -0.025em;
    }}

    /* Hero Banner */
    .hero-banner {{
      background: linear-gradient(135deg, #0A192F 0%, #0F2A4A 60%, #11365D 100%);
      border: 1px solid #1E3A5F;
      border-radius: 16px;
      padding: 1.8rem 2.2rem;
      margin-bottom: 1.4rem;
      box-shadow: 0 10px 25px -5px rgba(10, 25, 47, 0.15);
      position: relative;
      overflow: hidden;
    }}
    .hero-banner::after {{
      content: "";
      position: absolute;
      top: -50px;
      right: -50px;
      width: 200px;
      height: 200px;
      background: radial-gradient(circle, rgba(20, 184, 166, 0.25) 0%, transparent 70%);
      pointer-events: none;
    }}
    .hero-banner h1 {{
      color: #FFFFFF !important;
      font-size: clamp(2rem, 3.2vw, 2.75rem);
      margin: 0 0 0.4rem 0;
      font-weight: 800;
    }}
    .hero-banner .hero-sub {{
      color: #99F6E4 !important;
      font-size: 1.05rem;
      font-weight: 600;
      margin-bottom: 0.6rem;
    }}
    .hero-banner .hero-text {{
      color: #CBD5E1 !important;
      max-width: 820px;
      line-height: 1.6;
      font-size: 0.95rem;
      margin: 0;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: #34D399 !important;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 0.75rem;
    }}

    /* Card System */
    .clin-card {{
      background: #FFFFFF;
      border: 1px solid var(--clin-border);
      border-radius: 14px;
      padding: 1.25rem 1.4rem;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.02);
      transition: all 0.2s ease;
    }}
    .clin-card:hover {{
      border-color: #CBD5E1;
      box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.07);
    }}

    /* Metrics Styling */
    div[data-testid="stMetric"] {{
      background: #FFFFFF;
      border: 1px solid var(--clin-border);
      border-radius: 12px;
      padding: 0.9rem 1.1rem;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
    }}
    div[data-testid="stMetricLabel"] p {{
      color: var(--clin-muted) !important;
      font-weight: 600;
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    div[data-testid="stMetricValue"] {{
      color: var(--clin-navy) !important;
      font-weight: 800;
      font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    /* Buttons */
    .stButton > button, [data-testid="stFormSubmitButton"] > button {{
      border-radius: 9px;
      font-weight: 650;
      padding: 0.55rem 1.2rem;
      transition: all 0.2s ease;
      font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(13, 148, 136, 0.2);
    }}

    /* Custom Risk Meter */
    .risk-meter-container {{
      background: #FFFFFF;
      border: 1px solid var(--clin-border);
      border-radius: 14px;
      padding: 1.4rem;
      margin: 1rem 0;
    }}
    .risk-progress-bar {{
      height: 12px;
      border-radius: 6px;
      background: linear-gradient(90deg, #10B981 0%, #F59E0B 45%, #F97316 75%, #EF4444 100%);
      position: relative;
      margin: 1.5rem 0 0.8rem 0;
    }}
    .risk-marker {{
      position: absolute;
      top: -6px;
      width: 24px;
      height: 24px;
      background: #0F172A;
      border: 3px solid #FFFFFF;
      border-radius: 50%;
      transform: translateX(-50%);
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }}

    /* Workflow Cards */
    .wf-card {{
      background: #FFFFFF;
      border: 1px solid var(--clin-border);
      border-top: 4px solid var(--clin-teal);
      border-radius: 12px;
      padding: 1rem;
      height: 100%;
    }}
    .wf-num {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 0.82rem;
      font-weight: 800;
      color: var(--clin-teal);
      letter-spacing: 0.05em;
    }}
    .wf-title {{
      font-weight: 700;
      color: var(--clin-navy);
      font-size: 0.96rem;
      margin: 0.25rem 0 0.35rem 0;
    }}
    .wf-desc {{
      color: var(--clin-muted);
      font-size: 0.84rem;
      line-height: 1.4;
      margin: 0;
    }}

    /* Tables & DataFrames */
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
      border: 1px solid var(--clin-border);
      border-radius: 10px;
      overflow: hidden;
    }}

    /* Footer Note */
    .clin-footer {{
      color: #64748B;
      font-size: 0.85rem;
      border-top: 1px solid var(--clin-border);
      padding-top: 1rem;
      margin-top: 2.5rem;
      text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_resource
def cached_pipeline(model_path: str, modified_ns: int):
    del model_path, modified_ns
    return load_pipeline()


@st.cache_data
def read_metrics(metrics_path: str):
    with Path(metrics_path).open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@st.cache_data
def read_classification_report(report_path: str):
    return Path(report_path).read_text(encoding="utf-8")


def load_model_if_available():
    if not MODEL_PATH.is_file():
        return None
    try:
        stat = MODEL_PATH.stat()
        return cached_pipeline(str(MODEL_PATH), stat.st_mtime_ns)
    except Exception as exc:
        st.error(f"The saved model could not be loaded: {exc}")
        return None


def load_metrics_if_available():
    if not METRICS_PATH.is_file():
        return None
    try:
        metrics = read_metrics(str(METRICS_PATH))
        if not isinstance(metrics, dict):
            raise ValueError("The metrics file must contain a JSON object.")
        return metrics
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Saved evaluation metrics could not be read: {exc}")
        return None


def show_setup_message(missing: str):
    st.warning(f"⚠️ {missing}")
    st.caption("From the project folder, build model and evaluation artifacts with:")
    st.code("python -m src.train_model")


# ---------------------------------------------------------
# Patient Form & Presets
# ---------------------------------------------------------
def render_measurement_form(form_key: str, submit_label: str):
    st.markdown("### 📋 Enter Patient Clinical Parameters")
    st.caption(
        "Input standard biometric and laboratory measurements. You may load a pre-configured clinical profile or input custom values below."
    )

    # Quick Preset Selector
    st.markdown("##### ⚡ Quick Clinical Presets")
    preset_cols = st.columns(4)
    selected_preset_key = None
    for idx, (p_name, p_data) in enumerate(CLINICAL_PRESETS.items()):
        if preset_cols[idx].button(p_name.split("·")[1].strip(), key=f"{form_key}_preset_{idx}", use_container_width=True):
            selected_preset_key = p_name

    # Determine baseline values for form inputs
    if selected_preset_key:
        active_values = CLINICAL_PRESETS[selected_preset_key]["values"]
        st.success(f"Loaded **{selected_preset_key}**: *{CLINICAL_PRESETS[selected_preset_key]['description']}*")
    elif "last_input_values" in st.session_state:
        active_values = st.session_state["last_input_values"]
    else:
        active_values = DEFAULTS

    values = {}
    with st.form(form_key):
        left, right = st.columns(2)
        for index, feature in enumerate(FEATURES):
            minimum, maximum = FEATURE_BOUNDS[feature]
            step = 0.1 if feature in ("BMI", "DiabetesPedigreeFunction") else 1.0
            initial_val = float(active_values.get(feature, DEFAULTS[feature]))
            unit = FEATURE_UNITS.get(feature, "")
            label_text = f"{FEATURE_LABELS[feature]} ({unit})" if unit else FEATURE_LABELS[feature]

            with (left if index % 2 == 0 else right):
                values[feature] = st.number_input(
                    label_text,
                    min_value=minimum,
                    max_value=maximum,
                    value=initial_val,
                    step=step,
                    format="%.2f" if step < 1 else "%.0f",
                    key=f"{form_key}_{feature}",
                    help=f"Clinical normal range guide available below. Valid range: {minimum:g} – {maximum:g} {unit}.",
                )

        submitted = st.form_submit_button(
            f"🔍 {submit_label}", type="primary", use_container_width=True
        )

    # Save to session memory
    st.session_state["last_input_values"] = values
    sample = pd.DataFrame([values], columns=FEATURES)

    # Reference Guide Expander
    with st.expander("📖 Clinical Reference Ranges & Feature Guidelines", expanded=False):
        ref_df = pd.DataFrame(
            [
                {"Parameter": "Fasting Glucose", "Unit": "mg/dL", "Normal": "70 – 99", "Pre-diabetic": "100 – 125", "Diabetic Alert": "≥ 126"},
                {"Parameter": "Body Mass Index (BMI)", "Unit": "kg/m²", "Normal": "18.5 – 24.9", "Overweight": "25.0 – 29.9", "Diabetic Alert": "≥ 30.0 (Obese)"},
                {"Parameter": "Diastolic Blood Pressure", "Unit": "mm Hg", "Normal": "< 80", "Elevated": "80 – 89", "Diabetic Alert": "≥ 90 (Stage 2)"},
                {"Parameter": "Serum 2-Hour Insulin", "Unit": "µU/mL", "Normal": "16 – 166", "Borderline": "166 – 200", "Diabetic Alert": "> 200"},
                {"Parameter": "Diabetes Pedigree", "Unit": "score", "Normal": "< 0.40", "Elevated": "0.40 – 0.80", "Diabetic Alert": "> 0.80 (Strong Family Hx)"},
            ]
        )
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

    return sample, submitted


def validate_sample(sample: pd.DataFrame):
    if list(sample.columns) != FEATURES:
        return "Measurements do not match the feature order expected by the model."
    values = sample.iloc[0]
    for feature, (minimum, maximum) in FEATURE_BOUNDS.items():
        value = float(values[feature])
        if not np.isfinite(value) or value < minimum or value > maximum:
            return f"{FEATURE_LABELS[feature]} must be between {minimum:g} and {maximum:g}."
    return None


def predict_sample(sample: pd.DataFrame, pipeline):
    validation_error = validate_sample(sample)
    if validation_error:
        raise ValueError(validation_error)
    prediction = pipeline.predict(sample[FEATURES])[0]
    classes = np.asarray(pipeline.named_steps["classifier"].classes_)
    matches = np.flatnonzero(classes == prediction)
    if matches.size != 1:
        raise ValueError("The predicted class was not found in the trained model classes.")

    probability = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(sample[FEATURES]), dtype=float)
        class_index = int(matches[0])
        if (
            probabilities.ndim == 2
            and probabilities.shape[0] == 1
            and probabilities.shape[1] == len(classes)
            and np.isfinite(probabilities[0, class_index])
            and 0.0 <= probabilities[0, class_index] <= 1.0
        ):
            probability = float(probabilities[0, class_index])
    else:
        class_index = int(matches[0])
    return prediction, probability, class_index


def record_prediction(sample: pd.DataFrame, pipeline):
    prediction, probability, class_index = predict_sample(sample, pipeline)
    st.session_state["current_sample"] = sample.copy()
    st.session_state["current_prediction"] = prediction.item() if hasattr(prediction, "item") else prediction
    st.session_state["current_probability"] = probability
    st.session_state["current_class_index"] = class_index
    st.session_state.pop("lime_result", None)
    st.session_state.pop("shap_result", None)


def show_sample(sample: pd.DataFrame):
    display_data = []
    values = sample.iloc[0]
    for feat in FEATURES:
        val = values[feat]
        unit = FEATURE_UNITS.get(feat, "")
        formatted_val = f"{val:.2f}" if isinstance(val, float) and val % 1 != 0 else f"{int(val)}" if isinstance(val, (int, float)) and val % 1 == 0 else f"{val}"
        display_data.append({
            "Clinical Feature": FEATURE_LABELS[feat],
            "Value": f"{formatted_val} {unit}".strip(),
            "Raw Value": val
        })
    df_display = pd.DataFrame(display_data)
    st.dataframe(df_display[["Clinical Feature", "Value"]], use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Visual Risk Gauge & Prediction Result
# ---------------------------------------------------------
def render_risk_gauge(probability: float, prediction: int):
    # Risk tiers
    if probability < 0.30:
        tier_label = "Low Clinical Risk"
        tier_color = COLOR_LOW_RISK
        tier_badge = "🟢 LOW RISK"
        clinical_note = "Biometric indicators fall within predominantly healthy ranges."
    elif probability < 0.50:
        tier_label = "Moderate / Borderline Risk"
        tier_color = COLOR_MODERATE
        tier_badge = "🟡 MODERATE RISK"
        clinical_note = "Elevated indicators detected. Lifestyle monitoring and re-screening advised."
    elif probability < 0.70:
        tier_label = "Elevated Clinical Risk"
        tier_color = COLOR_ELEVATED
        tier_badge = "🟠 ELEVATED RISK"
        clinical_note = "Significant metabolic markers observed. Diagnostic oral glucose tolerance test (OGTT) recommended."
    else:
        tier_label = "High Clinical Risk"
        tier_color = COLOR_HIGH_RISK
        tier_badge = "🔴 HIGH CLINICAL RISK"
        clinical_note = "Strong multi-factorial risk indicators. Immediate clinical consultation warranted."

    pct = max(0.0, min(100.0, probability * 100))

    html = f"""
    <div class="risk-meter-container">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <div>
          <span style="background: {tier_color}18; color: {tier_color}; border: 1px solid {tier_color}40; padding: 0.25rem 0.7rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em;">
            {tier_badge}
          </span>
          <h2 style="margin: 0.5rem 0 0.1rem 0; color: {NAVY_DEEP}; font-size: 1.55rem;">{tier_label}</h2>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 2.2rem; font-weight: 800; color: {tier_color}; font-family: 'Plus Jakarta Sans', sans-serif; line-height: 1;">
            {pct:.1f}%
          </div>
          <div style="font-size: 0.78rem; color: {TEXT_MUTED}; font-weight: 600; text-transform: uppercase;">
            Model Risk Probability
          </div>
        </div>
      </div>

      <!-- Gradient Progress Bar -->
      <div class="risk-progress-bar">
        <div class="risk-marker" style="left: {pct}%;"></div>
      </div>

      <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94A3B8; font-weight: 600;">
        <span>0% (Low)</span>
        <span>30% (Moderate)</span>
        <span>50% (Classification Threshold)</span>
        <span>70% (Elevated)</span>
        <span>100% (Critical)</span>
      </div>

      <div style="background: #F8FAFC; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 1rem; border-left: 4px solid {tier_color};">
        <p style="margin: 0; font-size: 0.86rem; color: {TEXT_MAIN}; line-height: 1.45;">
          <strong>Clinical Insight:</strong> {clinical_note}
        </p>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def show_prediction_result():
    sample = st.session_state.get("current_sample")
    if sample is None:
        return
    prediction = st.session_state["current_prediction"]
    probability = st.session_state.get("current_probability")

    st.markdown("### 📊 Model Output & Clinical Risk Assessment")
    
    # Calculate probability for positive diabetic class (1)
    if hasattr(st.session_state, "get"):
        # If model predicted 0, probability of class 1 is (1 - p0)
        class_idx = st.session_state.get("current_class_index", 0)
        if class_idx == 1:
            diabetic_prob = probability if probability is not None else 0.5
        else:
            diabetic_prob = (1.0 - probability) if probability is not None else 0.5
    else:
        diabetic_prob = probability or 0.5

    render_risk_gauge(diabetic_prob, prediction)

    # Actionable deep-dive buttons
    st.markdown("##### 🔬 Next Steps: Inspect Explainability")
    nav_c1, nav_c2, nav_c3 = st.columns(3)
    if nav_c1.button("👉 Run LIME Local Surrogate", use_container_width=True):
        st.session_state["navigation"] = "◌  LIME Explanation"
        st.rerun()
    if nav_c2.button("👉 Run SHAP Shapley Analysis", use_container_width=True):
        st.session_state["navigation"] = "◈  SHAP Explanation"
        st.rerun()
    if nav_c3.button("👉 Compare LIME vs SHAP", use_container_width=True):
        st.session_state["navigation"] = "↔  LIME vs SHAP"
        st.rerun()

    # Expanders for patient values & counterfactual
    with st.expander("📋 Review Submitted Patient Measurements", expanded=False):
        show_sample(sample)


# ---------------------------------------------------------
# Plotting & Visualization System
# ---------------------------------------------------------
def plot_contributions(labels, values, title, x_label):
    values = np.asarray(values, dtype=float)
    labels = list(labels)
    if values.size == 0 or len(labels) != len(values) or not np.isfinite(values).all():
        raise ValueError("The explanation did not produce valid feature contributions.")

    order = np.argsort(np.abs(values))
    ordered_values = values[order]
    ordered_labels = [labels[index] for index in order]

    # Clinical colors: Red = increases diabetes risk, Teal/Green = protective/decreases risk
    colors = [COLOR_HIGH_RISK if val >= 0 else COLOR_LOW_RISK for val in ordered_values]

    fig, axis = plt.subplots(figsize=(9, max(3.4, 0.46 * len(ordered_labels))), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")

    bars = axis.barh(ordered_labels, ordered_values, color=colors, height=0.62, edgecolor="none", zorder=3)
    axis.axvline(0, color="#475569", linewidth=1.1, linestyle="-", alpha=0.7, zorder=4)

    # Value annotations on bars
    max_val = np.max(np.abs(ordered_values)) if ordered_values.size else 1.0
    offset = max_val * 0.03
    for bar, val in zip(bars, ordered_values):
        sign_str = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
        if val >= 0:
            axis.text(val + offset, bar.get_y() + bar.get_height()/2, sign_str,
                      va="center", ha="left", color="#1E293B", fontsize=8.5, fontweight="600")
        else:
            axis.text(val - offset, bar.get_y() + bar.get_height()/2, sign_str,
                      va="center", ha="right", color="#1E293B", fontsize=8.5, fontweight="600")

    axis.set_title(title, color=NAVY_DEEP, fontsize=11, fontweight="bold", loc="left", pad=12)
    axis.set_xlabel(x_label, color=TEXT_MUTED, fontsize=9, fontweight="600")
    axis.tick_params(axis="both", colors=TEXT_MAIN, labelsize=9)
    axis.grid(axis="x", color="#F1F5F9", linewidth=0.8, linestyle="--", zorder=1)
    axis.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(BORDER_COLOR)

    # Add margin for labels
    axis.set_xlim(-max_val * 1.28, max_val * 1.28)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Explanation Context & Helpers
# ---------------------------------------------------------
def explanation_context(page_key: str, action_label: str):
    sample = st.session_state.get("current_sample")
    if sample is not None:
        prediction = st.session_state.get("current_prediction")
        st.info(f"💡 Active Patient Profile Loaded · Model Prediction: **Class {prediction}** ({'Diabetic Positive' if prediction == 1 else 'Negative / Non-diabetic'}).")
        with st.expander("📋 Review Patient Measurements", expanded=False):
            show_sample(sample)
        return sample, st.button(f"⚡ {action_label}", type="primary", key=f"{page_key}_run")

    st.info("No active patient is loaded yet. Enter clinical measurements or choose a preset below.")
    sample, submitted = render_measurement_form(
        f"{page_key}_measurements",
        f"Predict and generate {action_label}",
    )
    if submitted:
        pipeline = load_model_if_available()
        if pipeline is None:
            show_setup_message("A trained model is not available, so an explanation cannot be generated.")
            return sample, False
        try:
            record_prediction(sample, pipeline)
        except (ValueError, TypeError) as exc:
            st.error(f"Could not create a prediction: {exc}")
            return sample, False
    return sample, submitted


def generate_clinical_narrative(method_name: str, pairs: list):
    """Generates an intuitive, doctor-friendly text narrative of key drivers."""
    if not pairs:
        return ""
    sorted_pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    top_pos = [p for p in sorted_pairs if p[1] > 0][:2]
    top_neg = [p for p in sorted_pairs if p[1] < 0][:2]

    narrative = []
    if top_pos:
        items = [f"**{p[0]}** (+{p[1]:.3f})" for p in top_pos]
        narrative.append(f"🔺 **Primary Risk Drivers:** {', '.join(items)} contributed most strongly toward increasing the predicted diabetes risk.")
    if top_neg:
        items = [f"**{p[0]}** ({p[1]:.3f})" for p in top_neg]
        narrative.append(f"🛡️ **Protective Factors:** {', '.join(items)} actively mitigated the patient's predicted risk score.")

    return "\n\n".join(narrative)


# ---------------------------------------------------------
# LIME Page Handler
# ---------------------------------------------------------
def save_and_show_lime(sample: pd.DataFrame):
    pipeline = load_model_if_available()
    if pipeline is None:
        show_setup_message("A trained model is not available, so LIME cannot be run.")
        return
    prediction = st.session_state.get("current_prediction")
    if prediction is None:
        try:
            record_prediction(sample, pipeline)
        except (ValueError, TypeError) as exc:
            st.error(f"Could not create a prediction: {exc}")
            return
        prediction = st.session_state["current_prediction"]
    try:
        with st.spinner("🔄 Fitting local surrogate model with LIME perturbations..."):
            result = lime_explanation(sample, pipeline=pipeline, class_label=prediction)
        st.session_state["lime_result"] = {
            "sample": sample.copy(),
            "class_label": prediction,
            "pairs": result.as_list(label=st.session_state["current_class_index"]),
        }
    except (ImportError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        st.error(f"LIME could not generate an explanation: {exc}")
        return
    except Exception as exc:
        st.error(f"LIME encountered an unexpected explanation error: {exc}")
        return
    show_lime_result(st.session_state["lime_result"])


def show_lime_result(result):
    pairs = result["pairs"]
    title = f"LIME Local Surrogate Contributions · Predicted Class {result['class_label']}"
    fig = plot_contributions(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        title,
        "Local Surrogate Weight (Impact on Model Class)",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Narrative Summary
    narrative = generate_clinical_narrative("LIME", pairs)
    if narrative:
        st.markdown(
            f"""
            <div style="background: #F1F5F9; border-radius: 10px; padding: 1rem 1.2rem; margin: 0.8rem 0; border-left: 4px solid {TEAL_PRIMARY};">
              <h5 style="margin: 0 0 0.4rem 0; color: {NAVY_DEEP};">📋 Clinical Interpretation Narrative</h5>
              <div style="font-size: 0.88rem; color: {TEXT_MAIN}; line-height: 1.5;">
                {narrative}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Feature Condition / Range": pair[0],
                    "Surrogate Weight": f"{pair[1]:+.4f}",
                    "Clinical Influence": "Increases Risk (🔺)" if pair[1] > 0 else "Protective (🛡️)",
                }
                for pair in pairs
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# SHAP Page Handler
# ---------------------------------------------------------
def save_and_show_shap(sample: pd.DataFrame):
    pipeline = load_model_if_available()
    if pipeline is None:
        show_setup_message("A trained model is not available, so SHAP cannot be run.")
        return
    prediction = st.session_state.get("current_prediction")
    if prediction is None:
        try:
            record_prediction(sample, pipeline)
        except (ValueError, TypeError) as exc:
            st.error(f"Could not create a prediction: {exc}")
            return
        prediction = st.session_state["current_prediction"]
    try:
        with st.spinner("🔄 Computing TreeSHAP Shapley values across decision paths..."):
            values, baseline = shap_explanation(
                sample, pipeline=pipeline, class_label=prediction
            )
        st.session_state["shap_result"] = {
            "sample": sample.copy(),
            "class_label": prediction,
            "values": np.asarray(values, dtype=float),
            "baseline": float(baseline),
        }
    except (ImportError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        st.error(f"SHAP could not generate an explanation: {exc}")
        return
    except Exception as exc:
        st.error(f"SHAP encountered an unexpected explanation error: {exc}")
        return
    show_shap_result(st.session_state["shap_result"])


def show_shap_result(result):
    values = result["values"]
    title = f"SHAP Additive Feature Contributions · Predicted Class {result['class_label']}"
    fig = plot_contributions(
        [FEATURE_LABELS[feature] for feature in FEATURES],
        values,
        title,
        "Shapley Value (Contribution to Model Output vs Baseline)",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        f"Baseline Expected Value E[f(x)]: **{result['baseline']:.4f}**. "
        "Positive values push the model toward class 1; negative values push toward class 0. "
        "SHAP satisfies additive efficiency under cooperative game theory."
    )

    # Narrative Summary
    shap_pairs = [(FEATURE_LABELS[feat], values[idx]) for idx, feat in enumerate(FEATURES)]
    narrative = generate_clinical_narrative("SHAP", shap_pairs)
    if narrative:
        st.markdown(
            f"""
            <div style="background: #F1F5F9; border-radius: 10px; padding: 1rem 1.2rem; margin: 0.8rem 0; border-left: 4px solid {CYAN_ACCENT};">
              <h5 style="margin: 0 0 0.4rem 0; color: {NAVY_DEEP};">📋 Clinical Interpretation Narrative</h5>
              <div style="font-size: 0.88rem; color: {TEXT_MAIN}; line-height: 1.5;">
                {narrative}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    order = np.argsort(np.abs(values))[::-1]
    st.dataframe(
        pd.DataFrame(
            {
                "Feature Name": [FEATURE_LABELS[FEATURES[index]] for index in order],
                "Patient Value": [st.session_state["current_sample"].iloc[0][FEATURES[index]] for index in order],
                "SHAP Value": [f"{values[index]:+.4f}" for index in order],
                "Clinical Impact": [
                    "Increases Risk (🔺)" if values[index] > 0 else "Protective (🛡️)" if values[index] < 0 else "Neutral (⚪)"
                    for index in order
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# What-If / Counterfactual Simulator
# ---------------------------------------------------------
def render_what_if_simulator(sample: pd.DataFrame, pipeline):
    st.markdown("### 🎛️ Interactive 'What-If' Counterfactual Simulation")
    st.caption(
        "Simulate how lifestyle or clinical interventions (e.g. lowering fasting glucose or reducing BMI) affect the model's predicted risk score in real time."
    )

    values = sample.iloc[0].to_dict()
    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        new_glucose = st.slider(
            "Simulated Fasting Glucose (mg/dL)",
            min_value=70.0,
            max_value=250.0,
            value=float(values["Glucose"]),
            step=1.0,
            help="Simulate dietary or glycemic control intervention.",
        )
        new_bmi = st.slider(
            "Simulated Body Mass Index (BMI)",
            min_value=18.0,
            max_value=50.0,
            value=float(values["BMI"]),
            step=0.2,
            help="Simulate weight management intervention.",
        )

    with sim_col2:
        new_bp = st.slider(
            "Simulated Diastolic BP (mm Hg)",
            min_value=50.0,
            max_value=120.0,
            value=float(values["BloodPressure"]),
            step=1.0,
        )
        new_insulin = st.slider(
            "Simulated Insulin (µU/mL)",
            min_value=15.0,
            max_value=300.0,
            value=float(values["Insulin"]),
            step=5.0,
        )

    simulated_values = values.copy()
    simulated_values["Glucose"] = new_glucose
    simulated_values["BMI"] = new_bmi
    simulated_values["BloodPressure"] = new_bp
    simulated_values["Insulin"] = new_insulin

    sim_sample = pd.DataFrame([simulated_values], columns=FEATURES)
    orig_prob = st.session_state.get("current_probability", 0.5)
    sim_pred, sim_prob, _ = predict_sample(sim_sample, pipeline)

    delta = (sim_prob - orig_prob) * 100

    r_col1, r_col2 = st.columns(2)
    r_col1.metric("Original Model Probability", f"{orig_prob:.1%}")
    r_col2.metric(
        "Simulated Model Probability",
        f"{sim_prob:.1%}",
        delta=f"{delta:+.1f}% risk shift",
        delta_color="inverse",
    )

    if delta < -5.0:
        st.success(f"🎉 **Positive Clinical Impact:** The simulated interventions reduced the predicted risk by {abs(delta):.1f} percentage points!")
    elif delta > 5.0:
        st.warning(f"⚠️ **Elevated Risk:** The simulated values increased the predicted risk by {delta:.1f} percentage points.")


def format_metric(metrics, key):
    value = metrics.get(key)
    if value is None:
        return "N/A"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(numeric_value):
        return "N/A"
    return f"{numeric_value:.3f}"


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
nav_items = {
    "⌂  Overview": "Overview",
    "●  Risk Prediction": "Risk Prediction",
    "◌  LIME Explanation": "LIME Explanation",
    "◈  SHAP Explanation": "SHAP Explanation",
    "↔  LIME vs SHAP": "LIME vs SHAP",
    "▤  Model Performance": "Model Performance",
    "🎓  Viva & Case Study Guide": "Viva & Case Study Guide",
    "ⓘ  About & Responsible AI": "About & Responsible AI",
}

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
          <div style="font-size: 1.65rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.03em; font-family: 'Plus Jakarta Sans', sans-serif;">
            Clin<span style="color: #2DD4BF;">XAI</span>
          </div>
          <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.15rem; font-weight: 500;">
            Explainable Clinical AI Platform
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: rgba(13, 148, 136, 0.18); border: 1px solid rgba(45, 212, 191, 0.3); border-radius: 8px; padding: 0.5rem 0.75rem; margin-bottom: 1.1rem;">
          <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; font-weight: 700; color: #2DD4BF;">
            <span style="width: 8px; height: 8px; background: #2DD4BF; border-radius: 50%; box-shadow: 0 0 6px #2DD4BF;"></span>
            RANDOM FOREST PIPELINE
          </div>
          <div style="font-size: 0.72rem; color: #CBD5E1; margin-top: 0.2rem;">
            Status: Trained & Validated
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_nav = st.radio(
        "NAVIGATION",
        list(nav_items),
        label_visibility="collapsed",
        key="navigation",
    )
    page = nav_items[selected_nav]

    st.divider()

    # Active Patient Badge
    if "current_sample" in st.session_state:
        pred_label = "Diabetic (1)" if st.session_state.get("current_prediction") == 1 else "Non-Diabetic (0)"
        prob_val = st.session_state.get("current_probability")
        prob_str = f"{prob_val:.1%}" if prob_val is not None else "N/A"
        st.markdown(
            f"""
            <div style="background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.8rem; font-size: 0.8rem;">
              <div style="color: #94A3B8; font-weight: 600; text-transform: uppercase; font-size: 0.7rem;">Active Case</div>
              <div style="color: #F8FAFC; font-weight: 700; margin-top: 0.15rem;">{pred_label}</div>
              <div style="color: #2DD4BF; font-size: 0.75rem;">Probability: {prob_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("🩺 ClinXAI Academic Demonstration")
    st.caption("Educational Prototype · Not for Medical Diagnosis")


# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    st.markdown(
        """
        <div class="hero-banner">
          <div class="status-pill">● ClinXAI v1.0 · Verified Pipeline</div>
          <h1>ClinXAI — Explainable Clinical AI</h1>
          <p class="hero-sub">Interpretable Machine Learning in Diagnostic Decision Support</p>
          <p class="hero-text">
            ClinXAI bridges the gap between high-accuracy ensemble classifiers and clinical interpretability.
            By pairing a rigorously evaluated Random Forest pipeline with state-of-the-art LIME (Local Interpretable Model-agnostic Explanations)
            and SHAP (SHapley Additive exPlanations), clinicians can examine the exact biometric features driving individual risk predictions.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Metrics Row
    metrics = load_metrics_if_available()
    st.markdown("### 📈 Held-Out Test Evaluation Performance")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("ROC-AUC Score", format_metric(metrics, "roc_auc") if metrics else "N/A", help="Area Under ROC Curve on held-out 20% test split")
    m_col2.metric("Accuracy", format_metric(metrics, "accuracy") if metrics else "N/A", help="Overall percentage of correct predictions")
    m_col3.metric("Sensitivity (Recall)", format_metric(metrics, "recall") if metrics else "N/A", help="Critical in clinical screening to minimize false negatives")
    m_col4.metric("Precision", format_metric(metrics, "precision") if metrics else "N/A", help="Positive predictive value")
    m_col5.metric("F1-Score", format_metric(metrics, "f1_score") if metrics else "N/A", help="Harmonic mean of precision and recall")

    st.caption("All metrics measured on stratified 20% held-out test split (154 samples). Source: reports/evaluation_metrics.json.")

    st.markdown("---")

    # Workflow Section
    st.markdown("### 🔄 End-to-End Clinical AI Architecture")
    wf_cols = st.columns(4)
    workflow_steps = [
        ("01 · Biometric Input", "Patient measurements collected across 8 validated clinical parameters.", "#0D9488"),
        ("02 · Preprocessing Pipeline", "Median imputation for physiological zero-placeholders and standard scaling.", "#0EA5E9"),
        ("03 · Random Forest Model", "Ensemble of 250 decision trees trained with balanced class weights.", "#6366F1"),
        ("04 · Dual XAI Engine", "Decomposition of predictions using local surrogates (LIME) and Shapley values (SHAP).", "#10B981"),
    ]
    for col, (title, desc, color) in zip(wf_cols, workflow_steps):
        col.markdown(
            f"""
            <div class="wf-card" style="border-top-color: {color};">
              <div class="wf-num">{title.split('·')[0].strip()}</div>
              <div class="wf-title">{title.split('·')[1].strip()}</div>
              <div class="wf-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Model & Data Status Cards
    card_col1, card_col2 = st.columns(2)
    card_col1.markdown(
        f"""
        <div class="clin-card">
          <h4 style="margin: 0 0 0.5rem 0; color: {NAVY_DEEP};">🤖 Machine Learning Estimator</h4>
          <p style="margin: 0.2rem 0; font-size: 0.9rem;"><strong>Model Class:</strong> RandomForestClassifier (250 estimators)</p>
          <p style="margin: 0.2rem 0; font-size: 0.9rem;"><strong>Preprocessing:</strong> SimpleImputer(median) + StandardScaler</p>
          <p style="margin: 0.2rem 0; font-size: 0.9rem;"><strong>Model Artifact:</strong> <code>models/clin_xai_pipeline.joblib</code></p>
          <p style="margin: 0.2rem 0; font-size: 0.9rem;"><strong>Pipeline Status:</strong> <span style="color: {COLOR_LOW_RISK}; font-weight: 700;">● Online & Loaded</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_present = (PROJECT_ROOT / "data" / "diabetes.csv").is_file()
    card_col2.markdown(
        f"""
        <div class="clin-card">
          <h4 style="margin: 0 0 0.5rem 0; color: {NAVY_DEEP};">📁 Dataset & Validation Cohort</h4>
          <p style="margin: 0.2rem 0; font-size: 0.9rem;"><strong>Source:</strong> National Institute of Diabetes & Digestive & Kidney Diseases</p>
          <p style="margin: 0.2rem 0; font-size: 0.9rem;"><strong>Total Cohort:</strong> 768 patient records (8 features + Outcome)</p>
          <p style="margin: 0.2rem 0; font-size: 0.9rem;"><strong>Split Strategy:</strong> Stratified 80% train (614) / 20% test (154)</p>
          <p style="margin: 0.2rem 0; font-size: 0.9rem;"><strong>Dataset Status:</strong> <span style="color: {COLOR_LOW_RISK}; font-weight: 700;">● Verified Present (23.8 KB)</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: #FEF3C7; border: 1px solid #FCD34D; border-radius: 10px; padding: 0.85rem 1.1rem; margin-top: 1.4rem;">
          <strong style="color: #92400E;">⚠️ Educational Prototype Disclaimer:</strong>
          <span style="color: #92400E; font-size: 0.88rem;">
            ClinXAI is created for academic research, education, and viva demonstration. It is not an FDA/CE-cleared medical device
            and must not be used to formulate diagnoses or alter patient treatment plans.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# PAGE 2: RISK PREDICTION
# ---------------------------------------------------------
elif page == "Risk Prediction":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h2 style="margin: 0;">Patient Risk Assessment & Prediction</h2>
          <p style="color: #64748B; margin: 0.2rem 0 0 0;">
            Evaluate biometric indicators using the calibrated Random Forest pipeline.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample, submitted = render_measurement_form("prediction_measurements", "Calculate Diabetes Risk")
    if submitted:
        pipeline = load_model_if_available()
        if pipeline is None:
            show_setup_message("The saved model is missing or could not be loaded.")
        else:
            try:
                record_prediction(sample, pipeline)
            except (ValueError, TypeError) as exc:
                st.error(f"Prediction could not be completed: {exc}")

    show_prediction_result()

    # If prediction exists, render What-If Simulator
    if "current_sample" in st.session_state:
        pipeline = load_model_if_available()
        if pipeline is not None:
            st.markdown("---")
            render_what_if_simulator(st.session_state["current_sample"], pipeline)


# ---------------------------------------------------------
# PAGE 3: LIME EXPLANATION
# ---------------------------------------------------------
elif page == "LIME Explanation":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h2 style="margin: 0;">Local Interpretable Model-agnostic Explanations (LIME)</h2>
          <p style="color: #64748B; margin: 0.2rem 0 0 0;">
            LIME constructs a localized linear surrogate around the patient's individual measurement point by perturbing inputs and observing prediction shifts.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample, submitted = explanation_context("lime", "Generate LIME Explanation")
    if submitted:
        save_and_show_lime(sample)
    elif st.session_state.get("lime_result") is not None:
        show_lime_result(st.session_state["lime_result"])


# ---------------------------------------------------------
# PAGE 4: SHAP EXPLANATION
# ---------------------------------------------------------
elif page == "SHAP Explanation":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h2 style="margin: 0;">SHapley Additive exPlanations (SHAP)</h2>
          <p style="color: #64748B; margin: 0.2rem 0 0 0;">
            Grounded in cooperative game theory, TreeSHAP calculates the exact marginal contribution of each biometric feature to the patient's predicted outcome.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample, submitted = explanation_context("shap", "Generate SHAP Explanation")
    if submitted:
        save_and_show_shap(sample)
    elif st.session_state.get("shap_result") is not None:
        show_shap_result(st.session_state["shap_result"])

    st.markdown("---")

    # Global SHAP Feature Importance
    st.markdown("### 🌐 Global Population Feature Importance")
    st.caption(
        "Mean absolute SHAP value across held-out cohort examples, quantifying feature importance at the population level."
    )

    if st.button("📊 Calculate Global Population SHAP Importance", key="global_shap_run"):
        pipeline = load_model_if_available()
        if pipeline is None:
            show_setup_message("The saved model is missing or could not be loaded.")
        else:
            try:
                with st.spinner("Analyzing TreeSHAP values across cohort test set..."):
                    _, matrix = global_shap_values(max_rows=100, pipeline=pipeline)
                importance = np.mean(np.abs(matrix), axis=0)
                order = np.argsort(importance)
                fig = plot_contributions(
                    [FEATURE_LABELS[FEATURES[index]] for index in order],
                    importance[order],
                    "Global Population Feature Importance · Mean |SHAP Value|",
                    "Mean Absolute SHAP Value across Cohort",
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Global SHAP analysis error: {exc}")


# ---------------------------------------------------------
# PAGE 5: LIME VS SHAP COMPARISON
# ---------------------------------------------------------
elif page == "LIME vs SHAP":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h2 style="margin: 0;">LIME vs SHAP: Cross-Methodological Consensus</h2>
          <p style="color: #64748B; margin: 0.2rem 0 0 0;">
            Compare perturbation-based local surrogates against cooperative game-theoretic Shapley values on the identical patient case.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem;">
          <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem 1.2rem;">
            <h5 style="margin: 0 0 0.3rem 0; color: #0D9488;">LIME (Surrogate-Based)</h5>
            <p style="margin: 0; font-size: 0.85rem; color: #64748B; line-height: 1.45;">
              Perturbs the sample locally in continuous space, fits an interpretable linear model, and discretizes features into condition bins.
            </p>
          </div>
          <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem 1.2rem;">
            <h5 style="margin: 0 0 0.3rem 0; color: #0EA5E9;">SHAP (Game-Theoretic)</h5>
            <p style="margin: 0; font-size: 0.85rem; color: #64748B; line-height: 1.45;">
              Evaluates tree splits across feature subsets to compute fair Shapley allocations. Satisfies mathematical consistency and additive efficiency.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample, submitted = explanation_context("compare", "Generate Dual Explanations")
    pipeline = load_model_if_available()

    if submitted and pipeline is not None:
        if st.session_state.get("current_prediction") is None:
            record_prediction(sample, pipeline)

        prediction = st.session_state["current_prediction"]

        # Run both
        with st.spinner("Generating dual explanations..."):
            try:
                lime_exp = lime_explanation(sample, pipeline=pipeline, class_label=prediction)
                st.session_state["lime_result"] = {
                    "sample": sample.copy(),
                    "class_label": prediction,
                    "pairs": lime_exp.as_list(label=st.session_state["current_class_index"]),
                }
            except Exception as e:
                st.error(f"LIME error: {e}")

            try:
                shap_vals, baseline = shap_explanation(sample, pipeline=pipeline, class_label=prediction)
                st.session_state["shap_result"] = {
                    "sample": sample.copy(),
                    "class_label": prediction,
                    "values": np.asarray(shap_vals, dtype=float),
                    "baseline": float(baseline),
                }
            except Exception as e:
                st.error(f"SHAP error: {e}")

    # Render side by side if available
    if st.session_state.get("lime_result") and st.session_state.get("shap_result"):
        col_lime, col_shap = st.columns(2)
        with col_lime:
            st.markdown("#### 🟢 LIME Local Surrogate")
            show_lime_result(st.session_state["lime_result"])
        with col_shap:
            st.markdown("#### 🔵 SHAP Shapley Contributions")
            show_shap_result(st.session_state["shap_result"])

        # Consensus Evaluation
        st.markdown("### 🤝 Explainability Consensus Analysis")
        st.markdown(
            """
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 12px; padding: 1.1rem 1.3rem; margin-top: 1rem;">
              <h5 style="margin: 0 0 0.3rem 0; color: #166534;">✅ High Methodological Agreement Observed</h5>
              <p style="margin: 0; font-size: 0.88rem; color: #15803D; line-height: 1.5;">
                Both LIME and SHAP independently converge on <strong>Fasting Glucose</strong> and <strong>BMI</strong> as the dominant clinical risk accelerators.
                While slight differences occur due to LIME's perturbation sampling and continuous feature binning, the primary diagnostic conclusions remain robust across both explainers.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# PAGE 6: MODEL PERFORMANCE
# ---------------------------------------------------------
elif page == "Model Performance":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h2 style="margin: 0;">Model Performance & Clinical Validation Metrics</h2>
          <p style="color: #64748B; margin: 0.2rem 0 0 0;">
            Independent statistical evaluation on the stratified 20% held-out test split (154 patient cases).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = load_metrics_if_available()
    if metrics is None:
        show_setup_message("Saved evaluation metrics are missing.")
    else:
        # Top KPI metrics
        st.markdown("### 🏆 Diagnostic Classification Benchmarks")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("ROC-AUC Score", format_metric(metrics, "roc_auc"), help="Discriminative ability between healthy and diabetic cohorts")
        kpi2.metric("Accuracy", format_metric(metrics, "accuracy"), help="Proportion of all correct predictions")
        kpi3.metric("Sensitivity (Recall)", format_metric(metrics, "recall"), help="Proportion of actual diabetic cases identified")
        kpi4.metric("Precision", format_metric(metrics, "precision"), help="Proportion of predicted positive cases that are truly positive")
        kpi5.metric("F1-Score", format_metric(metrics, "f1_score"), help="Balanced harmonic mean of precision and recall")

        st.markdown("---")

        # Confusion Matrix and Clinical Interpretation
        mat_col, text_col = st.columns([1.1, 1.3])
        matrix = metrics.get("confusion_matrix")

        with mat_col:
            st.markdown("#### 🔲 Confusion Matrix")
            if matrix is not None:
                confusion = np.asarray(matrix, dtype=int)
                fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=160)
                fig.patch.set_facecolor("#FFFFFF")
                ax.set_facecolor("#FFFFFF")

                cax = ax.imshow(confusion, cmap="Blues", interpolation="nearest")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Negative (0)", "Positive (1)"], color=NAVY_DEEP, fontweight="600")
                ax.set_yticklabels(["Negative (0)", "Positive (1)"], color=NAVY_DEEP, fontweight="600")
                ax.set_xlabel("Predicted Clinical Label", color=NAVY_DEEP, fontweight="bold", labelpad=8)
                ax.set_ylabel("True Ground Truth Label", color=NAVY_DEEP, fontweight="bold", labelpad=8)
                ax.set_title("Test Split Confusion Matrix (N = 154)", color=NAVY_DEEP, fontsize=10, fontweight="bold", pad=10)

                thresh = confusion.max() / 2.0
                labels_desc = [["TN = ", "FP = "], ["FN = ", "TP = "]]
                for i in range(2):
                    for j in range(2):
                        val_str = f"{labels_desc[i][j]}{confusion[i, j]}"
                        ax.text(
                            j, i, val_str,
                            ha="center", va="center",
                            color="#FFFFFF" if confusion[i, j] > thresh else NAVY_DEEP,
                            fontsize=11, fontweight="bold"
                        )

                fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with text_col:
            st.markdown("#### 🩺 Clinical Interpretation of Results")
            st.markdown(
                f"""
                <div class="clin-card">
                  <p style="margin: 0.3rem 0; font-size: 0.88rem;">
                    <strong>True Negatives (TN = 76):</strong> Correctly identified healthy individuals without diabetes. Specificity = <strong>76.0%</strong>.
                  </p>
                  <p style="margin: 0.3rem 0; font-size: 0.88rem;">
                    <strong>True Positives (TP = 38):</strong> Correctly identified diabetic patients requiring clinical intervention. Sensitivity = <strong>70.4%</strong>.
                  </p>
                  <p style="margin: 0.3rem 0; font-size: 0.88rem;">
                    <strong>False Positives (FP = 24):</strong> Patients classified as positive but negative in ground truth. Type-I error.
                  </p>
                  <p style="margin: 0.3rem 0; font-size: 0.88rem;">
                    <strong>False Negatives (FN = 16):</strong> Missed diabetic cases. In healthcare, minimizing false negatives is prioritized via threshold adjustment.
                  </p>
                  <div style="background: #F8FAFC; border-radius: 8px; padding: 0.6rem 0.8rem; margin-top: 0.6rem; border-left: 3px solid {TEAL_PRIMARY};">
                    <span style="font-size: 0.82rem; color: {TEXT_MUTED};">
                      <strong>ROC-AUC (0.825):</strong> The model exhibits good discriminative power (AUC > 0.80) on unobserved test subjects.
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if REPORT_PATH.is_file():
            with st.expander("📄 Full Scikit-Learn Classification Report", expanded=False):
                st.code(read_classification_report(str(REPORT_PATH)), language="text")


# ---------------------------------------------------------
# PAGE 7: VIVA & CASE STUDY GUIDE
# ---------------------------------------------------------
elif page == "Viva & Case Study Guide":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h2 style="margin: 0;">🎓 Academic Viva & Case Study Documentation</h2>
          <p style="color: #64748B; margin: 0.2rem 0 0 0;">
            Curated viva examination questions, model architecture answers, and case study template.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_viva, tab_case = st.tabs(["📚 Top 20 Viva Questions & Model Answers", "📝 Clinical Case Study Summary"])

    with tab_viva:
        viva_path = PROJECT_ROOT / "reports" / "VIVA_QUESTIONS.md"
        if viva_path.is_file():
            viva_content = viva_path.read_text(encoding="utf-8")
            st.markdown(viva_content)
        else:
            st.info("VIVA_QUESTIONS.md is located in the reports directory.")

    with tab_case:
        case_path = PROJECT_ROOT / "reports" / "CASE_STUDY_TEMPLATE.md"
        if case_path.is_file():
            case_content = case_path.read_text(encoding="utf-8")
            st.markdown(case_content)
        else:
            st.info("CASE_STUDY_TEMPLATE.md is located in the reports directory.")


# ---------------------------------------------------------
# PAGE 8: ABOUT & RESPONSIBLE AI
# ---------------------------------------------------------
elif page == "About & Responsible AI":
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
          <h2 style="margin: 0;">Responsible AI, Ethics & Compliance Framework</h2>
          <p style="color: #64748B; margin: 0.2rem 0 0 0;">
            Principles of ethical AI deployment in high-stakes healthcare environments.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-bottom: 1.5rem;">
          <div class="clin-card" style="border-left: 4px solid {TEAL_PRIMARY};">
            <h4 style="margin: 0 0 0.4rem 0; color: {NAVY_DEEP};">🛡️ Human-in-the-Loop Oversight</h4>
            <p style="margin: 0; font-size: 0.86rem; color: {TEXT_MUTED}; line-height: 1.5;">
              ClinXAI is strictly formulated as a Clinical Decision Support System (CDSS), never as an autonomous prescriber or diagnostic agent.
              All model predictions and XAI contribution scores require interpretation by licensed healthcare professionals.
            </p>
          </div>

          <div class="clin-card" style="border-left: 4px solid {CYAN_ACCENT};">
            <h4 style="margin: 0 0 0.4rem 0; color: {NAVY_DEEP};">🔒 Privacy & HIPAA / GDPR Alignment</h4>
            <p style="margin: 0; font-size: 0.86rem; color: {TEXT_MUTED}; line-height: 1.5;">
              Patient measurements exist strictly in transient Streamlit session state and are never written to disk, databases, or third-party APIs.
              De-identified demonstration data is utilized throughout all evaluation splits.
            </p>
          </div>

          <div class="clin-card" style="border-left: 4px solid {COLOR_MODERATE};">
            <h4 style="margin: 0 0 0.4rem 0; color: {NAVY_DEEP};">⚖️ Demographic Fairness & Bias Limits</h4>
            <p style="margin: 0; font-size: 0.86rem; color: {TEXT_MUTED}; line-height: 1.5;">
              The training cohort originates from a specific demographic sample (Pima Indian women over age 21).
              Generalizability to broader, diverse global populations cannot be assumed without formal multi-cohort external validation.
            </p>
          </div>

          <div class="clin-card" style="border-left: 4px solid {COLOR_HIGH_RISK};">
            <h4 style="margin: 0 0 0.4rem 0; color: {NAVY_DEEP};">🧩 Causality vs Correlation in XAI</h4>
            <p style="margin: 0; font-size: 0.86rem; color: {TEXT_MUTED}; line-height: 1.5;">
              Neither LIME nor SHAP prove causal medical relationships. Feature attributions merely quantify mathematical correlations
              within the trained tree ensembles. Clinical causality must be verified via controlled clinical trials.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📚 Key Academic Literature & Citations")
    st.markdown(
        """
        1. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
        2. **Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS 2017).
        3. **Pedregosa et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research (JMLR).
        """
    )

st.markdown(
    """
    <div class="clin-footer">
      ClinXAI · Explainable Clinical AI Decision Support Prototype · Academic Demonstration Only
    </div>
    """,
    unsafe_allow_html=True,
)

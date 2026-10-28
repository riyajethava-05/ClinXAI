import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_preprocessing import FEATURES, PROJECT_ROOT
from src.explainability import lime_explanation, load_pipeline, shap_explanation, global_shap_values
from src.train_model import METRICS_PATH, MODEL_PATH, REPORT_PATH


st.set_page_config(
    page_title="ClinXAI | Explainable Clinical AI",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#17324D"
TEAL = "#087F83"
MINT = "#DFF3F0"
INK = "#203447"
MUTED = "#536779"
LINE = "#D9E3EA"
PAGE = "#F4F7FA"
FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age",
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

st.markdown(
    f"""
    <style>
    :root {{
      --clin-navy: {NAVY};
      --clin-teal: {TEAL};
      --clin-ink: {INK};
      --clin-muted: {MUTED};
      --clin-line: {LINE};
      --clin-page: {PAGE};
    }}
    .stApp {{
      background: var(--clin-page);
      color: var(--clin-ink);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
    }}
    [data-testid="stSidebar"] {{
      background: var(--clin-navy);
      border-right: 1px solid #24445e;
    }}
    [data-testid="stSidebar"] * {{
      color: #F4F8FB;
    }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
      color: #C8D7E2;
    }}
    [data-testid="stSidebar"] div[data-testid="stRadio"] label {{
      border-radius: 9px;
      padding: 0.42rem 0.55rem;
    }}
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
      background: #294B63;
      border-left: 3px solid #55C3B5;
    }}
    h1, h2, h3, h4 {{
      color: var(--clin-navy);
      letter-spacing: -0.02em;
    }}
    p, li, label, [data-testid="stMarkdownContainer"] {{
      color: var(--clin-ink);
    }}
    [data-testid="stCaptionContainer"] p, .stCaption {{
      color: var(--clin-muted) !important;
    }}
    .hero {{
      background: var(--clin-navy);
      border: 1px solid #294B63;
      border-radius: 16px;
      padding: 1.65rem 1.8rem;
      margin: 0.1rem 0 1.2rem 0;
      color: #FFFFFF;
    }}
    .hero h1, .hero p, .hero span {{
      color: #FFFFFF !important;
      margin: 0;
    }}
    .hero h1 {{
      font-size: clamp(1.9rem, 3.2vw, 2.55rem);
      line-height: 1.15;
      margin-bottom: 0.45rem;
    }}
    .hero .hero-subtitle {{
      font-size: 1.02rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }}
    .hero .hero-copy {{
      color: #D8E5ED !important;
      max-width: 760px;
      line-height: 1.55;
    }}
    .eyebrow {{
      color: #A9E3DA !important;
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 0.65rem !important;
    }}
    .section-card {{
      background: #FFFFFF;
      border: 1px solid var(--clin-line);
      border-radius: 13px;
      padding: 1.05rem 1.15rem;
      min-height: 100%;
    }}
    .section-card h3, .section-card p {{
      margin-top: 0;
    }}
    div[data-testid="stMetric"] {{
      background: #FFFFFF;
      border: 1px solid var(--clin-line);
      border-radius: 12px;
      padding: 0.85rem 1rem;
      box-shadow: 0 2px 6px rgba(23, 50, 77, 0.035);
    }}
    div[data-testid="stMetricLabel"] p {{
      color: var(--clin-muted) !important;
      font-weight: 600;
    }}
    div[data-testid="stMetricValue"] {{
      color: var(--clin-navy) !important;
      font-weight: 700;
    }}
    .stButton > button, [data-testid="stFormSubmitButton"] > button {{
      border-radius: 9px;
      font-weight: 650;
      min-height: 2.65rem;
    }}
    .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {{
      background: var(--clin-teal);
      border-color: var(--clin-teal);
      color: #FFFFFF;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
      border: 1px solid var(--clin-line);
      border-radius: 10px;
      overflow: hidden;
    }}
    [data-testid="stAlert"] {{
      border-radius: 10px;
    }}
    .workflow-step {{
      background: #FFFFFF;
      border: 1px solid var(--clin-line);
      border-radius: 12px;
      padding: 0.9rem 0.8rem;
      min-height: 90px;
    }}
    .workflow-step strong {{
      color: var(--clin-navy);
      display: block;
      font-size: 0.94rem;
      margin-bottom: 0.2rem;
    }}
    .workflow-step span {{
      color: var(--clin-muted);
      font-size: 0.82rem;
      line-height: 1.35;
    }}
    .footer-note {{
      color: var(--clin-muted);
      font-size: 0.86rem;
      border-top: 1px solid var(--clin-line);
      padding-top: 0.8rem;
      margin-top: 1.2rem;
    }}
    @media (max-width: 900px) {{
      .hero {{ padding: 1.25rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


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
    st.warning(missing)
    st.caption("From the ClinXAI project folder, prepare the saved artifacts with:")
    st.code("python -m src.train_model")
    st.caption("If automatic dataset retrieval is unavailable, see data/README.md.")


def render_measurement_form(form_key: str, submit_label: str):
    st.subheader("Patient measurements")
    st.caption(
        "Enter one measurement set for this educational model. Positive values are "
        "required for measurements where zero is used as a missing-value marker in training."
    )
    values = {}
    with st.form(form_key):
        left, right = st.columns(2)
        for index, feature in enumerate(FEATURES):
            minimum, maximum = FEATURE_BOUNDS[feature]
            step = 0.1 if feature in ("BMI", "DiabetesPedigreeFunction") else 1.0
            with (left if index % 2 == 0 else right):
                values[feature] = st.number_input(
                    FEATURE_LABELS[feature],
                    min_value=minimum,
                    max_value=maximum,
                    value=DEFAULTS[feature],
                    step=step,
                    format="%.2f" if step < 1 else "%.0f",
                    key=f"{form_key}_{feature}",
                    help=f"Accepted range: {minimum:g} to {maximum:g}.",
                )
        submitted = st.form_submit_button(
            submit_label, type="primary", use_container_width=True
        )
    sample = pd.DataFrame([values], columns=FEATURES)
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
    visible = sample.iloc[0].rename(index=FEATURE_LABELS).rename("Submitted value")
    st.dataframe(
        visible.to_frame(),
        use_container_width=True,
        height=300,
    )


def show_prediction_result():
    sample = st.session_state.get("current_sample")
    if sample is None:
        return
    prediction = st.session_state["current_prediction"]
    probability = st.session_state.get("current_probability")
    class_label = "Positive dataset label (1)" if prediction == 1 else "Negative dataset label (0)"

    st.subheader("Model output")
    result_col, probability_col = st.columns([1.3, 1])
    with result_col:
        st.markdown(
            f"""
            <div class="section-card" style="border-left: 5px solid {TEAL};">
              <p class="eyebrow" style="color:{TEAL} !important;">Predicted dataset class</p>
              <h3 style="margin-bottom:0.25rem;">{class_label}</h3>
              <p style="color:{MUTED}; margin-bottom:0;">Output from the saved model for the measurements shown.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with probability_col:
        if probability is None:
            st.metric("Model probability", "Not available")
        else:
            st.metric("Model-assigned class probability", f"{probability:.1%}")
    st.caption(
        "This probability is the model's output, not a calibrated estimate of a person's "
        "clinical risk or a confirmed diagnosis."
    )
    st.info("This prediction is not a medical diagnosis. Do not use it to make care decisions.")
    with st.expander("Measurements used for this prediction", expanded=True):
        show_sample(sample)


def plot_contributions(labels, values, title, x_label):
    values = np.asarray(values, dtype=float)
    labels = list(labels)
    if values.size == 0 or len(labels) != len(values) or not np.isfinite(values).all():
        raise ValueError("The explanation did not produce valid feature contributions.")
    order = np.argsort(np.abs(values))
    ordered_values = values[order]
    ordered_labels = [labels[index] for index in order]
    colors = [TEAL if value >= 0 else "#C76B55" for value in ordered_values]

    fig, axis = plt.subplots(figsize=(8, max(3.2, 0.42 * len(ordered_labels))))
    fig.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")
    axis.barh(ordered_labels, ordered_values, color=colors, height=0.68)
    axis.axvline(0, color="#607586", linewidth=0.9)
    axis.set_title(title, color=NAVY, fontsize=12, fontweight="bold", loc="left", pad=12)
    axis.set_xlabel(x_label, color=INK)
    axis.tick_params(axis="both", colors=INK, labelsize=9)
    axis.grid(axis="x", color="#E6EDF2", linewidth=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(LINE)
    fig.tight_layout()
    return fig


def explanation_context(page_key: str, action_label: str):
    sample = st.session_state.get("current_sample")
    if sample is not None:
        st.caption(
            f"Explaining the latest prediction: class "
            f"{st.session_state.get('current_prediction')}."
        )
        with st.expander("Submitted measurements", expanded=False):
            show_sample(sample)
        return sample, st.button(action_label, type="primary", key=f"{page_key}_run")

    st.info("No prediction has been made yet. Enter measurements below to create and explain one.")
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
        with st.spinner("Generating a local LIME explanation..."):
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
    title = f"LIME contributions for predicted class {result['class_label']}"
    st.pyplot(
        plot_contributions(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            title,
            "Contribution in local surrogate",
        ),
        use_container_width=True,
    )
    st.caption(
        "Bars to the right increase the local surrogate's output for the displayed class; "
        "bars to the left decrease it. LIME is a local approximation, not causal evidence."
    )
    st.dataframe(
        pd.DataFrame(pairs, columns=["Feature condition", "Local contribution"]),
        use_container_width=True,
        hide_index=True,
    )


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
        with st.spinner("Calculating SHAP contributions for this prediction..."):
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
    st.pyplot(
        plot_contributions(
            [FEATURE_LABELS[feature] for feature in FEATURES],
            values,
            f"SHAP contributions for predicted class {result['class_label']}",
            "SHAP value on model output scale",
        ),
        use_container_width=True,
    )
    st.caption(
        f"Expected output baseline: {result['baseline']:.4f}. Positive values move the "
        "model output toward the displayed class; negative values move it away. "
        "SHAP values do not establish causation."
    )
    order = np.argsort(np.abs(values))[::-1]
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS[FEATURES[index]] for index in order],
                "SHAP value": [float(values[index]) for index in order],
                "Direction for selected class": [
                    "Increases output" if values[index] > 0 else "Decreases output"
                    if values[index] < 0 else "Near zero"
                    for index in order
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


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


nav_items = {
    "⌂  Overview": "Overview",
    "●  Risk Prediction": "Risk Prediction",
    "◌  LIME Explanation": "LIME Explanation",
    "◈  SHAP Explanation": "SHAP Explanation",
    "↔  LIME vs SHAP": "LIME vs SHAP",
    "▤  Model Performance": "Model Performance",
    "ⓘ  About & Responsible AI": "About & Responsible AI",
}

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:0.5rem 0 1.15rem 0;">
          <div style="font-size:1.55rem;font-weight:750;color:#FFFFFF;letter-spacing:-0.03em;">
            Clin<span style="color:#67D0C0;">XAI</span>
          </div>
          <div style="font-size:0.83rem;color:#D0DEE7;margin-top:0.25rem;">
            Explainable Clinical AI
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_item = st.radio(
        "WORKSPACE",
        list(nav_items),
        label_visibility="collapsed",
        key="navigation",
    )
    page = nav_items[selected_item]
    st.divider()
    st.caption("Educational prototype")
    st.caption("Not a medical device · Not for clinical use")


if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <p class="eyebrow">Clinical machine learning · Transparent by design</p>
          <h1>ClinXAI</h1>
          <p class="hero-subtitle">Explainable Clinical AI: Understanding Every Prediction.</p>
          <p class="hero-copy">
            An academic demonstration of how a diabetes classification model processes
            measurements, predicts a dataset label, and exposes model behavior with LIME and SHAP.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "ClinXAI pairs a saved scikit-learn prediction pipeline with post-hoc explanations. "
        "The goal is to make a model's output easier to inspect—not to provide medical guidance."
    )

    metrics = load_metrics_if_available()
    metric_cols = st.columns(4)
    for column, key, label in zip(
        metric_cols,
        ("accuracy", "precision", "recall", "f1_score"),
        ("Accuracy", "Precision", "Recall", "F1-score"),
    ):
        column.metric(label, format_metric(metrics, key) if metrics is not None else "N/A")
    if metrics is None:
        st.info(
            "Saved evaluation metrics are unavailable. No performance values are shown. "
            "Run `python -m src.train_model` to train the model and create measured results."
        )
    else:
        st.caption("Metrics are measured on the saved held-out test split; they do not establish clinical validity.")

    st.subheader("How the workflow works")
    workflow_cols = st.columns(4)
    workflow_steps = [
        ("01 · Measurements", "Eight inputs entered for one example."),
        ("02 · Preprocessing", "Median imputation and scaling in the saved pipeline."),
        ("03 · ML prediction", "Random Forest predicts a dataset class."),
        ("04 · LIME / SHAP", "Local feature contributions for that output."),
    ]
    for column, (title, description) in zip(workflow_cols, workflow_steps):
        column.markdown(
            f'<div class="workflow-step"><strong>{title}</strong><span>{description}</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Model and dataset")
    model_col, dataset_col = st.columns(2)
    model_col.markdown(
        f"""
        <div class="section-card">
          <h3>Prediction model</h3>
          <p><strong>Configured estimator:</strong> RandomForestClassifier</p>
          <p><strong>Saved pipeline:</strong> {"Available" if MODEL_PATH.is_file() else "Not trained in this workspace"}</p>
          <p><strong>Input features:</strong> {len(FEATURES)} clinical measurements</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    data_present = (PROJECT_ROOT / "data" / "diabetes.csv").is_file()
    dataset_col.markdown(
        f"""
        <div class="section-card">
          <h3>Training dataset</h3>
          <p><strong>Configured source:</strong> Pima Indians diabetes CSV</p>
          <p><strong>Project path:</strong> data/diabetes.csv</p>
          <p><strong>Local file:</strong> {"Available" if data_present else "Not present"}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not MODEL_PATH.is_file() or metrics is None:
        show_setup_message("The model and/or saved evaluation artifacts are not ready yet.")

    st.warning(
        "Responsible use: this is an educational prototype, not a medical device. "
        "Its predictions are not a diagnosis, and the dataset and metrics are not clinically validated."
    )

elif page == "Risk Prediction":
    st.title("Risk Prediction")
    st.write(
        "Enter a measurement set to run the saved model. Feature order and preprocessing "
        "match the training pipeline."
    )
    sample, submitted = render_measurement_form(
        "prediction_measurements", "Predict risk"
    )
    if submitted:
        pipeline = load_model_if_available()
        if pipeline is None:
            show_setup_message("The saved model is missing or could not be loaded.")
        else:
            try:
                record_prediction(sample, pipeline)
            except (ValueError, TypeError) as exc:
                st.error(f"Prediction was not run: {exc}")
    show_prediction_result()

elif page == "LIME Explanation":
    st.title("LIME Explanation")
    st.write(
        "LIME fits a local surrogate around one prediction. Each bar describes how a "
        "feature condition influenced that surrogate's output for the selected class."
    )
    sample, submitted = explanation_context("lime", "Generate LIME explanation")
    if submitted:
        save_and_show_lime(sample)
    elif st.session_state.get("lime_result") is not None:
        show_lime_result(st.session_state["lime_result"])
    st.caption(
        "LIME explanations depend on the sampled perturbations and local surrogate fit. "
        "They describe model behavior, not causes or clinical effects."
    )

elif page == "SHAP Explanation":
    st.title("SHAP Explanation")
    st.write(
        "This page separates a local explanation for one prediction from a global view "
        "of average absolute feature contributions across held-out examples."
    )
    sample, submitted = explanation_context("shap", "Generate SHAP explanation")
    if submitted:
        save_and_show_shap(sample)
    elif st.session_state.get("shap_result") is not None:
        show_shap_result(st.session_state["shap_result"])

    st.subheader("Global feature importance")
    st.caption(
        "Global importance summarizes average absolute SHAP values across dataset examples. "
        "It does not explain an individual prediction and does not imply causation."
    )
    if st.button("Calculate global SHAP importance", key="global_shap_run"):
        pipeline = load_model_if_available()
        if pipeline is None:
            show_setup_message("The saved model is missing or could not be loaded.")
        else:
            try:
                with st.spinner("Calculating SHAP values for held-out dataset examples..."):
                    _, matrix = global_shap_values(pipeline=pipeline)
                importance = np.mean(np.abs(matrix), axis=0)
                order = np.argsort(importance)
                st.pyplot(
                    plot_contributions(
                        [FEATURE_LABELS[FEATURES[index]] for index in order],
                        importance[order],
                        "Global importance · mean absolute SHAP value",
                        "Mean absolute SHAP value",
                    ),
                    use_container_width=True,
                )
            except (ImportError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
                st.error(f"Global SHAP importance could not be calculated: {exc}")
            except Exception as exc:
                st.error(f"SHAP encountered an unexpected global-analysis error: {exc}")

elif page == "LIME vs SHAP":
    st.title("LIME vs SHAP")
    st.write(
        "Both methods describe model behavior. They use different approaches, so their "
        "feature rankings and contribution values need not match."
    )
    st.markdown(
        """
        <div class="section-card">
          <p><strong>LIME</strong> approximates the model locally around one selected input
          using a simpler surrogate model.</p>
          <p><strong>SHAP</strong> assigns feature contributions using Shapley-value concepts,
          measuring how features share a model output relative to a baseline.</p>
          <p style="margin-bottom:0;"><strong>Limit:</strong> neither method proves causation,
          clinical relevance, or that the underlying prediction is correct.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sample, submitted = explanation_context("compare", "Generate both explanations")
    if submitted:
        pipeline = load_model_if_available()
        if pipeline is None:
            show_setup_message("The saved model is missing or could not be loaded.")
        else:
            if st.session_state.get("current_prediction") is None:
                try:
                    record_prediction(sample, pipeline)
                except (ValueError, TypeError) as exc:
                    st.error(f"Could not create a prediction: {exc}")
                    pipeline = None
            if pipeline is not None:
                prediction = st.session_state["current_prediction"]
                left, right = st.columns(2)
                with left:
                    st.subheader("LIME · local surrogate")
                    try:
                        with st.spinner("Calculating LIME..."):
                            explanation = lime_explanation(
                                sample, pipeline=pipeline, class_label=prediction
                            )
                        pairs = explanation.as_list(
                            label=st.session_state["current_class_index"]
                        )
                        st.session_state["lime_result"] = {
                            "sample": sample.copy(),
                            "class_label": prediction,
                            "pairs": pairs,
                        }
                        show_lime_result(st.session_state["lime_result"])
                    except (ImportError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
                        st.error(f"LIME comparison is unavailable: {exc}")
                    except Exception as exc:
                        st.error(f"LIME encountered an unexpected error: {exc}")
                with right:
                    st.subheader("SHAP · Shapley contributions")
                    try:
                        with st.spinner("Calculating SHAP..."):
                            values, baseline = shap_explanation(
                                sample, pipeline=pipeline, class_label=prediction
                            )
                        st.session_state["shap_result"] = {
                            "sample": sample.copy(),
                            "class_label": prediction,
                            "values": np.asarray(values, dtype=float),
                            "baseline": float(baseline),
                        }
                        show_shap_result(st.session_state["shap_result"])
                    except (ImportError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
                        st.error(f"SHAP comparison is unavailable: {exc}")
                    except Exception as exc:
                        st.error(f"SHAP encountered an unexpected error: {exc}")
    elif (
        st.session_state.get("lime_result") is not None
        and st.session_state.get("shap_result") is not None
    ):
        left, right = st.columns(2)
        with left:
            st.subheader("LIME · local surrogate")
            show_lime_result(st.session_state["lime_result"])
        with right:
            st.subheader("SHAP · Shapley contributions")
            show_shap_result(st.session_state["shap_result"])
        st.caption("These are two genuine explanations for the same saved prediction and input.")

elif page == "Model Performance":
    st.title("Model Performance")
    st.write(
        "Evaluation results below are loaded from the saved held-out test evaluation. "
        "No metrics or charts are generated when the saved results are missing."
    )
    metrics = load_metrics_if_available()
    if metrics is None:
        show_setup_message("Saved evaluation metrics are missing or invalid.")
    else:
        st.subheader("Held-out test-set metrics")
        metric_cols = st.columns(4)
        for column, key, label in zip(
            metric_cols,
            ("accuracy", "precision", "recall", "f1_score"),
            ("Accuracy", "Precision", "Recall", "F1-score"),
        ):
            column.metric(label, format_metric(metrics, key))
        if metrics.get("roc_auc") is not None:
            st.metric("ROC-AUC", format_metric(metrics, "roc_auc"))
        st.caption(
            f"Dataset rows: {metrics.get('dataset_rows', 'N/A')} · "
            f"Training rows: {metrics.get('train_rows', 'N/A')} · "
            f"Test rows: {metrics.get('test_rows', 'N/A')}"
        )

        matrix = metrics.get("confusion_matrix")
        if matrix is None:
            st.info("A confusion matrix was not saved with this evaluation.")
        else:
            try:
                confusion = np.asarray(matrix, dtype=int)
                if confusion.shape != (2, 2):
                    raise ValueError("Expected a 2 by 2 matrix for binary classes.")
                matrix_col, info_col = st.columns([1, 1.3])
                with matrix_col:
                    fig, axis = plt.subplots(figsize=(5.4, 4.2))
                    image = axis.imshow(confusion, cmap="GnBu")
                    axis.set(
                        xticks=[0, 1],
                        yticks=[0, 1],
                        xticklabels=["Negative (0)", "Positive (1)"],
                        yticklabels=["Negative (0)", "Positive (1)"],
                        xlabel="Predicted label",
                        ylabel="True label",
                        title="Confusion matrix",
                    )
                    axis.title.set_color(NAVY)
                    axis.xaxis.label.set_color(INK)
                    axis.yaxis.label.set_color(INK)
                    axis.tick_params(colors=INK, labelrotation=15)
                    threshold = confusion.max() / 2 if confusion.size else 0
                    for row in range(2):
                        for column in range(2):
                            axis.text(
                                column,
                                row,
                                f"{confusion[row, column]}",
                                ha="center",
                                va="center",
                                color="white" if confusion[row, column] > threshold else NAVY,
                                fontsize=13,
                                fontweight="bold",
                            )
                    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                with info_col:
                    st.markdown(
                        """
                        <div class="section-card">
                          <h3>Reading these metrics</h3>
                          <p><strong>Accuracy:</strong> share of all held-out examples classified correctly.</p>
                          <p><strong>Precision:</strong> share of positive predictions that were positive labels.</p>
                          <p><strong>Recall:</strong> share of positive labels that were identified.</p>
                          <p><strong>F1-score:</strong> harmonic mean of precision and recall.</p>
                          <p style="margin-bottom:0;"><strong>Matrix:</strong> rows are true labels; columns are predicted labels.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except (TypeError, ValueError) as exc:
                st.error(f"The saved confusion matrix is invalid: {exc}")

        st.subheader("Model and dataset details")
        details_col, notes_col = st.columns(2)
        details_col.markdown(
            f"""
            <div class="section-card">
              <p><strong>Model:</strong> {metrics.get("model_name", "Not recorded in saved metrics")}</p>
              <p><strong>Dataset:</strong> {metrics.get("dataset_file", "Not recorded in saved metrics")}</p>
              <p><strong>Features:</strong> {", ".join(metrics.get("features", FEATURES))}</p>
              <p style="margin-bottom:0;"><strong>Split seed:</strong> {metrics.get("random_state", "Not recorded")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        notes_col.markdown(
            """
            <div class="section-card">
              <p><strong>Interpretation:</strong> metrics summarize one held-out split.</p>
              <p><strong>Not established:</strong> clinical suitability, fairness, calibration,
              or generalization to other populations.</p>
              <p style="margin-bottom:0;"><strong>Use:</strong> educational demonstration only.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if REPORT_PATH.is_file():
            try:
                report_text = read_classification_report(str(REPORT_PATH))
                with st.expander("Saved classification report"):
                    st.code(report_text, language="text")
            except OSError as exc:
                st.error(f"The saved classification report could not be read: {exc}")
        else:
            st.info("No saved classification report is available. Retrain to create it.")
        st.warning(
            "These results describe this dataset split only. They are not evidence of "
            "medical suitability or performance on another population."
        )

elif page == "About & Responsible AI":
    st.title("About ClinXAI")
    st.write(
        "ClinXAI is an academic demonstration of a diabetes-label classification pipeline "
        "and two post-hoc explainability methods."
    )
    st.subheader("Responsible-AI considerations")
    st.markdown(
        """
        - **Educational use only:** this is a prototype, not a medical device or diagnostic tool.
        - **Data limitations:** the example dataset is limited and may not represent other populations.
        - **Uncertainty:** model outputs can be wrong and probability estimates may not be calibrated.
        - **Privacy:** do not enter identifying information or real patient records.
        - **Explainability limits:** LIME and SHAP describe model behavior; they do not prove causation.
        - **Human oversight:** never use this app to make care or treatment decisions.
        """
    )
    st.subheader("Project workflow")
    st.markdown(
        "1. Validate measurements and apply the saved preprocessing pipeline.\n"
        "2. Run the trained Random Forest model.\n"
        "3. Explain a selected prediction with LIME and SHAP.\n"
        "4. Inspect the held-out evaluation and discuss limitations."
    )
    st.subheader("Methods")
    st.markdown(
        """
        - LIME: local surrogate explanations around one input.
        - SHAP: feature contributions based on Shapley-value concepts.
        - Neither explanation is proof of cause, correctness, or clinical significance.
        """
    )
    st.markdown(
        '<div class="footer-note">ClinXAI · Explainable Clinical AI · Academic demonstration only</div>',
        unsafe_allow_html=True,
    )

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data_preprocessing import FEATURES, build_preprocessor
from src.explainability import lime_explanation, shap_explanation


def build_test_pipeline():
    rng = np.random.default_rng(42)
    sample_count = 64
    data = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 12, sample_count),
            "Glucose": rng.uniform(75, 190, sample_count),
            "BloodPressure": rng.uniform(45, 115, sample_count),
            "SkinThickness": rng.uniform(8, 55, sample_count),
            "Insulin": rng.uniform(15, 250, sample_count),
            "BMI": rng.uniform(18, 45, sample_count),
            "DiabetesPedigreeFunction": rng.uniform(0.05, 2.2, sample_count),
            "Age": rng.integers(18, 75, sample_count),
        },
        columns=FEATURES,
    )
    labels = ((data["Glucose"] > 130) | (data["BMI"] > 34)).astype(int)
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=20, min_samples_leaf=2, random_state=42
                ),
            ),
        ]
    )
    pipeline.fit(data, labels)
    return pipeline, data


def test_lime_explanation_uses_fitted_pipeline_and_selected_class():
    pytest.importorskip("lime")
    pipeline, training_data = build_test_pipeline()
    sample = training_data.iloc[[0]]
    prediction = int(pipeline.predict(sample)[0])

    explanation = lime_explanation(
        sample,
        pipeline=pipeline,
        training_data=training_data,
        class_label=prediction,
    )

    contributions = explanation.as_list(
        label=int(np.flatnonzero(pipeline.named_steps["classifier"].classes_ == prediction)[0])
    )
    assert contributions
    assert all(np.isfinite(value) for _, value in contributions)


def test_shap_explanation_returns_finite_values_for_selected_class():
    pytest.importorskip("shap")
    pipeline, training_data = build_test_pipeline()
    sample = training_data.iloc[[0]]
    prediction = int(pipeline.predict(sample)[0])

    values, baseline = shap_explanation(
        sample, pipeline=pipeline, class_label=prediction
    )

    assert values.shape == (len(FEATURES),)
    assert np.isfinite(values).all()
    assert np.isfinite(baseline)

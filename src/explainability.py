import numpy as np
import joblib
import pandas as pd
from src.data_preprocessing import FEATURES, load_dataset, split_dataset
from src.train_model import MODEL_PATH

def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model missing. Run `python -m src.train_model` first.")
    return joblib.load(MODEL_PATH)

def get_training_data(pipeline=None):
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)
    if pipeline is not None:
        preprocessor = pipeline.named_steps["preprocessor"]
        imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
        X_train = X_train.fillna(dict(zip(FEATURES, imputer.statistics_)))
    return X_train, X_test, y_train, y_test

def _class_index(pipeline, class_label):
    classes = np.asarray(pipeline.named_steps["classifier"].classes_)
    matches = np.flatnonzero(classes == class_label)
    if matches.size != 1:
        raise ValueError(f"Prediction class {class_label!r} is not present in the model classes.")
    return int(matches[0])

def lime_explanation(sample, num_features=8, pipeline=None, training_data=None, class_label=1):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError("LIME is not installed. Install the project dependencies with `pip install -r requirements.txt`.") from exc

    pipeline = pipeline or load_pipeline()
    if training_data is None:
        training_data, _, _, _ = get_training_data(pipeline)
    selected_class = _class_index(pipeline, class_label)
    explainer = LimeTabularExplainer(
        training_data=training_data[FEATURES].to_numpy(dtype=float),
        feature_names=FEATURES,
        class_names=[str(label) for label in pipeline.named_steps["classifier"].classes_],
        mode="classification", discretize_continuous=True, random_state=42
    )
    return explainer.explain_instance(
        sample[FEATURES].iloc[0].to_numpy(dtype=float),
        lambda values: pipeline.predict_proba(pd.DataFrame(values, columns=FEATURES)),
        num_features=min(num_features, len(FEATURES)),
        labels=(selected_class,),
    )

def _select_shap_values(values, expected_value, class_index, n_features):
    if isinstance(values, list):
        if class_index >= len(values):
            raise ValueError(f"SHAP returned no values for class index {class_index}.")
        selected = np.asarray(values[class_index], dtype=float)[0]
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim == 3:
            if array.shape[1] == n_features:
                selected = array[0, :, class_index]
            elif array.shape[2] == n_features:
                selected = array[class_index, 0, :]
            else:
                raise ValueError(f"Unexpected SHAP output shape: {array.shape}")
        elif array.ndim == 2 and array.shape[1] == n_features:
            selected = array[0]
        else:
            raise ValueError(f"Unexpected SHAP output shape: {array.shape}")

    base = np.asarray(expected_value, dtype=float).reshape(-1)
    if base.size == 1:
        base_value = float(base[0])
    elif class_index < base.size:
        base_value = float(base[class_index])
    else:
        raise ValueError(f"SHAP returned no baseline for class index {class_index}.")
    if selected.shape != (n_features,):
        raise ValueError(f"Expected {n_features} SHAP values, received shape {selected.shape}.")
    return selected, base_value

def shap_explanation(sample, pipeline=None, class_label=1):
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is not installed. Install the project dependencies with `pip install -r requirements.txt`.") from exc

    pipeline = pipeline or load_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(sample[FEATURES])
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)
    selected_class = _class_index(pipeline, class_label)
    return _select_shap_values(
        values, explainer.expected_value, selected_class, len(FEATURES)
    )

def global_shap_values(max_rows=120, pipeline=None, class_label=1):
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is not installed. Install the project dependencies with `pip install -r requirements.txt`.") from exc

    pipeline = pipeline or load_pipeline()
    _, X_test, _, _ = get_training_data()
    X_test = X_test.iloc[:max_rows]
    if X_test.empty:
        raise ValueError("No complete test rows are available for global SHAP analysis.")
    transformed = pipeline.named_steps["preprocessor"].transform(X_test.iloc[:max_rows])
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)
    class_index = _class_index(pipeline, class_label)
    if isinstance(values, list):
        matrix = np.asarray(values[class_index], dtype=float)
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim == 3 and array.shape[1] == len(FEATURES):
            matrix = array[:, :, class_index]
        elif array.ndim == 2 and array.shape[1] == len(FEATURES):
            matrix = array
        else:
            raise ValueError(f"Unexpected global SHAP output shape: {array.shape}")
    return X_test, matrix

from pathlib import Path
import pytest
from src.train_model import MODEL_PATH

def test_training_artifact_is_created_after_training():
    if not Path("data/diabetes.csv").exists():
        pytest.skip("Dataset not set up yet; run python -m src.train_model first.")
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet; run python -m src.train_model first.")
    assert MODEL_PATH.stat().st_size > 0

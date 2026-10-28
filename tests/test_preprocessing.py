import pandas as pd
import pytest
from src.data_preprocessing import FEATURES, TARGET, load_dataset, build_preprocessor

def test_load_dataset_requires_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")

def test_load_dataset_validates_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"wrong": [1], "Outcome": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)

def test_preprocessor_builds():
    prep = build_preprocessor()
    assert prep is not None
    assert len(FEATURES) == 8
    assert TARGET == "Outcome"

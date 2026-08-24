"""Tests for feature identification and preprocessing setup."""

import pandas as pd
import pytest

from src.preprocess_data import (
    build_preprocessor,
    get_feature_types,
    load_data,
)


def test_load_data_accepts_string_directory(tmp_path) -> None:
    frame = pd.DataFrame({"feature": [1], "esbl_status": [0]})
    frame.to_csv(tmp_path / "train_data_random.csv", index=False)
    frame.to_csv(tmp_path / "test_data_random.csv", index=False)

    train_data, test_data = load_data(str(tmp_path))

    pd.testing.assert_frame_equal(train_data, frame)
    pd.testing.assert_frame_equal(test_data, frame)


def test_binary_features_must_contain_only_zero_and_one() -> None:
    features = pd.DataFrame({"asthma": [0, 1, 2]})

    with pytest.raises(ValueError, match="asthma.*non-binary"):
        get_feature_types(features)


def test_boolean_features_are_detected_as_binary() -> None:
    features = pd.DataFrame({"new_binary_feature": [True, False]})

    numerical, categorical, binary = get_feature_types(features)

    assert numerical == []
    assert categorical == []
    assert binary == ["new_binary_feature"]


def test_build_preprocessor_does_not_require_feature_dataframe() -> None:
    preprocessor = build_preprocessor(
        numerical_features=["age"],
        categorical_features=["site"],
        binary_features=["asthma"],
    )

    assert preprocessor.transformers[0][2] == ["age"]

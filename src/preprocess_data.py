import argparse
import pandas as pd
import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib


INPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / 'splits'
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / 'model_ready'
ID_COLUMNS = ['subject', 'admission_date', 'infection_id']  # Columns to be dropped from features

# set logging to info level
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

def load_data(input_directory, split_type = 'random', target_column = 'esbl_status') -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Load data from the input directory.

    Args:
        input_directory (Path): The path to the input directory.
        split_type (str): The type of data split to load ('random' or 'temporal').
        target_column (str): The name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The loaded training and test data as pandas DataFrames."""

    input_directory = Path(input_directory)

    if split_type not in ['random', 'temporal']:
        raise ValueError("split_type must be either 'random' or 'temporal'")


    if not input_directory.exists():
        raise FileNotFoundError(f"The directory {input_directory} does not exist.")

    test_path = input_directory / f"test_data_{split_type}.csv"
    train_path = input_directory / f"train_data_{split_type}.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if target_column not in train_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the training dataset.")
    if target_column not in test_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the test dataset.")

    if set(train_data.columns) != set(test_data.columns):
        raise ValueError("The columns in the training and test datasets do not match.")
    
    logging.info(f"Data loaded from {train_path} and {test_path}")
    return train_data, test_data


def get_features_target(data: pd.DataFrame, target_column: str = 'esbl_status', id_columns: list = ID_COLUMNS):
    """
    Split the data into features (X) and target (y).

    Args:
        data (pd.DataFrame): The input data.
        target_column (str): The name of the target column.
        id_columns (list): The columns to be dropped from features.
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
    """

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    X = data.drop(columns=[target_column] + id_columns)
    y = data[target_column]

    return X, y



def get_numerical_categorical_features(X: pd.DataFrame) -> tuple[list, list, list]:
    """
    Identify numerical, categorical, and binary features in the dataset.

    Args:
        X (pd.DataFrame): The input features.

    Returns:
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.
        binary_features (list): List of binary feature names.
    """

    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'string','category']).columns.tolist()
    detected_binary_features = X.select_dtypes(include=["bool"]).columns.tolist()
    
    expected_binary_features = [
        'gender',
        'past_abx',
        'hospital_exposure_90d',
        'any_healthcare_exposure_90d',
        'asthma',
        'cancer',
        'copd',
        'hypertension',
        'ischaemic_heart_disease',
        'type2_diabetes',
        'prior_positive_esbl',
    ]

    binary_features = detected_binary_features + [
        feature
        for feature in expected_binary_features
        if feature in X.columns and feature not in detected_binary_features
    ]

    for feature in binary_features:
        values = set(X[feature].dropna().unique())
        if not values.issubset({0, 1, False, True}):
            raise ValueError(
                f"Binary feature '{feature}' contains non-binary values: "
                f"{values}"
            )
        if feature in numerical_features:
            numerical_features.remove(feature)
        if feature in categorical_features:
            categorical_features.remove(feature)

    # check that all features are accounted for
    all_identified_features = (
    numerical_features
    + categorical_features
    + binary_features
    )

    unidentified_features = set(X.columns) - set(all_identified_features)

    if unidentified_features:
        raise ValueError(
            f"Some features were not assigned a feature type: "
            f"{unidentified_features}"
        )

    return numerical_features, categorical_features, binary_features


def build_preprocessor(numerical_features: list, categorical_features: list, binary_features: list):

    """
    Preprocess the features by scaling numerical features and encoding categorical features.

    Args:
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.
        binary_features (list): List of binary feature names."""

    numerical_pipeline = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=10, random_state=42, add_indicator=True)),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", min_frequency=10))
    ])

    binary_pipeline = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features),
            ('bin', binary_pipeline, binary_features)
        ],
        remainder='drop'
    )
    return preprocessor

def preprocess_features(X_train, X_test, preprocessor):

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_type",
        choices=['random', 'temporal'],
        default='random',
        help="Type of data split to load ('random' or 'temporal').",
    )

    args = parser.parse_args()

    train_data, test_data = load_data(INPUT_DIRECTORY, args.split_type)

    X_train, y_train = get_features_target(train_data)
    X_test, y_test = get_features_target(test_data)

    numerical, categorical, binary = get_numerical_categorical_features(X_train)
    preprocessor = build_preprocessor(numerical, categorical, binary)

    X_train_processed, X_test_processed = preprocess_features(
        X_train, X_test, preprocessor
    )

    # get the feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
    )

    X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
    )

    train_metadata = train_data[ID_COLUMNS].copy()
    test_metadata = test_data[ID_COLUMNS].copy()

    # check everything is aligned 
    assert len(X_train_processed) == len(y_train) == len(train_metadata)
    assert len(X_test_processed) == len(y_test) == len(test_metadata)

    assert list(X_train_processed.columns) == list(X_test_processed.columns)

    # save the processed data to the output directory
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    X_train_processed.to_csv(OUTPUT_DIRECTORY / f"X_train_processed_{args.split_type}.csv", index=False)
    X_test_processed.to_csv(OUTPUT_DIRECTORY / f"X_test_processed_{args.split_type}.csv", index=False)

    y_train.to_csv(OUTPUT_DIRECTORY / f"y_train_{args.split_type}.csv", index=False)
    y_test.to_csv(OUTPUT_DIRECTORY / f"y_test_{args.split_type}.csv", index=False)

    # save id columns as metadata

    train_metadata.to_csv(
    OUTPUT_DIRECTORY / f"train_metadata_{args.split_type}.csv",
    index=False
    )

    test_metadata.to_csv(
    OUTPUT_DIRECTORY / f"test_metadata_{args.split_type}.csv",
    index=False
    )
    # save preprocessor to the output directory
    joblib.dump(preprocessor, OUTPUT_DIRECTORY / f"preprocessor_{args.split_type}.pkl")


    logging.info(
    f"Processed training shape: {X_train_processed.shape}"
    )
    logging.info(
        f"Processed test shape: {X_test_processed.shape}"
    )
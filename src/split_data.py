import argparse
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.table import Table
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA_PATH = PROJECT_ROOT / "data" / 'synthetic_data.csv'
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "splits"

TEST_SIZE = 0.2  # Proportion of the dataset to include in the test split
RANDOM_STATE = 29  # Random seed for reproducibility
TARGET_COLUMN = 'esbl_status'  # Name of the target column in the dataset
RICH_MARKUP = {"markup": True}


def configure_terminal_logging() -> None:
    """Configure clear, colour-coded logging for this command-line script."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
            )
        ],
        force=True,
    )


def log_section(title: str) -> None:
    """Write a visually distinct section heading to the terminal."""

    logging.info(
        "[bold cyan]-- %s --[/bold cyan]",
        escape(title),
        extra=RICH_MARKUP,
    )


def display_split_statistics(split_stats: dict) -> None:
    """Display train/test statistics in a compact Rich table."""

    labels = {
        "training_rows": "Training rows",
        "test_rows": "Test rows",
        "training_patients": "Training patients",
        "test_patients": "Test patients",
        "training_esbl_prevalence": "Training ESBL prevalence",
        "test_esbl_prevalence": "Test ESBL prevalence",
    }

    table = Table(
        title="Train/Test Split Statistics",
        title_style="bold blue",
        header_style="bold cyan",
        border_style="blue",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right", style="green")

    for key, value in split_stats.items():
        display_value = (
            f"{value:.2%}" if key.endswith("_prevalence") else str(value)
        )
        table.add_row(labels.get(key, key.replace("_", " ").title()), display_value)

    Console().print(table)


def load_data(input_file, target_column=TARGET_COLUMN):
    """Read the input CSV file into a pandas DataFrame."""

    if not Path(input_file).exists():
        logging.error(f"Input file {input_file} does not exist.")
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    try:
        data = pd.read_csv(input_file)
        if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        logging.info(f"Data loaded successfully from {input_file}")
        return data
    
    except Exception as e:
        logging.error(f"Error reading data from {input_file}: {e}")
        raise 

def split_data(
    data,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    target_column=TARGET_COLUMN,
    split_type="random",
    cluster_column="subject",
    date_column="admission_date"
):
    """Split data into training and testing sets based on the specified split type.

    Ensures that data from the same cluster (e.g. patient) are not included
    in both training and testing sets.
    """

    if split_type == "random":

        # Create one representative row per patient for patient-level splitting
        unique_patients_data = (
            data.groupby(cluster_column)
            .first()
            .reset_index()
        )

        # Randomly split patients while stratifying by the target
        unique_patient_train_data, unique_patient_test_data = train_test_split(
            unique_patients_data,
            test_size=test_size,
            random_state=random_state,
            stratify=unique_patients_data[target_column]
        )

        # Restore all episodes belonging to patients assigned to each set
        train_data = data[
            data[cluster_column].isin(
                unique_patient_train_data[cluster_column]
            )
        ]

        test_data = data[
            data[cluster_column].isin(
                unique_patient_test_data[cluster_column]
            )
        ]

        logging.info(
            f"Data split randomly with test size {test_size}."
        )

    elif split_type == "temporal":

        if date_column not in data.columns:
            raise ValueError(
                f"Temporal split requires '{date_column}' to be present "
                "in the dataset."
            )

        # Convert the date column to datetime
        data = data.copy()
        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce"
        )

        # Check whether any dates failed to convert
        if data[date_column].isna().any():
            n_invalid = data[date_column].isna().sum()
            raise ValueError(
                f"{n_invalid} values in '{date_column}' could not be "
                "converted to datetime."
            )

        # Sort chronologically so that the first row for each patient
        # represents their earliest episode
        data = data.sort_values(by=date_column)

        unique_patients_data = (
            data.groupby(cluster_column)
            .first()
            .reset_index()
        )

        # Determine the temporal cutoff based on patients' first episodes
        cutoff_date = unique_patients_data[date_column].quantile(
            1 - test_size
        )

        unique_patient_train_data = unique_patients_data[
            unique_patients_data[date_column] <= cutoff_date
        ]

        unique_patient_test_data = unique_patients_data[
            unique_patients_data[date_column] > cutoff_date
        ]

        # Restore episodes only when they fall on the same side of the
        # temporal cutoff as the patient's assigned dataset
        train_data = data[
            data[cluster_column].isin(
                unique_patient_train_data[cluster_column]
            )
            & (data[date_column] <= cutoff_date)
        ]

        test_data = data[
            data[cluster_column].isin(
                unique_patient_test_data[cluster_column]
            )
            & (data[date_column] > cutoff_date)
        ]

        logging.info(
            f"Data split temporally at cutoff date {cutoff_date}."
        )

    else:
        raise ValueError(
            "Invalid split type. Choose 'random' or 'temporal'."
        )

    logging.info(f"Initial dataset size: {len(data)} rows")
    logging.info(f"Training set size: {len(train_data)} rows")
    logging.info(f"Testing set size: {len(test_data)} rows")
    logging.info(
        f"Number of episodes dropped: "
        f"{len(data) - (len(train_data) + len(test_data))} rows ({(len(data) - (len(train_data) + len(test_data))) / len(data) * 100:.2f}%)"
    )

    return train_data, test_data

def validate_split(data, train_data, test_data, cluster_column='subject', temporal_split=False, date_column='admission_date', target_column='esbl_status'):
    """Validate that no clusters (e.g., subjects) are present in both training and testing sets."""


    train_clusters = set(train_data[cluster_column].unique())
    test_clusters = set(test_data[cluster_column].unique())
    intersection = train_clusters.intersection(test_clusters)
    
    if intersection:
        logging.error(f"Data split validation failed. The following clusters are present in both training and testing sets: {intersection}")
        raise ValueError(f"Data split validation failed. The following clusters are present in both training and testing sets: {intersection}")
    else:
        logging.info("Data split validation passed. No clusters are present in both training and testing sets.")

    if not temporal_split:
        len_original = len(data)
        len_split = len(train_data) + len(test_data)
        if len_original != len_split:
            logging.error(f"Data split validation failed. The number of rows in the original dataset ({len_original}) does not match the sum of training and testing sets ({len_split}).")
            raise ValueError(f"Data split validation failed. The number of rows in the original dataset ({len_original}) does not match the sum of training and testing sets ({len_split}).")
    
    if temporal_split:
        if train_data[date_column].max() >= test_data[date_column].min():
            logging.error("Temporal split validation failed. Training data contains dates that are not earlier than testing data.")
            raise ValueError("Temporal split validation failed. Training data contains dates that are not earlier than testing data.")
        else:
            logging.info("Temporal split validation passed. All training data dates are earlier than testing data dates.")

    # Number of training rows
    training_rows = len(train_data)
    # Number of test rows
    test_rows = len(test_data)

    # Number of training patients
    training_patients = train_data[cluster_column].nunique()
    # Number of test patients
    test_patients = test_data[cluster_column].nunique()

    # ESBL prevalence training
    training_esbl_prevalence = train_data[target_column].mean()
    # ESBL prevalence test
    test_esbl_prevalence = test_data[target_column].mean()

    # return a dictionary with the statistics
    train_test_stats = {
        "training_rows": training_rows,
        "test_rows": test_rows,
        "training_patients": training_patients,
        "test_patients": test_patients,
        "training_esbl_prevalence": training_esbl_prevalence,
        "test_esbl_prevalence": test_esbl_prevalence
    }

    return train_test_stats

def save_split(train_data, test_data, output_dir=OUTPUT_DIRECTORY, split_type='random'):
    """Save the training and testing sets to CSV files in the specified output directory."""

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    train_file_path = Path(output_dir) / f"train_data_{split_type}.csv"
    test_file_path = Path(output_dir) / f"test_data_{split_type}.csv"

    try:
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)
        logging.info(f"Training data saved to {train_file_path}")
        logging.info(f"Testing data saved to {test_file_path}")


    except Exception as e:
        logging.error(f"Error saving split data: {e}")
        raise
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=INPUT_DATA_PATH, help="Path to the input CSV file")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Proportion of the dataset to include in the test split")
    parser.add_argument('--split_type', choices = ('random', 'temporal'), default='random', help="Type of split to perform: 'random' or 'temporal'")
    args = parser.parse_args()

    configure_terminal_logging()

    logging.info(
        "[bold blue]ESBL train/test data split[/bold blue]",
        extra=RICH_MARKUP,
    )
    logging.info("Input file : %s", args.input_file)
    logging.info("Split type : %s", args.split_type)
    logging.info("Test size  : %.0f%%", args.test_size * 100)

    log_section("Load data")
    data = load_data(args.input_file)

    log_section("Create train/test split")
    train_data, test_data = split_data(data, test_size=args.test_size, split_type=args.split_type)

    log_section("Validate split")
    split_stats = validate_split(data, train_data, test_data, temporal_split=(args.split_type == 'temporal'))

    display_split_statistics(split_stats)

    log_section("Save split")
    save_split(train_data, test_data, split_type=args.split_type)

    logging.info(
        "[bold green]Data split completed successfully.[/bold green]",
        extra=RICH_MARKUP,
    )


 

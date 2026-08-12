"""Summarise synthetic data using the real-data summary as a schema.

This local script intentionally has no dependency on the iCARE configuration
or utility modules. It writes tables with the same structure as the aggregate
real-data summaries, making straightforward comparisons possible.
"""

import argparse
from pathlib import Path

import pandas as pd

from summarise_real_data import (
    load_data,
    save_summary,
    summarise_categorical_columns,
    summarise_continuous_columns,
    summarise_dataset,
    summarise_dependencies,
    summarise_missingness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "synthetic_data.csv"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data" / "real_data_summary"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "synthetic_data_summary"
OUTCOME_COLUMN = "esbl_status"
PATIENT_ID_COLUMN = "subject"


def parse_args() -> argparse.Namespace:
    """Parse the small set of paths needed for local summarisation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--reference-summary-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Real-data summaries used only to identify variable groups.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def reference_variable_groups(reference_dir: Path) -> tuple[list[str], list[str]]:
    """Read continuous and categorical variable names from summary CSVs."""
    continuous_path = reference_dir / "continuous_summary.csv"
    categorical_path = reference_dir / "categorical_summary.csv"
    if not continuous_path.exists() or not categorical_path.exists():
        raise FileNotFoundError(
            "The reference directory must contain continuous_summary.csv and "
            "categorical_summary.csv."
        )

    continuous = (
        pd.read_csv(continuous_path)["variable"].drop_duplicates().tolist()
    )
    categorical = (
        pd.read_csv(categorical_path)["variable"].drop_duplicates().tolist()
    )
    return continuous, categorical


def main() -> None:
    """Create all synthetic-data summary tables."""
    args = parse_args()
    frame = load_data(args.input)
    continuous, categorical = reference_variable_groups(
        args.reference_summary_dir
    )

    expected = set(continuous + categorical + [PATIENT_ID_COLUMN, "admission_date"])
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(f"Synthetic dataset is missing columns: {missing}")

    # Binary variables are listed in the categorical summary but have numeric
    # values. They belong in both the frequency and dependency summaries.
    binary = [
        column
        for column in categorical
        if column != OUTCOME_COLUMN and pd.api.types.is_numeric_dtype(frame[column])
    ]

    summaries = {
        "dataset_summary.csv": summarise_dataset(
            frame, PATIENT_ID_COLUMN, OUTCOME_COLUMN
        ),
        "continuous_summary.csv": summarise_continuous_columns(
            frame, continuous
        ),
        "categorical_summary.csv": summarise_categorical_columns(
            frame, categorical
        ),
        "missingness_summary.csv": summarise_missingness(frame),
        "dependency_correlations.csv": summarise_dependencies(
            frame, continuous + binary
        ),
    }

    for filename, summary in summaries.items():
        save_summary(summary, filename, args.output_dir)

    print("\nSynthetic summary complete.")
    print(summaries["dataset_summary.csv"].to_string(index=False))


if __name__ == "__main__":
    main()

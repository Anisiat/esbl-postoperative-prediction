"""Summarise synthetic data using the real-data summary as a schema.

This local script intentionally has no dependency on the iCARE configuration
or utility modules. It writes tables with the same structure as the aggregate
real-data summaries, making straightforward comparisons possible.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "synthetic_data.csv"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data" / "real_data_summary"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "synthetic_data_summary"
OUTCOME_COLUMN = "esbl_status"
PATIENT_ID_COLUMN = "subject"
CATEGORICAL_PAIRS = (
    ("site", "organism_bug"),
    ("tfc", "prophylaxis_group"),
)


def load_data(path: Path) -> pd.DataFrame:
    """Load a non-empty synthetic CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The synthetic dataset is empty.")
    return frame


def summarise_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise dataset size, patients and outcome."""
    outcome = pd.to_numeric(frame[OUTCOME_COLUMN], errors="coerce")
    values = {
        "n_rows": len(frame),
        "n_columns": frame.shape[1],
        "n_unique_patients": frame[PATIENT_ID_COLUMN].nunique(dropna=True),
        "n_exact_duplicate_rows": int(frame.duplicated().sum()),
        "n_outcome_observed": int(outcome.notna().sum()),
        "n_outcome_positive": float(outcome.sum()),
        "outcome_prevalence": float(outcome.mean()),
    }
    return pd.DataFrame({"metric": values.keys(), "value": values.values()})


def summarise_continuous(
    frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Create the same continuous summary produced inside iCARE."""
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    summary = (
        numeric.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
        .T.reset_index()
        .rename(columns={
            "index": "variable", "5%": "p05", "25%": "q1",
            "50%": "median", "75%": "q3", "95%": "p95",
        })
    )
    summary["missing_count"] = summary["variable"].map(frame[columns].isna().sum())
    summary["missing_rate"] = summary["variable"].map(frame[columns].isna().mean())
    summary["skewness"] = summary["variable"].map(numeric.skew())
    return summary


def categorical_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Replace missing categories with the same explicit summary label."""
    return frame[column].astype("object").where(
        frame[column].notna(), "__MISSING__"
    )


def summarise_categorical(
    frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Summarise the frequency of every category."""
    summaries = []
    for column in columns:
        summary = (
            categorical_values(frame, column).value_counts(dropna=False)
            .rename_axis("category").reset_index(name="count")
        )
        summary.insert(0, "variable", column)
        summary["proportion"] = summary["count"] / len(frame)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def summarise_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Create joint count tables for the selected categorical pairs."""
    summaries = []
    for first, second in CATEGORICAL_PAIRS:
        pair = pd.DataFrame({
            first: categorical_values(frame, first),
            second: categorical_values(frame, second),
        })
        summary = (
            pair.value_counts(dropna=False).rename("count").reset_index()
            .rename(columns={first: "category_1", second: "category_2"})
        )
        summary.insert(0, "variable_2", second)
        summary.insert(0, "variable_1", first)
        summary["proportion"] = summary["count"] / len(frame)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def summarise_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise dtype and missingness for each column."""
    return pd.DataFrame({
        "variable": frame.columns,
        "dtype": frame.dtypes.astype(str).values,
        "missing_count": frame.isna().sum().values,
        "missing_rate": frame.isna().mean().values,
    }).sort_values(["missing_rate", "variable"], ascending=[False, True])


def summarise_dependencies(
    frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Return a square Spearman matrix including missingness indicators."""
    values: dict[str, pd.Series] = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values[column] = numeric
        if numeric.isna().any():
            values[f"__missing__{column}"] = numeric.isna().astype(int)
    correlation = pd.DataFrame(values).corr(method="spearman").fillna(0.0)
    for column in correlation.columns:
        correlation.loc[column, column] = 1.0
    return correlation.rename_axis("variable").reset_index()


def save_summary(summary: pd.DataFrame, filename: str, directory: Path) -> None:
    """Save a summary CSV."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")


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
        "dataset_summary.csv": summarise_dataset(frame),
        "continuous_summary.csv": summarise_continuous(frame, continuous),
        "categorical_summary.csv": summarise_categorical(frame, categorical),
        "categorical_pairwise_summary.csv": summarise_pairs(frame),
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

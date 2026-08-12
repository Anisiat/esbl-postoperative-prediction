"""Create aggregate summaries of the real ESBL modelling dataset.

Run this script inside the SDE after updating the paths below if necessary.

Expected project structure
--------------------------
project_root/
├── config/
│   └── config.yaml
├── data/
│   └── processed/
│       └── lr_data.csv
├── outputs/
└── scripts/
    └── summarise_data.py

The configuration file is expected to contain a ``logistic_regression``
section defining the outcome, cluster column, identifier columns, and
continuous, binary, and categorical predictors.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from utils import data_cleaning_tools as dct


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This assumes the script is stored in: project_root/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "lr_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_data_summary"

# These clinically useful pairs are disjoint, which lets the synthetic-data
# generator reproduce both tables exactly without a complex joint model.
CATEGORICAL_PAIRS = (
    ("site", "organism_bug"),
    ("tfc", "prophylaxis_group"),
)


# ---------------------------------------------------------------------------
# Loading and configuration
# ---------------------------------------------------------------------------

def load_data(input_path: Path) -> pd.DataFrame:
    """Load the processed logistic-regression modelling dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    return df


def get_column_groups(config: dict[str, Any]) -> dict[str, Any]:
    """Extract modelling column groups from the YAML configuration."""
    try:
        model_config = config["logistic_regression"]
        predictors = model_config["predictors"]

        return {
            "outcome": model_config["outcome"],
            "cluster": model_config["cluster_col"],
            "id_columns": model_config["id_columns"],
            "continuous": predictors["continuous"],
            "binary": predictors["binary"],
            "categorical": predictors["categorical"],
        }

    except KeyError as error:
        raise KeyError(
            f"Required configuration entry not found: {error}"
        ) from error


def as_list(value: str | list[str]) -> list[str]:
    """Return a string or list of strings as a list."""
    if isinstance(value, str):
        return [value]

    return list(value)


def validate_columns(
    df: pd.DataFrame,
    column_groups: dict[str, Any],
) -> None:
    """Check that all columns specified in the config exist in the data."""
    expected_columns = (
        column_groups["id_columns"]
        + column_groups["continuous"]
        + column_groups["binary"]
        + column_groups["categorical"]
        + as_list(column_groups["cluster"])
        + [column_groups["outcome"]]
    )

    missing_columns = sorted(
        set(expected_columns).difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "The following configured columns are missing from the dataset: "
            f"{missing_columns}"
        )


# ---------------------------------------------------------------------------
# Summary functions
# ---------------------------------------------------------------------------

def summarise_dataset(
    df: pd.DataFrame,
    patient_id_column: str,
    outcome_column: str,
) -> pd.DataFrame:
    """Produce high-level counts and outcome prevalence."""
    outcome_numeric = pd.to_numeric(
        df[outcome_column],
        errors="coerce",
    )

    summary = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_unique_patients": df[patient_id_column].nunique(dropna=True),
        "n_exact_duplicate_rows": int(df.duplicated().sum()),
        "n_outcome_observed": int(outcome_numeric.notna().sum()),
        "n_outcome_positive": float(outcome_numeric.sum()),
        "outcome_prevalence": float(outcome_numeric.mean()),
    }

    return pd.DataFrame(
        {
            "metric": list(summary.keys()),
            "value": list(summary.values()),
        }
    )


def summarise_continuous_columns(
    df: pd.DataFrame,
    continuous_columns: list[str],
) -> pd.DataFrame:
    """Calculate descriptive statistics for continuous predictors."""
    if not continuous_columns:
        return pd.DataFrame()

    numeric_df = df[continuous_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    summary = (
        numeric_df
        .describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "variable",
                "5%": "p05",
                "25%": "q1",
                "50%": "median",
                "75%": "q3",
                "95%": "p95",
            }
        )
    )

    missing_counts = df[continuous_columns].isna().sum()
    missing_rates = df[continuous_columns].isna().mean()
    skewness = numeric_df.skew()

    summary["missing_count"] = summary["variable"].map(
        missing_counts
    )
    summary["missing_rate"] = summary["variable"].map(
        missing_rates
    )
    summary["skewness"] = summary["variable"].map(
        skewness
    )

    return summary


def summarise_categorical_column(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Calculate counts and proportions for one categorical column."""
    values = df[column].astype("object").where(
        df[column].notna(),
        "__MISSING__",
    )

    summary = (
        values
        .value_counts(dropna=False)
        .rename_axis("category")
        .reset_index(name="count")
    )

    summary.insert(0, "variable", column)
    summary["proportion"] = summary["count"] / len(df)

    return summary


def summarise_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Summarise category counts and proportions across columns."""
    if not categorical_columns:
        return pd.DataFrame()

    summaries = [
        summarise_categorical_column(df, column)
        for column in categorical_columns
    ]

    return pd.concat(summaries, ignore_index=True)


def summarise_categorical_pairs(
    df: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """Return joint counts for selected categorical predictor pairs."""
    summaries: list[pd.DataFrame] = []
    for first, second in pairs:
        pair = df[[first, second]].astype("object").where(
            df[[first, second]].notna(),
            "__MISSING__",
        )
        summary = (
            pair.value_counts(dropna=False)
            .rename("count")
            .reset_index()
            .rename(columns={first: "category_1", second: "category_2"})
        )
        summary.insert(0, "variable_2", second)
        summary.insert(0, "variable_1", first)
        summary["proportion"] = summary["count"] / len(df)
        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True)


def summarise_missingness(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate missingness and dtype for every dataframe column."""
    summary = pd.DataFrame(
        {
            "variable": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_count": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
        }
    )

    return (
        summary
        .sort_values(
            ["missing_rate", "variable"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def summarise_dependencies(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """Export rank correlations without exporting patient-level values.

    Missingness indicators are included so that, for example, missing CRP and
    temperature values can occur together at approximately the real-data rate.
    Spearman correlation is used because it describes monotonic relationships
    without assuming that skewed clinical variables are normally distributed.
    """
    dependency_data: dict[str, pd.Series] = {}
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        dependency_data[column] = values
        if values.isna().any():
            dependency_data[f"__missing__{column}"] = values.isna().astype(int)

    correlation = pd.DataFrame(dependency_data).corr(method="spearman")
    correlation = correlation.fillna(0.0)
    for column in correlation.columns:
        correlation.loc[column, column] = 1.0

    # Keeping one variable per row and column makes the exported file directly
    # readable as a conventional square correlation matrix.
    return correlation.rename_axis("variable").reset_index()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_summary(
    summary: pd.DataFrame,
    filename: str,
    output_dir: Path,
) -> None:
    """Save one aggregate summary table as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    summary.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete real-data summary workflow."""
    print(f"Loading configuration from: {DEFAULT_CONFIG_PATH}")
    config = dct.load_config(DEFAULT_CONFIG_PATH)

    print(f"Loading modelling data from: {INPUT_PATH}")
    df = load_data(INPUT_PATH)

    column_groups = get_column_groups(config)
    validate_columns(df, column_groups)

    cluster_columns = as_list(column_groups["cluster"])
    patient_id_column = cluster_columns[0]
    outcome_column = column_groups["outcome"]

    dataset_summary = summarise_dataset(
        df=df,
        patient_id_column=patient_id_column,
        outcome_column=outcome_column,
    )

    continuous_summary = summarise_continuous_columns(
        df=df,
        continuous_columns=column_groups["continuous"],
    )

    categorical_summary = summarise_categorical_columns(
        df=df,
        categorical_columns=(
            column_groups["categorical"]
            + column_groups["binary"]
            + [outcome_column]
        ),
    )

    categorical_pair_summary = summarise_categorical_pairs(
        df=df,
        pairs=CATEGORICAL_PAIRS,
    )

    missingness_summary = summarise_missingness(df)

    dependency_summary = summarise_dependencies(
        df=df,
        numeric_columns=(
            column_groups["continuous"] + column_groups["binary"]
        ),
    )

    save_summary(
        dataset_summary,
        "dataset_summary.csv",
        OUTPUT_DIR,
    )

    save_summary(
        continuous_summary,
        "continuous_summary.csv",
        OUTPUT_DIR,
    )

    save_summary(
        categorical_summary,
        "categorical_summary.csv",
        OUTPUT_DIR,
    )

    save_summary(
        categorical_pair_summary,
        "categorical_pairwise_summary.csv",
        OUTPUT_DIR,
    )

    save_summary(
        missingness_summary,
        "missingness_summary.csv",
        OUTPUT_DIR,
    )

    save_summary(
        dependency_summary,
        "dependency_correlations.csv",
        OUTPUT_DIR,
    )

    print("\nSummary complete.")
    print(dataset_summary.to_string(index=False))


if __name__ == "__main__":
    main()

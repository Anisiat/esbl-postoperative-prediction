from pathlib import Path

import pandas as pd


INPUT_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "synthetic_data"
OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "synthetic_data_cleaned"
)

REQUIRED_COLUMNS = frozenset(
    {
        "subject",
        "medication_name_short",
        "therapeutical_class",
        "order_dt_tm",
        "admission_medicine_y_n",
        "gp_to_continue",
    }
)
MISSING_TEXT_VALUES = frozenset({"", "none", "null", "nan", "n/a", "na"})
TRUE_VALUES = frozenset({"y", "yes", "true", "1"})
FALSE_VALUES = frozenset({"n", "no", "false", "0"})


def _clean_optional_string(
    values: pd.Series,
    *,
    lowercase: bool = False,
) -> pd.Series:
    """Standardise a string column while preserving missing values."""

    cleaned = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    if lowercase:
        cleaned = cleaned.str.lower()

    is_missing = cleaned.str.lower().isin(MISSING_TEXT_VALUES)
    return cleaned.mask(is_missing, pd.NA)


def _clean_yes_no(values: pd.Series, *, column_name: str) -> pd.Series:
    """Convert common yes/no encodings to pandas nullable booleans."""

    cleaned = _clean_optional_string(values, lowercase=True)
    is_true = cleaned.isin(TRUE_VALUES)
    is_false = cleaned.isin(FALSE_VALUES)
    invalid = cleaned.notna() & ~is_true & ~is_false

    if invalid.any():
        unexpected = sorted(cleaned.loc[invalid].unique())
        raise ValueError(
            f"Unexpected values in {column_name}: {unexpected}; "
            "expected yes/no values"
        )

    result = pd.Series(pd.NA, index=cleaned.index, dtype="boolean")
    result.loc[is_true] = True
    result.loc[is_false] = False
    return result


def clean_prescriptions(prescriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Clean longitudinal prescribing records for feature engineering.

    Medication records remain supporting evidence and are not converted into
    Charlson conditions here. Temporal filtering and medicine-to-condition
    mapping belong in the downstream feature-building step.
    """

    df = prescriptions_df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if df.columns.duplicated().any():
        duplicates = sorted(df.columns[df.columns.duplicated()].unique())
        raise ValueError(
            f"Duplicate prescription columns after normalisation: {duplicates}"
        )

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required prescription columns: {missing_columns}"
        )

    df["subject"] = _clean_optional_string(df["subject"])
    if df["subject"].isna().any():
        raise ValueError("Prescription records contain missing subject identifiers")

    for column in ("medication_name_short", "therapeutical_class"):
        df[column] = _clean_optional_string(df[column], lowercase=True)

    df["order_dt_tm"] = pd.to_datetime(
        df["order_dt_tm"],
        errors="coerce",
    )

    n_missing_order_dates = df["order_dt_tm"].isna().sum()

    if n_missing_order_dates:
        print(
            f"{n_missing_order_dates} prescription records "
            "have missing order_dt_tm"
        )
        
    for column in ("admission_medicine_y_n", "gp_to_continue"):
        df[column] = _clean_yes_no(df[column], column_name=column)

    has_medication_information = ~(
        df["medication_name_short"].isna()
        & df["therapeutical_class"].isna()
    )
    df = df.loc[has_medication_information].copy()

    return df.drop_duplicates().reset_index(drop=True)


def main(
    input_directory: Path = INPUT_DIRECTORY,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> Path:
    """Clean the prescribing CSV and write the cleaned table."""

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    input_path = input_directory / "prescriptions.csv"
    output_path = output_directory / "prescriptions_cleaned.csv"

    prescriptions_df = pd.read_csv(
        input_path,
        dtype={
            "SUBJECT": "string",
            "MEDICATION_NAME_SHORT": "string",
            "THERAPEUTICAL_CLASS": "string",
            "ADMISSION_MEDICINE_Y_N": "string",
            "GP_TO_CONTINUE": "string",
        },
    )
    prescriptions_cleaned = clean_prescriptions(prescriptions_df)

    output_directory.mkdir(parents=True, exist_ok=True)
    prescriptions_cleaned.to_csv(output_path, index=False)

    print(f"Saved cleaned prescribing data to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()

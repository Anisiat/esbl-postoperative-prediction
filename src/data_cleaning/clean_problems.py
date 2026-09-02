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
        "problem_code",
        "problem_desc",
        "problem_dt_tm",
    }
)
MISSING_TEXT_VALUES = frozenset({"", "none", "null", "nan"})


def _clean_optional_string(values: pd.Series) -> pd.Series:
    """Strip a string column and standardise text missing-value markers."""

    cleaned = values.astype("string").str.strip()
    is_missing = cleaned.str.lower().isin(MISSING_TEXT_VALUES)
    return cleaned.mask(is_missing, pd.NA)


def clean_problems(problems_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise historical problem-list records.

    ``problem_code`` is treated as a SNOMED CT concept identifier and kept as
    text so it can be matched exactly to the long-format SNOMED mapping's
    ``snomed_code`` column without risking numeric precision loss.
    """

    df = problems_df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if df.columns.duplicated().any():
        duplicates = sorted(df.columns[df.columns.duplicated()].unique())
        raise ValueError(
            f"Duplicate problem columns after normalisation: {duplicates}"
        )

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required problem columns: {missing_columns}")

    df["subject"] = _clean_optional_string(df["subject"])
    df["problem_code"] = _clean_optional_string(df["problem_code"])
    df["problem_desc"] = _clean_optional_string(df["problem_desc"])


    df["problem_dt_tm"] = pd.to_datetime(
    df["problem_dt_tm"],
    errors="coerce",
    )

    if df["subject"].isna().any():
        raise ValueError(
            "Problem table contains missing subject identifiers."
        )

    df = df.dropna(
        subset=["problem_code", "problem_desc"],
        how="all",
    )

    return df.drop_duplicates().reset_index(drop=True)


def main(
    input_directory: Path = INPUT_DIRECTORY,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> Path:
    """Clean the problem-list CSV and write the cleaned table."""

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    input_path = input_directory / "problems.csv"
    output_path = output_directory / "problems_cleaned.csv"

    problems_df = pd.read_csv(
        input_path,
        dtype={
            "SUBJECT": "string",
            "PROBLEM_CODE": "string",
            "PROBLEM_DESC": "string",
        },
    )
    problems_cleaned = clean_problems(problems_df)

    output_directory.mkdir(parents=True, exist_ok=True)
    problems_cleaned.to_csv(output_path, index=False)

    print(f"Saved cleaned problem-list data to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()

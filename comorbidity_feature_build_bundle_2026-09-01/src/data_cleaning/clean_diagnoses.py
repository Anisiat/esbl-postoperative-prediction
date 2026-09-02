import pandas as pd
from pathlib import Path


INPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "synthetic_data"
)

OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "synthetic_data_cleaned"
)


def clean_diagnoses(
    diagnoses_df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and standardise episode diagnosis records."""

    df = diagnoses_df.copy()

    # Standardise column names
    df.columns = df.columns.str.lower()

    # Parse diagnosis date
    df["diagnosis_date"] = pd.to_datetime(
        df["diagnosis_date"],
        errors="coerce",
    )

    # Clean ICD-10 codes
    df["diagnosis_code_icd"] = (
        df["diagnosis_code_icd"]
        .astype("string")
        .str.upper()
        .str.strip()
        .str.replace(".", "", regex=False)
        .replace(
            {
                "NONE": pd.NA,
                "": pd.NA,
            }
        )
    )

    # Clean SNOMED codes
    df["diagnosis_code_snomed"] = (
        df["diagnosis_code_snomed"]
        .astype("string")
        .str.strip()
        .replace(
            {
                "none": pd.NA,
                "NONE": pd.NA,
                "": pd.NA,
            }
        )
    )

    # Clean descriptions for easier debugging / validation
    for col in [
        "diagnosis_desc_icd",
        "diagnosis_desc_snomed",
    ]:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "none": pd.NA,
                    "NONE": pd.NA,
                    "": pd.NA,
                }
            )
        )

    # Remove exact duplicate records
    df = df.drop_duplicates()

    return df


if __name__ == "__main__":

    diagnoses_df = pd.read_csv(
        INPUT_DIRECTORY / "diagnoses.csv",
        dtype={
            "DIAGNOSIS_CODE_ICD": "string",
            "DIAGNOSIS_CODE_SNOMED": "string",
            "SUBJECT": "string",
            "SPELL_IDENTIFIER": "string",
        },
    )

    diagnoses_cleaned = clean_diagnoses(
        diagnoses_df
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnoses_cleaned.to_csv(
        OUTPUT_DIRECTORY / "diagnoses_cleaned.csv",
        index=False,
    )
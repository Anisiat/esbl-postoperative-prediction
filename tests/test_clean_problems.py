"""Tests for cleaning historical problem-list records."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data_cleaning import clean_problems as cleaner


def _problem_frame(**overrides: object) -> pd.DataFrame:
    """Build a minimal valid problem-list table."""

    values: dict[str, object] = {
        "SUBJECT": ["P001"],
        "PROBLEM_CODE": ["44054006"],
        "PROBLEM_DESC": ["Type 2 diabetes mellitus"],
        "PROBLEM_DT_TM": ["2019-02-01 10:00:00"],
    }
    values.update(overrides)
    return pd.DataFrame(values)


def test_clean_problems_normalises_columns_and_string_fields() -> None:
    source = pd.DataFrame(
        {
            " SUBJECT ": [" P001 "],
            "PROBLEM_CODE": [" 1082601000119104 "],
            "Problem_Desc ": ["  Heart failure  "],
            "PROBLEM_DT_TM": ["2024-02-29 12:34:56"],
            "SOURCE_SYSTEM": ["clinical"],
        }
    )

    result = cleaner.clean_problems(source)

    assert result.columns.tolist() == [
        "subject",
        "problem_code",
        "problem_desc",
        "problem_dt_tm",
        "source_system",
    ]
    assert result.loc[0, "subject"] == "P001"
    assert result.loc[0, "problem_code"] == "1082601000119104"
    assert result.loc[0, "problem_desc"] == "Heart failure"
    assert result.loc[0, "source_system"] == "clinical"
    assert isinstance(result["subject"].dtype, pd.StringDtype)
    assert isinstance(result["problem_code"].dtype, pd.StringDtype)
    assert isinstance(result["problem_desc"].dtype, pd.StringDtype)


@pytest.mark.parametrize(
    "missing_value",
    [None, pd.NA, "", "   ", "none", " NONE ", "null", " Null ", "nan"],
)
def test_clean_problems_standardises_missing_string_values(
    missing_value: object,
) -> None:
    source = _problem_frame(
        SUBJECT=[missing_value],
        PROBLEM_CODE=[missing_value],
        PROBLEM_DESC=[missing_value],
    )

    result = cleaner.clean_problems(source)

    assert pd.isna(result.at[0, "subject"])
    assert pd.isna(result.at[0, "problem_code"])
    assert pd.isna(result.at[0, "problem_desc"])


def test_clean_problems_parses_dates_and_coerces_invalid_values() -> None:
    source = _problem_frame(
        SUBJECT=["P001", "P002", "P003"],
        PROBLEM_CODE=["44054006", "84114007", "13645005"],
        PROBLEM_DESC=["Diabetes", "Heart failure", "COPD"],
        PROBLEM_DT_TM=["2019-02-01 10:00:00", "not-a-date", None],
    )

    result = cleaner.clean_problems(source)

    assert result.at[0, "problem_dt_tm"] == pd.Timestamp("2019-02-01 10:00:00")
    assert pd.isna(result.at[1, "problem_dt_tm"])
    assert pd.isna(result.at[2, "problem_dt_tm"])
    assert result["problem_dt_tm"].dtype == "datetime64[ns]"


def test_clean_problems_keeps_integer_snomed_codes_as_exact_strings() -> None:
    source = _problem_frame(PROBLEM_CODE=[230690007])

    result = cleaner.clean_problems(source)

    assert result.at[0, "problem_code"] == "230690007"
    assert isinstance(result.at[0, "problem_code"], str)
    assert isinstance(result["problem_code"].dtype, pd.StringDtype)


def test_clean_problems_deduplicates_only_identical_cleaned_records() -> None:
    source = pd.DataFrame(
        {
            "SUBJECT": ["P001", " P001 ", "P001", "P001"],
            "PROBLEM_CODE": ["44054006"] * 4,
            "PROBLEM_DESC": [
                "Type 2 diabetes mellitus",
                " Type 2 diabetes mellitus ",
                "Diabetes mellitus",
                "Type 2 diabetes mellitus",
            ],
            "PROBLEM_DT_TM": [
                "2019-02-01 10:00:00",
                "2019-02-01 10:00:00",
                "2019-02-01 10:00:00",
                "2020-02-01 10:00:00",
            ],
        }
    )

    result = cleaner.clean_problems(source)

    assert len(result) == 3
    assert result.index.tolist() == [0, 1, 2]
    assert result["problem_desc"].tolist() == [
        "Type 2 diabetes mellitus",
        "Diabetes mellitus",
        "Type 2 diabetes mellitus",
    ]
    assert result["problem_dt_tm"].tolist() == [
        pd.Timestamp("2019-02-01 10:00:00"),
        pd.Timestamp("2019-02-01 10:00:00"),
        pd.Timestamp("2020-02-01 10:00:00"),
    ]


def test_clean_problems_does_not_mutate_its_input() -> None:
    source = _problem_frame(
        SUBJECT=[" P001 "],
        PROBLEM_CODE=[" 44054006 "],
        PROBLEM_DESC=[" Diabetes "],
    )
    original = source.copy(deep=True)

    cleaner.clean_problems(source)

    assert_frame_equal(source, original)


def test_clean_problems_rejects_missing_required_columns() -> None:
    source = _problem_frame().drop(columns="PROBLEM_CODE")

    with pytest.raises(
        ValueError,
        match=r"Missing required problem columns: \['problem_code'\]",
    ):
        cleaner.clean_problems(source)


def test_clean_problems_rejects_columns_that_collide_after_normalisation() -> None:
    source = _problem_frame()
    source[" problem_code "] = source["PROBLEM_CODE"]

    with pytest.raises(
        ValueError,
        match=r"Duplicate problem columns after normalisation: \['problem_code'\]",
    ):
        cleaner.clean_problems(source)


def test_cleaned_problem_codes_join_exactly_to_long_snomed_mapping() -> None:
    project_root = Path(__file__).resolve().parents[1]
    mapping = pd.read_csv(
        project_root / "configs" / "charlson_snomed_mapping.csv",
        dtype={"snomed_code": "string"},
    )
    source = _problem_frame(
        SUBJECT=["P001", "P002"],
        PROBLEM_CODE=[" 44054006 ", "38341003"],
        PROBLEM_DESC=["Type 2 diabetes mellitus", "Hypertension"],
        PROBLEM_DT_TM=["2019-02-01", "2024-05-20"],
    )

    cleaned = cleaner.clean_problems(source)
    result = cleaned.merge(
        mapping,
        left_on="problem_code",
        right_on="snomed_code",
        how="left",
        validate="many_to_many",
    )

    diabetes = result.loc[result["subject"].eq("P001")].iloc[0]
    hypertension = result.loc[result["subject"].eq("P002")].iloc[0]
    assert diabetes["snomed_code"] == "44054006"
    assert diabetes["comorbidity"] == "diabetes_without_complication"
    assert pd.isna(hypertension["snomed_code"])
    assert pd.isna(hypertension["comorbidity"])


def test_main_reads_identifiers_as_text_and_writes_cleaned_csv(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "nested" / "output"
    input_directory.mkdir()
    (input_directory / "problems.csv").write_text(
        "SUBJECT,PROBLEM_CODE,PROBLEM_DESC,PROBLEM_DT_TM\n"
        "0001,1082601000119104, Heart failure ,2020-08-12 15:30:00\n",
        encoding="utf-8",
    )

    output_path = cleaner.main(input_directory, output_directory)

    assert output_path == output_directory / "problems_cleaned.csv"
    assert output_path.is_file()
    result = pd.read_csv(
        output_path,
        dtype={"subject": "string", "problem_code": "string"},
    )
    assert result.columns.tolist() == [
        "subject",
        "problem_code",
        "problem_desc",
        "problem_dt_tm",
    ]
    assert result.at[0, "subject"] == "0001"
    assert result.at[0, "problem_code"] == "1082601000119104"
    assert result.at[0, "problem_desc"] == "Heart failure"

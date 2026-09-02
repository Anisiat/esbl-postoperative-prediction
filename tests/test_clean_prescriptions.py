"""Tests for cleaning longitudinal pharmacy prescribing records."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data_cleaning import clean_prescriptions as cleaner


def _prescription_frame(**overrides: object) -> pd.DataFrame:
    """Build a minimal valid prescribing table."""

    values: dict[str, object] = {
        "SUBJECT": ["P001"],
        "MEDICATION_NAME_SHORT": ["metformin"],
        "THERAPEUTICAL_CLASS": ["metabolic agents"],
        "ORDER_DT_TM": ["2021-10-01 08:00:00"],
        "ADMISSION_MEDICINE_Y_N": ["Y"],
        "GP_TO_CONTINUE": ["Y"],
    }
    values.update(overrides)
    return pd.DataFrame(values)


def test_clean_prescriptions_normalises_headers_and_text_fields() -> None:
    source = pd.DataFrame(
        {
            " SUBJECT ": [" P001 "],
            "Medication_Name_Short ": ["  Insulin   Soluble  "],
            "THERAPEUTICAL_CLASS": ["  Metabolic   Agents "],
            "ORDER_DT_TM": ["2024-02-29 12:34:56"],
            "ADMISSION_MEDICINE_Y_N": [" y "],
            "GP_TO_CONTINUE": [" N "],
            "SOURCE_SYSTEM": ["Pharmacy"],
        }
    )

    result = cleaner.clean_prescriptions(source)

    assert result.columns.tolist() == [
        "subject",
        "medication_name_short",
        "therapeutical_class",
        "order_dt_tm",
        "admission_medicine_y_n",
        "gp_to_continue",
        "source_system",
    ]
    assert result.at[0, "subject"] == "P001"
    assert result.at[0, "medication_name_short"] == "insulin soluble"
    assert result.at[0, "therapeutical_class"] == "metabolic agents"
    assert bool(result.at[0, "admission_medicine_y_n"])
    assert not bool(result.at[0, "gp_to_continue"])
    assert result.at[0, "source_system"] == "Pharmacy"
    assert isinstance(result["subject"].dtype, pd.StringDtype)
    assert isinstance(result["medication_name_short"].dtype, pd.StringDtype)
    assert isinstance(result["therapeutical_class"].dtype, pd.StringDtype)
    assert result["admission_medicine_y_n"].dtype == "boolean"
    assert result["gp_to_continue"].dtype == "boolean"


@pytest.mark.parametrize(
    "missing_value",
    [None, pd.NA, "", "   ", "none", " NONE ", "null", "nan", "n/a", "NA"],
)
def test_clean_prescriptions_standardises_missing_medication_text(
    missing_value: object,
) -> None:
    medication_missing = _prescription_frame(
        MEDICATION_NAME_SHORT=[missing_value],
        THERAPEUTICAL_CLASS=["metabolic agents"],
    )
    class_missing = _prescription_frame(
        MEDICATION_NAME_SHORT=["metformin"],
        THERAPEUTICAL_CLASS=[missing_value],
    )

    medication_result = cleaner.clean_prescriptions(medication_missing)
    class_result = cleaner.clean_prescriptions(class_missing)

    assert pd.isna(medication_result.at[0, "medication_name_short"])
    assert pd.isna(class_result.at[0, "therapeutical_class"])
    assert isinstance(
        medication_result["medication_name_short"].dtype,
        pd.StringDtype,
    )
    assert isinstance(
        class_result["therapeutical_class"].dtype,
        pd.StringDtype,
    )


@pytest.mark.parametrize(
    ("encoded_value", "expected"),
    [
        ("Y", True),
        (" y ", True),
        ("YES", True),
        (" true ", True),
        ("1", True),
        (1, True),
        (True, True),
        ("N", False),
        (" n ", False),
        ("NO", False),
        (" false ", False),
        ("0", False),
        (0, False),
        (False, False),
    ],
)
def test_clean_prescriptions_converts_supported_yes_no_encodings(
    encoded_value: object,
    expected: bool,
) -> None:
    source = _prescription_frame(
        ADMISSION_MEDICINE_Y_N=[encoded_value],
        GP_TO_CONTINUE=[encoded_value],
    )

    result = cleaner.clean_prescriptions(source)

    assert result.at[0, "admission_medicine_y_n"] == expected
    assert result.at[0, "gp_to_continue"] == expected
    assert result["admission_medicine_y_n"].dtype == "boolean"
    assert result["gp_to_continue"].dtype == "boolean"


@pytest.mark.parametrize(
    "missing_value",
    [None, pd.NA, "", "   ", "none", "null", "nan", "n/a", "NA"],
)
def test_clean_prescriptions_keeps_missing_yes_no_values_nullable(
    missing_value: object,
) -> None:
    source = _prescription_frame(
        ADMISSION_MEDICINE_Y_N=[missing_value],
        GP_TO_CONTINUE=[missing_value],
    )

    result = cleaner.clean_prescriptions(source)

    assert pd.isna(result.at[0, "admission_medicine_y_n"])
    assert pd.isna(result.at[0, "gp_to_continue"])
    assert result["admission_medicine_y_n"].dtype == "boolean"
    assert result["gp_to_continue"].dtype == "boolean"


@pytest.mark.parametrize(
    "column",
    ["ADMISSION_MEDICINE_Y_N", "GP_TO_CONTINUE"],
)
def test_clean_prescriptions_rejects_invalid_yes_no_values(column: str) -> None:
    source = _prescription_frame(**{column: ["perhaps"]})

    with pytest.raises(
        ValueError,
        match=rf"Unexpected values in {column.lower()}:.*perhaps",
    ):
        cleaner.clean_prescriptions(source)


def test_clean_prescriptions_parses_dates_and_coerces_invalid_values() -> None:
    source = _prescription_frame(
        SUBJECT=["P001", "P002", "P003"],
        MEDICATION_NAME_SHORT=["metformin", "furosemide", "salbutamol"],
        THERAPEUTICAL_CLASS=[
            "metabolic agents",
            "cardiovascular agents",
            "respiratory agents",
        ],
        ORDER_DT_TM=["2021-10-01 08:00:00", "not-a-date", None],
        ADMISSION_MEDICINE_Y_N=["Y", "N", None],
        GP_TO_CONTINUE=["Y", "N", None],
    )

    result = cleaner.clean_prescriptions(source)

    assert result.at[0, "order_dt_tm"] == pd.Timestamp("2021-10-01 08:00:00")
    assert pd.isna(result.at[1, "order_dt_tm"])
    assert pd.isna(result.at[2, "order_dt_tm"])
    assert result["order_dt_tm"].dtype == "datetime64[ns]"


def test_clean_prescriptions_drops_only_records_without_medication_information() -> None:
    source = _prescription_frame(
        SUBJECT=["P001", "P002", "P003", "P004"],
        MEDICATION_NAME_SHORT=["metformin", None, "furosemide", " none "],
        THERAPEUTICAL_CLASS=[None, "respiratory agents", "", " null "],
        ORDER_DT_TM=["2021-10-01"] * 4,
        ADMISSION_MEDICINE_Y_N=["Y"] * 4,
        GP_TO_CONTINUE=["Y"] * 4,
    )

    result = cleaner.clean_prescriptions(source)

    assert result["subject"].tolist() == ["P001", "P002", "P003"]
    assert result["medication_name_short"].tolist()[:1] == ["metformin"]
    assert pd.isna(result.at[1, "medication_name_short"])
    assert result.at[1, "therapeutical_class"] == "respiratory agents"
    assert pd.isna(result.at[2, "therapeutical_class"])


def test_clean_prescriptions_deduplicates_only_identical_cleaned_records() -> None:
    source = pd.DataFrame(
        {
            "SUBJECT": ["P001"] * 5,
            "MEDICATION_NAME_SHORT": [
                "Metformin",
                " metformin ",
                "metformin",
                "metformin",
                "metformin",
            ],
            "THERAPEUTICAL_CLASS": [
                "Metabolic Agents",
                " metabolic   agents ",
                "metabolic agents",
                "cardiovascular agents",
                "metabolic agents",
            ],
            "ORDER_DT_TM": [
                "2021-10-01 08:00:00",
                "2021-10-01 08:00:00",
                "2022-10-01 08:00:00",
                "2021-10-01 08:00:00",
                "2021-10-01 08:00:00",
            ],
            "ADMISSION_MEDICINE_Y_N": ["Y", "yes", "Y", "Y", "Y"],
            "GP_TO_CONTINUE": ["Y", "true", "Y", "Y", "N"],
        }
    )

    result = cleaner.clean_prescriptions(source)

    assert len(result) == 4
    assert result.index.tolist() == [0, 1, 2, 3]
    assert result["order_dt_tm"].tolist() == [
        pd.Timestamp("2021-10-01 08:00:00"),
        pd.Timestamp("2022-10-01 08:00:00"),
        pd.Timestamp("2021-10-01 08:00:00"),
        pd.Timestamp("2021-10-01 08:00:00"),
    ]
    assert result["therapeutical_class"].tolist() == [
        "metabolic agents",
        "metabolic agents",
        "cardiovascular agents",
        "metabolic agents",
    ]
    assert result["gp_to_continue"].tolist() == [True, True, True, False]


def test_clean_prescriptions_does_not_mutate_its_input() -> None:
    source = _prescription_frame(
        SUBJECT=[" P001 "],
        MEDICATION_NAME_SHORT=[" Metformin "],
        THERAPEUTICAL_CLASS=[" Metabolic   Agents "],
        ADMISSION_MEDICINE_Y_N=["yes"],
    )
    original = source.copy(deep=True)

    cleaner.clean_prescriptions(source)

    assert_frame_equal(source, original)


@pytest.mark.parametrize(
    "missing_subject",
    [None, pd.NA, "", "   ", "none", "null", "nan", "n/a", "NA"],
)
def test_clean_prescriptions_rejects_missing_subject_identifiers(
    missing_subject: object,
) -> None:
    source = _prescription_frame(SUBJECT=[missing_subject])

    with pytest.raises(
        ValueError,
        match="Prescription records contain missing subject identifiers",
    ):
        cleaner.clean_prescriptions(source)


def test_clean_prescriptions_rejects_missing_required_columns() -> None:
    source = _prescription_frame().drop(columns="GP_TO_CONTINUE")

    with pytest.raises(
        ValueError,
        match=r"Missing required prescription columns: \['gp_to_continue'\]",
    ):
        cleaner.clean_prescriptions(source)


def test_clean_prescriptions_rejects_headers_that_collide_after_normalisation() -> None:
    source = _prescription_frame()
    source[" medication_name_short "] = source["MEDICATION_NAME_SHORT"]

    with pytest.raises(
        ValueError,
        match=(
            r"Duplicate prescription columns after normalisation: "
            r"\['medication_name_short'\]"
        ),
    ):
        cleaner.clean_prescriptions(source)


def test_main_preserves_subject_text_and_writes_cleaned_csv(tmp_path: Path) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "nested" / "output"
    input_directory.mkdir()
    (input_directory / "prescriptions.csv").write_text(
        "SUBJECT,MEDICATION_NAME_SHORT,THERAPEUTICAL_CLASS,ORDER_DT_TM,"
        "ADMISSION_MEDICINE_Y_N,GP_TO_CONTINUE\n"
        "0001, Insulin   Soluble , Metabolic Agents ,2020-08-12 15:30:00,yes,0\n",
        encoding="utf-8",
    )

    output_path = cleaner.main(input_directory, output_directory)

    assert output_path == output_directory / "prescriptions_cleaned.csv"
    assert output_path.is_file()
    result = pd.read_csv(
        output_path,
        dtype={
            "subject": "string",
            "medication_name_short": "string",
            "therapeutical_class": "string",
        },
    )
    assert result.columns.tolist() == [
        "subject",
        "medication_name_short",
        "therapeutical_class",
        "order_dt_tm",
        "admission_medicine_y_n",
        "gp_to_continue",
    ]
    assert result.at[0, "subject"] == "0001"
    assert result.at[0, "medication_name_short"] == "insulin soluble"
    assert result.at[0, "therapeutical_class"] == "metabolic agents"
    assert result.at[0, "order_dt_tm"] == "2020-08-12 15:30:00"
    assert bool(result.at[0, "admission_medicine_y_n"])
    assert not bool(result.at[0, "gp_to_continue"])

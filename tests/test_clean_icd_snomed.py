"""Tests for validation and normalisation of Charlson code mappings."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data_cleaning import clean_icd_snomed_prescribing as cleaner


EXPECTED_COMORBIDITIES = frozenset(
    {
        "aids_hiv",
        "cerebrovascular_disease",
        "chronic_pulmonary_disease",
        "congestive_heart_failure",
        "dementia",
        "diabetes_with_complication",
        "diabetes_without_complication",
        "hemiplegia_or_paraplegia",
        "malignancy",
        "metastatic_solid_tumor",
        "mild_liver_disease",
        "moderate_or_severe_liver_disease",
        "myocardial_infarction",
        "peptic_ulcer_disease",
        "peripheral_vascular_disease",
        "renal_disease",
        "rheumatic_disease",
    }
)


def _mapping_with_categories(categories: set[str] | frozenset[str]) -> pd.DataFrame:
    """Build the minimum mapping accepted by category validation."""
    return pd.DataFrame({"comorbidity": sorted(categories)})


def _source_label(comorbidity: str, *, snomed: bool) -> str:
    """Select the source-specific spelling for one canonical category."""
    labels = [
        label
        for label, canonical in cleaner.CATEGORY_MAP.items()
        if canonical == comorbidity
    ]
    return labels[-1] if snomed else labels[0]


def test_expected_comorbidities_are_the_explicit_17_category_set() -> None:
    assert cleaner.EXPECTED_COMORBIDITIES == EXPECTED_COMORBIDITIES
    assert len(cleaner.EXPECTED_COMORBIDITIES) == 17


def test_main_validates_complete_categories_before_overwriting_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even matching input sets are invalid when both omit a canonical name."""
    incomplete = EXPECTED_COMORBIDITIES - {"aids_hiv"}
    monkeypatch.setattr(
        cleaner,
        "clean_quan_mapping",
        lambda _path: _mapping_with_categories(incomplete),
    )
    monkeypatch.setattr(
        cleaner,
        "clean_snomed_mapping",
        lambda _path: _mapping_with_categories(incomplete),
    )

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    icd_output = output_directory / "charlson_icd10_mapping.csv"
    snomed_output = output_directory / "charlson_snomed_mapping.csv"
    icd_output.write_text("existing ICD output\n", encoding="utf-8")
    snomed_output.write_text("existing SNOMED output\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"category validation failed:.*ICD-10 missing.*aids_hiv",
    ):
        cleaner.main(tmp_path / "input", output_directory)

    assert icd_output.read_text(encoding="utf-8") == "existing ICD output\n"
    assert snomed_output.read_text(encoding="utf-8") == "existing SNOMED output\n"


def test_category_validation_accepts_multiple_codes_per_category() -> None:
    icd_base = pd.DataFrame(
        {
            "comorbidity": sorted(EXPECTED_COMORBIDITIES),
            "icd_code": [
                f"code-{index}"
                for index in range(len(EXPECTED_COMORBIDITIES))
            ],
        }
    )
    snomed_base = pd.DataFrame(
        {
            "comorbidity": sorted(EXPECTED_COMORBIDITIES),
            "snomed_code": [
                f"code-{index}"
                for index in range(len(EXPECTED_COMORBIDITIES))
            ],
        }
    )
    icd_mapping = pd.concat(
        [
            icd_base,
            pd.DataFrame({"comorbidity": ["aids_hiv"], "icd_code": ["B20"]}),
        ],
        ignore_index=True,
    )
    snomed_mapping = pd.concat(
        [
            snomed_base,
            pd.DataFrame(
                {"comorbidity": ["aids_hiv"], "snomed_code": ["62479008"]}
            ),
        ],
        ignore_index=True,
    )

    cleaner.validate_comorbidity_names(icd_mapping, snomed_mapping)


@pytest.mark.parametrize("mapping_name", ["ICD-10", "SNOMED CT"])
def test_category_validation_rejects_duplicate_comorbidity_code_rows(
    mapping_name: str,
) -> None:
    code_column = "icd_code" if mapping_name == "ICD-10" else "snomed_code"
    mapping = pd.DataFrame(
        {
            "comorbidity": [*sorted(EXPECTED_COMORBIDITIES), "aids_hiv"],
            code_column: [
                *[f"code-{index}" for index in range(len(EXPECTED_COMORBIDITIES))],
                "code-0",
            ],
        }
    )
    # Select the actual code assigned to aids_hiv instead of relying on its
    # position in the sorted canonical set.
    aids_code = mapping.loc[
        mapping["comorbidity"].eq("aids_hiv"),
        code_column,
    ].iloc[0]
    mapping.loc[mapping.index[-1], code_column] = aids_code
    other_code_column = (
        "snomed_code" if mapping_name == "ICD-10" else "icd_code"
    )
    other_mapping = pd.DataFrame(
        {
            "comorbidity": sorted(EXPECTED_COMORBIDITIES),
            other_code_column: [
                f"other-{index}"
                for index in range(len(EXPECTED_COMORBIDITIES))
            ],
        }
    )

    icd_mapping, snomed_mapping = (
        (mapping, other_mapping)
        if mapping_name == "ICD-10"
        else (other_mapping, mapping)
    )

    with pytest.raises(
        ValueError,
        match=rf"{mapping_name} has duplicate comorbidity/code rows.*aids_hiv",
    ):
        cleaner.validate_comorbidity_names(icd_mapping, snomed_mapping)


def test_icd10_prefixes_are_normalised_deduplicated_and_returned_as_a_list() -> None:
    assert cleaner.clean_icd10_prefixes(
        " i25.2 | C0 | i25.2 | e11.9 "
    ) == ["I252", "C0", "E119"]


@pytest.mark.parametrize("value", [None, "I21||I22", "I21|not-a-code"])
def test_invalid_icd10_prefix_lists_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        cleaner.clean_icd10_prefixes(value)


def test_quan_mapping_explodes_prefixes_to_one_scalar_code_per_row(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "charlson_icd.csv"
    pd.DataFrame(
        {
            "category": [
                "Myocardial infarction",
                "Myocardial infarction",
                "Age",
            ],
            "icd10_codes": [" i25.2 | C0 | i25.2 ", "E11.9", "Z00"],
            "weights": [1, 1, 2],
        }
    ).to_csv(source_path, index=False)

    result = cleaner.clean_quan_mapping(source_path)

    assert result.columns.tolist() == [
        "comorbidity",
        "icd_code",
        "weight",
    ]
    assert result.to_dict("records") == [
        {
            "comorbidity": "myocardial_infarction",
            "icd_code": "I252",
            "weight": 1,
        },
        {
            "comorbidity": "myocardial_infarction",
            "icd_code": "C0",
            "weight": 1,
        },
        {
            "comorbidity": "myocardial_infarction",
            "icd_code": "E119",
            "weight": 1,
        },
    ]
    assert all(isinstance(code, str) for code in result["icd_code"])
    assert not result["icd_code"].str.contains(
        r"[,|\[\]\"]",
        regex=True,
    ).any()


def test_quan_mapping_rejects_conflicting_category_weights(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "charlson_icd.csv"
    pd.DataFrame(
        {
            "category": ["Myocardial infarction", "Myocardial infarction"],
            "icd10_codes": ["I21", "I22"],
            "weights": [1, 2],
        }
    ).to_csv(source_path, index=False)

    with pytest.raises(ValueError, match=r"Conflicting Quan weights.*myocardial"):
        cleaner.clean_quan_mapping(source_path)


def test_snomed_codes_remain_strings_and_require_valid_concept_sctids() -> None:
    assert cleaner.clean_snomed_code(22298006) == "22298006"
    assert cleaner.is_valid_sctid("22298006")

    with pytest.raises(ValueError, match="invalid check digit"):
        cleaner.clean_snomed_code("22298007")

    with pytest.raises(ValueError, match="stored as text"):
        cleaner.clean_snomed_code(22298006.0)

    description_body = "1234501"
    description_sctid = (
        description_body
        + cleaner.calculate_sctid_check_digit(description_body)
    )
    assert cleaner.is_valid_sctid(description_sctid)
    with pytest.raises(ValueError, match="not in a concept partition"):
        cleaner.clean_snomed_code(description_sctid)


def test_snomed_excel_rounding_repair_is_explicitly_audited(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_code = "12237231000119100"
    repaired_code = "12237231000119107"
    concept_id = "36712807"
    concept_name = "Right lower-limb atherosclerosis pain at rest"
    source = pd.DataFrame(
        {
            "Comorbid Condition": ["Peripheral vascular disease"],
            "Concept Code": [source_code],
            "Concept ID": [concept_id],
            "Concept Name": [f"  {concept_name}  "],
        }
    )
    monkeypatch.setattr(
        cleaner.pd,
        "read_excel",
        lambda *_args, **_kwargs: source.copy(),
    )

    with caplog.at_level(logging.WARNING):
        result = cleaner.clean_snomed_mapping(Path("mapping.xlsx"))

    assert result.columns.tolist() == [
        "comorbidity",
        "snomed_code",
    ]
    assert result.loc[0, "comorbidity"] == "peripheral_vascular_disease"
    assert result.at[0, "snomed_code"] == repaired_code
    assert isinstance(result.at[0, "snomed_code"], str)
    assert result.attrs["snomed_code_repairs"] == [
        {
            "source_code": source_code,
            "cleaned_code": repaired_code,
            "concept_id": concept_id,
            "source_row": 2,
            "concept_name": concept_name,
        }
    ]
    assert f"{source_code} -> {repaired_code}" in caplog.text

    assert cleaner.clean_snomed_code(
        source_code,
        concept_id=concept_id,
    ) == repaired_code
    for unaudited_concept_id in (None, "wrong-concept-id"):
        with pytest.raises(ValueError, match="requires OMOP concept"):
            cleaner.clean_snomed_code(
                source_code,
                concept_id=unaudited_concept_id,
            )

    with pytest.raises(ValueError, match="invalid check digit"):
        cleaner.clean_snomed_mapping(
            Path("mapping.xlsx"),
            repair_excel_rounding=False,
        )


def test_snomed_mapping_keeps_one_code_per_row_and_deduplicates_within_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct codes remain source ordered while duplicate pairs are removed."""
    source = pd.DataFrame(
        {
            "Comorbid Condition": [
                "Malignancy, except skin neoplasms",
                "Malignancy, except skin neoplasms",
                "Malignancy, except skin neoplasms",
            ],
            "Concept Code": ["372087000", "22298006", "372087000"],
            "Concept ID": ["1", "2", "1"],
            "Concept Name": [
                "Malignant neoplasm",
                "Myocardial infarction",
                "Malignant neoplasm",
            ],
        }
    )
    monkeypatch.setattr(
        cleaner.pd,
        "read_excel",
        lambda *_args, **_kwargs: source.copy(),
    )

    result = cleaner.clean_snomed_mapping(Path("mapping.xlsx"))

    assert result.columns.tolist() == [
        "comorbidity",
        "snomed_code",
    ]
    assert result.to_dict("records") == [
        {
            "comorbidity": "malignancy",
            "snomed_code": "372087000",
        },
        {
            "comorbidity": "malignancy",
            "snomed_code": "22298006",
        },
    ]
    assert all(isinstance(code, str) for code in result["snomed_code"])
    assert not result["snomed_code"].str.contains(
        r"[,|\[\]\"]",
        regex=True,
    ).any()


def test_primary_malignancy_is_not_mapped_to_metastatic_solid_tumor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary malignancy must not be treated as metastatic disease."""
    source = pd.DataFrame(
        {
            "Comorbid Condition": [
                "Malignancy, except skin neoplasms",
                "Metastatic solid tumor",
            ],
            "Concept Code": ["372087000", "372087000"],
            "Concept ID": ["1", "1"],
            "Concept Name": ["Malignant neoplasm", "Malignant neoplasm"],
        }
    )
    monkeypatch.setattr(
        cleaner.pd,
        "read_excel",
        lambda *_args, **_kwargs: source.copy(),
    )

    result = cleaner.clean_snomed_mapping(Path("mapping.xlsx"))

    assert result["comorbidity"].tolist() == ["malignancy"]
    assert result["snomed_code"].tolist() == ["372087000"]


def test_generic_hiv_cns_disorder_is_not_mapped_to_dementia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame(
        {
            "Comorbid Condition": ["AIDS/HIV", "Dementia"],
            "Concept Code": ["713571008", "713571008"],
            "Concept ID": ["37017319", "37017319"],
            "Concept Name": [
                "CNS disorder co-occurrent with HIV infection",
                "CNS disorder co-occurrent with HIV infection",
            ],
        }
    )
    monkeypatch.setattr(
        cleaner.pd,
        "read_excel",
        lambda *_args, **_kwargs: source.copy(),
    )

    result = cleaner.clean_snomed_mapping(Path("mapping.xlsx"))

    assert result.to_dict("records") == [
        {
            "comorbidity": "aids_hiv",
            "snomed_code": "713571008",
        }
    ]


def test_legitimate_cross_category_snomed_code_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A combined condition may legitimately support two categories."""
    source = pd.DataFrame(
        {
            "Comorbid Condition": [
                "Congestive heart failure",
                "Renal disease",
            ],
            "Concept Code": ["194781004", "194781004"],
            "Concept ID": ["1", "1"],
            "Concept Name": [
                "Hypertensive heart and renal disease with heart and renal failure",
                "Hypertensive heart and renal disease with heart and renal failure",
            ],
        }
    )
    monkeypatch.setattr(
        cleaner.pd,
        "read_excel",
        lambda *_args, **_kwargs: source.copy(),
    )

    result = cleaner.clean_snomed_mapping(Path("mapping.xlsx"))

    assert result["comorbidity"].tolist() == [
        "congestive_heart_failure",
        "renal_disease",
    ]
    assert result["snomed_code"].tolist() == ["194781004", "194781004"]


def test_unreviewed_cross_category_snomed_code_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame(
        {
            "Comorbid Condition": [
                "Congestive heart failure",
                "Renal disease",
            ],
            "Concept Code": ["22298006", "22298006"],
            "Concept ID": ["1", "1"],
            "Concept Name": ["Unreviewed compound", "Unreviewed compound"],
        }
    )
    monkeypatch.setattr(
        cleaner.pd,
        "read_excel",
        lambda *_args, **_kwargs: source.copy(),
    )

    with pytest.raises(ValueError, match="unreviewed CCI category overlaps"):
        cleaner.clean_snomed_mapping(Path("mapping.xlsx"))


def test_snomed_codes_round_trip_as_separate_scalar_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_sctid = "12237231000119107"
    source = pd.DataFrame(
        {
            "Comorbid Condition": [
                "Peripheral vascular disease",
                "Peripheral vascular disease",
            ],
            "Concept Code": ["22298006", long_sctid],
            "Concept ID": ["1", "36712807"],
            "Concept Name": [
                "Myocardial infarction",
                "Right lower-limb atherosclerosis pain at rest",
            ],
        }
    )
    monkeypatch.setattr(
        cleaner.pd,
        "read_excel",
        lambda *_args, **_kwargs: source.copy(),
    )

    mapping = cleaner.clean_snomed_mapping(Path("mapping.xlsx"))
    output_path = tmp_path / "mapping.csv"
    mapping.to_csv(output_path, index=False)
    reloaded = pd.read_csv(output_path, dtype={"snomed_code": "string"})

    assert reloaded.columns.tolist() == ["comorbidity", "snomed_code"]
    assert reloaded["snomed_code"].tolist() == ["22298006", long_sctid]
    assert all(isinstance(code, str) for code in reloaded["snomed_code"])
    assert all(cleaner.is_valid_sctid(code) for code in reloaded["snomed_code"])
    assert not reloaded["snomed_code"].str.contains(
        r"[,|\[\]\"]",
        regex=True,
    ).any()
    assert reloaded.at[1, "snomed_code"] == long_sctid


def test_main_cleans_and_writes_all_mappings_to_requested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()

    quan_source = pd.DataFrame(
        {
            "category": [
                _source_label(category, snomed=False)
                for category in sorted(EXPECTED_COMORBIDITIES)
            ],
            "icd10_codes": [" i25.2 | c0 "] * len(EXPECTED_COMORBIDITIES),
            "weights": [1] * len(EXPECTED_COMORBIDITIES),
        }
    )
    quan_source.to_csv(input_directory / "charlson_icd.csv", index=False)

    snomed_codes = []
    for index in range(len(EXPECTED_COMORBIDITIES)):
        identifier_body = f"{100000 + index}00"
        snomed_codes.append(
            identifier_body
            + cleaner.calculate_sctid_check_digit(identifier_body)
        )

    snomed_source = pd.DataFrame(
        {
            "Comorbid Condition": [
                _source_label(category, snomed=True)
                for category in sorted(EXPECTED_COMORBIDITIES)
            ],
            "Concept Code": snomed_codes,
            "Concept ID": range(1, len(EXPECTED_COMORBIDITIES) + 1),
            "Concept Name": [
                f"Concept for {category}"
                for category in sorted(EXPECTED_COMORBIDITIES)
            ],
        }
    )

    def fake_read_excel(path: Path, **kwargs: object) -> pd.DataFrame:
        assert Path(path) == input_directory / "charlson_snomed.xlsx"
        assert kwargs["dtype"] == {"Concept Code": "string"}
        return snomed_source.copy()

    monkeypatch.setattr(cleaner.pd, "read_excel", fake_read_excel)

    icd_path, snomed_path, medication_path = cleaner.main(
        input_directory,
        output_directory,
    )

    assert icd_path == output_directory / "charlson_icd10_mapping.csv"
    assert snomed_path == output_directory / "charlson_snomed_mapping.csv"
    assert medication_path == output_directory / "charlson_medication_mapping.csv"
    assert icd_path.is_file()
    assert snomed_path.is_file()
    assert medication_path.is_file()

    icd_result = pd.read_csv(icd_path, dtype={"icd_code": "string"})
    snomed_result = pd.read_csv(snomed_path, dtype={"snomed_code": "string"})
    medication_result = pd.read_csv(medication_path, dtype="string")

    assert set(icd_result["comorbidity"]) == EXPECTED_COMORBIDITIES
    assert set(snomed_result["comorbidity"]) == EXPECTED_COMORBIDITIES
    assert icd_result.columns.tolist() == [
        "comorbidity",
        "icd_code",
        "weight",
    ]
    assert snomed_result.columns.tolist() == [
        "comorbidity",
        "snomed_code",
    ]
    assert medication_result.columns.tolist() == [
        "comorbidity",
        "medication_name",
    ]
    assert len(icd_result) == 2 * len(EXPECTED_COMORBIDITIES)
    assert len(snomed_result) == len(EXPECTED_COMORBIDITIES)
    assert set(icd_result["icd_code"]) == {"I252", "C0"}
    assert snomed_result["snomed_code"].tolist() == snomed_codes
    assert all(isinstance(code, str) for code in icd_result["icd_code"])
    assert all(isinstance(code, str) for code in snomed_result["snomed_code"])
    assert not icd_result["icd_code"].str.contains(
        r"[,|\[\]\"]",
        regex=True,
    ).any()
    assert not snomed_result["snomed_code"].str.contains(
        r"[,|\[\]\"]",
        regex=True,
    ).any()
    assert all(
        cleaner.is_valid_sctid(code)
        for code in snomed_result["snomed_code"]
    )
    pd.testing.assert_frame_equal(
        medication_result,
        cleaner.build_medication_mapping().astype("string"),
    )

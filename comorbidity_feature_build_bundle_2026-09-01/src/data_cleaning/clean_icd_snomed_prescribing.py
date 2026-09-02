import logging
import re
from pathlib import Path

import pandas as pd

"""
Charlson comorbidity mapping sources.

ICD-10:
    Quan et al. (2005), Coding algorithms for defining comorbidities
    in ICD-9-CM and ICD-10 administrative data.

SNOMED CT:
    Charlson SNOMED mappings extracted from the comorbidity/comorbpy
    implementation and clinically reviewed against SNOMED CT terminology.
    Primary malignant neoplasm (372087000) is mapped to malignancy only,
    not metastatic solid tumor.

Medication:
    Manually curated supplementary mapping of relatively specific chronic
    medications to selected Charlson conditions. Medication evidence is
    supportive only and does not distinguish disease severity/complications.
"""


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIRECTORY = PROJECT_ROOT / "configs"
OUTPUT_DIRECTORY = PROJECT_ROOT / "configs"


# =============================================================================
# Canonical Charlson category names
# =============================================================================

CATEGORY_MAP = {
    # Shared / near-identical names
    "Myocardial infarction": "myocardial_infarction",
    "Congestive heart failure": "congestive_heart_failure",
    "Peripheral vascular disease": "peripheral_vascular_disease",
    "Cerebrovascular disease": "cerebrovascular_disease",
    "Dementia": "dementia",
    "Chronic pulmonary disease": "chronic_pulmonary_disease",
    "Rheumatic disease": "rheumatic_disease",
    "Peptic ulcer disease": "peptic_ulcer_disease",
    "Mild liver disease": "mild_liver_disease",
    "Hemiplegia or paraplegia": "hemiplegia_or_paraplegia",
    "Renal disease": "renal_disease",
    "Moderate or severe liver disease": "moderate_or_severe_liver_disease",
    "Metastatic solid tumor": "metastatic_solid_tumor",
    "AIDS/HIV": "aids_hiv",

    # ICD/Quan wording
    "Diabetes without chronic complication":
        "diabetes_without_complication",
    "Diabetes with chronic complication":
        "diabetes_with_complication",
    "Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin":
        "malignancy",

    # SNOMED wording
    "Diabetes, without chronic complications":
        "diabetes_without_complication",
    "Diabetes, with chronic complications":
        "diabetes_with_complication",
    "Malignancy, except skin neoplasms":
        "malignancy",
}

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

# =============================================================================
# Medication mapping
# =============================================================================

MEDICATION_COMORBIDITY_MAP = {
    "chronic_pulmonary_disease": [
        # Maintenance inhaled corticosteroids / combination inhalers
        "beclometasone",
        "budesonide",
        "fluticasone",
        "beclometasone-formoterol",
        "budesonide-formoterol",
        "fluticasone-salmeterol",
        "fluticasone-vilanterol",
        "fluticasone-formoterol",

        # LAMA / COPD maintenance therapy
        "tiotropium",
        "glycopyrronium",
        "aclidinium",
        "umeclidinium",
        "glycopyrronium-indacaterol",
        "umeclidinium-vilanterol",
        "fluticasone/umeclidinium/vilanterol",
        "beclometasone/formoterol/glycopyrronium",
    ],

    "diabetes_without_complication": [
        # Relatively diabetes-specific therapies
        "gliclazide",
        "linagliptin",
        "sitagliptin",
        "alogliptin",
    ],
}


# Quan codes are deliberately prefixes rather than necessarily complete ICD-10
# codes. Two-character prefixes such as C0 and C1 therefore remain valid.
ICD10_PREFIX_PATTERN = re.compile(r"^[A-Z][0-9][A-Z0-9]{0,5}$")
SCTID_PATTERN = re.compile(r"^[1-9][0-9]{5,17}$")
SNOMED_CONCEPT_PARTITIONS = frozenset({"00", "10"})

# Explicitly reject clinically incorrect category assignments present in the
# source workbook. Concept 372087000 is Primary malignant neoplasm and supports
# the general malignancy category, not metastatic solid tumour.
SNOMED_CATEGORY_CODE_EXCLUSIONS = frozenset(
    {
        ("metastatic_solid_tumor", "372087000"),
    }
)


# The source workbook stored these SCTIDs as numeric Excel cells, which rounded
# away their check digits. Keeping the observed source -> repaired pairs
# explicit prevents an unrelated typo from being "repaired" into a different,
# checksum-valid identifier. New invalid values must be reviewed and added
# deliberately rather than guessed at runtime.
EXCEL_ROUNDED_SNOMED_REPAIRS = {
    "1082601000119100": ("46269816", "1082601000119104"),
    "1082621000119100": ("46269818", "1082621000119108"),
    "1085021000119100": ("46269835", "1085021000119106"),
    "1085091000119100": ("46269836", "1085091000119108"),
    "1092801000119100": ("46273476", "1092801000119102"),
    "10692681000119100": ("46269801", "10692681000119108"),
    "10708511000119100": ("45770892", "10708511000119108"),
    "1073711000119100": ("42534834", "1073711000119105"),
    "1073721000119100": ("42534835", "1073721000119103"),
    "1073741000119100": ("35609009", "1073741000119109"),
    "1073751000119100": ("36685020", "1073751000119106"),
    "1073771000119100": ("36685022", "1073771000119102"),
    "1073791000119100": ("42534836", "1073791000119101"),
    "1073811000119100": ("37108591", "1073811000119102"),
    "1073821000119100": ("35609010", "1073821000119109"),
    "1073831000119100": ("36685024", "1073831000119107"),
    "12236951000119100": ("36717006", "12236951000119108"),
    "12236991000119100": ("35611566", "12236991000119103"),
    "12237111000119100": ("36712805", "12237111000119107"),
    "12237191000119100": ("36712806", "12237191000119103"),
    "12237231000119100": ("36712807", "12237231000119107"),
    "15649901000119100": ("36717279", "15649901000119104"),
    "15649941000119100": ("35615028", "15649941000119102"),
    "15649991000119100": ("36712963", "15649991000119105"),
    "15713081000119100": ("46270162", "15713081000119108"),
    "15713121000119100": ("46270163", "15713121000119105"),
    "15928141000119100": ("36687122", "15928141000119107"),
    "16276361000119100": ("37109056", "16276361000119109"),
    "16837681000119100": ("37309626", "16837681000119104"),
}


# Tables from the Verhoeff algorithm used for the mandatory SCTID check digit.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INVERSE = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def clean_icd10_prefixes(value: object) -> list[str]:
    """Normalise a source value into distinct Quan ICD-10 prefixes."""

    if pd.isna(value):
        raise ValueError("ICD-10 prefix list is missing")

    prefixes = []
    for raw_prefix in str(value).split("|"):
        prefix = re.sub(r"\s+", "", raw_prefix).upper().replace(".", "")

        if not prefix:
            raise ValueError(f"ICD-10 prefix list contains an empty value: {value!r}")

        if not ICD10_PREFIX_PATTERN.fullmatch(prefix):
            raise ValueError(
                f"Invalid ICD-10 prefix {raw_prefix!r} after normalisation "
                f"to {prefix!r}"
            )

        if prefix not in prefixes:
            prefixes.append(prefix)

    return prefixes


def calculate_sctid_check_digit(identifier_body: str) -> str:
    """Calculate the Verhoeff check digit for an SCTID body."""

    if not identifier_body or not identifier_body.isdigit():
        raise ValueError("An SCTID body must contain digits only")

    checksum = 0
    for position, digit in enumerate(reversed(identifier_body)):
        checksum = _VERHOEFF_D[checksum][
            _VERHOEFF_P[(position + 1) % 8][int(digit)]
        ]

    return str(_VERHOEFF_INVERSE[checksum])


def is_valid_sctid(code: str) -> bool:
    """Return whether a value has a valid SCTID shape and check digit."""

    if not isinstance(code, str) or not SCTID_PATTERN.fullmatch(code):
        return False

    checksum = 0
    for position, digit in enumerate(reversed(code)):
        checksum = _VERHOEFF_D[checksum][
            _VERHOEFF_P[position % 8][int(digit)]
        ]

    return checksum == 0


def clean_snomed_code(
    value: object,
    *,
    concept_id: object = None,
    repair_excel_rounding: bool = True,
) -> str:
    """Clean and validate one SNOMED CT concept identifier.

    Excel preserves only 15 significant digits in numeric cells. The known
    rounded values from this source workbook can be repaired using the
    explicit ``EXCEL_ROUNDED_SNOMED_REPAIRS`` audit mapping. Other malformed
    identifiers are never guessed or repaired.
    """

    if pd.isna(value):
        raise ValueError("SNOMED CT concept identifier is missing")

    code = str(value).strip()

    if not SCTID_PATTERN.fullmatch(code):
        raise ValueError(
            "SNOMED CT concept identifiers must contain 6 to 18 digits, "
            f"must not start with zero, and must be stored as text; got {code!r}"
        )

    partition = code[-3:-1]
    if partition not in SNOMED_CONCEPT_PARTITIONS:
        raise ValueError(
            f"SNOMED CT identifier {code!r} is not in a concept partition"
        )

    if is_valid_sctid(code):
        return code

    if repair_excel_rounding and code in EXCEL_ROUNDED_SNOMED_REPAIRS:
        expected_concept_id, repaired = EXCEL_ROUNDED_SNOMED_REPAIRS[code]
        supplied_concept_id = None if pd.isna(concept_id) else str(concept_id).strip()
        if supplied_concept_id != expected_concept_id:
            raise ValueError(
                f"Audited SNOMED repair for {code!r} requires OMOP concept "
                f"{expected_concept_id}, got {supplied_concept_id!r}"
            )
        calculated = code[:-1] + calculate_sctid_check_digit(code[:-1])
        if repaired != calculated or not is_valid_sctid(repaired):
            raise RuntimeError(
                f"Invalid audited SNOMED repair configured for {code!r}"
            )
        return repaired

    raise ValueError(f"SNOMED CT identifier {code!r} has an invalid check digit")


# =============================================================================
# Quan ICD-10 mapping
# =============================================================================

def clean_quan_mapping(
    input_path: Path,
) -> pd.DataFrame:
    """Clean Quan ICD-10 Charlson mapping."""

    df = pd.read_csv(input_path)

    # Age is not a comorbidity category
    df = df[df["category"] != "Age"].copy()

    df["comorbidity"] = (
        df["category"]
        .astype("string")
        .str.strip()
        .map(CATEGORY_MAP)
    )

    if df["comorbidity"].isna().any():
        missing = df.loc[
            df["comorbidity"].isna(),
            "category",
        ].unique()

        raise ValueError(
            f"Unmapped Quan categories: {missing}"
        )

    # Keep only fields needed downstream and expand the source prefix list so
    # every output row contains exactly one ICD-10 prefix.
    df = df[
        [
            "comorbidity",
            "icd10_codes",
            "weights",
        ]
    ].copy()

    weight_counts = df.groupby("comorbidity")["weights"].nunique(dropna=False)
    conflicting_weights = sorted(weight_counts[weight_counts > 1].index)
    if conflicting_weights:
        raise ValueError(
            "Conflicting Quan weights for comorbidities: "
            f"{conflicting_weights}"
        )

    df["icd_code"] = df["icd10_codes"].map(clean_icd10_prefixes)
    df = df.explode(
        "icd_code",
        ignore_index=True,
    )
    df = df.rename(columns={"weights": "weight"})

    df = df.drop_duplicates(
        subset=["comorbidity", "icd_code"],
        keep="first",
    ).reset_index(drop=True)

    # Reorder columns
    df = df[
        [
            "comorbidity",
            "icd_code",
            "weight",
        ]
    ]

    return df


# =============================================================================
# SNOMED mapping
# =============================================================================

def clean_snomed_mapping(
    input_path: Path,
    *,
    repair_excel_rounding: bool = True,
) -> pd.DataFrame:
    """Clean SNOMED mapping into one exact concept code per row."""

    df = pd.read_excel(
        input_path,
        dtype={
            "Concept Code": "string",
        },
    )

    df["comorbidity"] = (
        df["Comorbid Condition"]
        .str.strip()
        .map(CATEGORY_MAP)
    )

    if df["comorbidity"].isna().any():
        missing = df.loc[
            df["comorbidity"].isna(),
            "Comorbid Condition",
        ].drop_duplicates().tolist()

        raise ValueError(
            f"Unmapped SNOMED categories: {missing}"
        )

    source_codes = (
        df["Concept Code"]
        .astype("string")
        .str.strip()
    )

    cleaned_codes = []
    repairs = []
    errors = []

    for row_number, source_code in source_codes.items():
        try:
            cleaned_code = clean_snomed_code(
                source_code,
                concept_id=df.at[row_number, "Concept ID"],
                repair_excel_rounding=repair_excel_rounding,
            )
        except ValueError as error:
            errors.append(f"row {row_number + 2}: {error}")
            cleaned_codes.append(pd.NA)
            continue

        cleaned_codes.append(cleaned_code)

        if cleaned_code != source_code:
            repairs.append(
                {
                    "source_code": source_code,
                    "cleaned_code": cleaned_code,
                    "concept_id": str(df.at[row_number, "Concept ID"]).strip(),
                    "source_row": row_number + 2,
                    "concept_name": str(df.at[row_number, "Concept Name"]).strip(),
                }
            )

    if errors:
        preview = "; ".join(errors[:10])
        remainder = len(errors) - 10
        if remainder:
            preview += f"; and {remainder} more"
        raise ValueError(f"Invalid SNOMED CT identifiers: {preview}")

    df["snomed_code"] = pd.Series(
        cleaned_codes,
        index=df.index,
        dtype="string",
    )

    excluded_category_codes = pd.Series(False, index=df.index)
    for comorbidity, code in SNOMED_CATEGORY_CODE_EXCLUSIONS:
        excluded_category_codes |= (
            df["comorbidity"].eq(comorbidity)
            & df["snomed_code"].eq(code)
        )

    if excluded_category_codes.any():
        logging.warning(
            "Excluded %d clinically invalid SNOMED category/code mapping(s).",
            int(excluded_category_codes.sum()),
        )
        df = df.loc[~excluded_category_codes].copy()

    df["concept_name"] = (
        df["Concept Name"]
        .astype("string")
        .str.strip()
    )

    if df["concept_name"].isna().any() or df["concept_name"].eq("").any():
        raise ValueError("SNOMED mapping contains missing concept names")

    # Keep only fields needed downstream.
    df = df[
        [
            "comorbidity",
            "snomed_code",
            "concept_name",
        ]
    ].copy()

    # A code may intentionally belong to more than one Charlson category, so
    # deduplicate only within each category. Conflicting names for the same
    # category/code pair are rejected rather than silently discarding one.
    name_counts = (
        df.groupby(
            ["comorbidity", "snomed_code"],
            dropna=False,
        )["concept_name"]
        .nunique(dropna=False)
    )
    conflicts = name_counts[name_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            "Conflicting SNOMED concept names for category/code pairs: "
            f"{list(conflicts.index)}"
        )

    df = df.drop_duplicates(
        subset=["comorbidity", "snomed_code"],
        keep="first",
    ).reset_index(drop=True)
    df = df[["comorbidity", "snomed_code"]].copy()
    df.attrs["snomed_code_repairs"] = repairs

    if repairs:
        print(
            f"Repaired {len(repairs)} SNOMED identifiers "
            "affected by known Excel rounding."
        )

    return df


def validate_comorbidity_names(
    icd_mapping: pd.DataFrame,
    snomed_mapping: pd.DataFrame,
) -> None:
    """Require both mappings to contain the same complete canonical set."""

    mapping_sets = {
        "ICD-10": set(icd_mapping["comorbidity"].dropna()),
        "SNOMED CT": set(snomed_mapping["comorbidity"].dropna()),
    }
    problems = []

    for mapping_name, mapping in (
        ("ICD-10", icd_mapping),
        ("SNOMED CT", snomed_mapping),
    ):
        if mapping["comorbidity"].isna().any():
            problems.append(f"{mapping_name} contains missing comorbidity names")

    # Repeated comorbidity names are expected in long format. Repeated
    # category/code pairs are not: they would duplicate the same mapping.
    for mapping_name, mapping, code_column in (
        ("ICD-10", icd_mapping, "icd_code"),
        ("SNOMED CT", snomed_mapping, "snomed_code"),
    ):
        if code_column not in mapping.columns:
            continue
        if mapping[code_column].isna().any() or mapping[code_column].eq("").any():
            problems.append(f"{mapping_name} contains missing codes")
        duplicate_pairs = mapping.duplicated(
            subset=["comorbidity", code_column],
            keep=False,
        )
        if duplicate_pairs.any():
            duplicates = sorted(
                {
                    tuple(pair)
                    for pair in mapping.loc[
                        duplicate_pairs,
                        ["comorbidity", code_column],
                    ].itertuples(index=False, name=None)
                }
            )
            problems.append(
                f"{mapping_name} has duplicate comorbidity/code rows "
                f"{duplicates}"
            )

    for mapping_name, names in mapping_sets.items():
        missing = sorted(EXPECTED_COMORBIDITIES - names)
        unexpected = sorted(names - EXPECTED_COMORBIDITIES)
        if missing:
            problems.append(f"{mapping_name} missing {missing}")
        if unexpected:
            problems.append(f"{mapping_name} has unexpected {unexpected}")

    only_in_icd = sorted(mapping_sets["ICD-10"] - mapping_sets["SNOMED CT"])
    only_in_snomed = sorted(mapping_sets["SNOMED CT"] - mapping_sets["ICD-10"])
    if only_in_icd:
        problems.append(f"only in ICD-10 {only_in_icd}")
    if only_in_snomed:
        problems.append(f"only in SNOMED CT {only_in_snomed}")

    if problems:
        raise ValueError("Comorbidity category validation failed: " + "; ".join(problems))


def build_medication_mapping() -> pd.DataFrame:
    """Create the medication-to-Charlson mapping."""

    df = pd.DataFrame(
        [
            {
                "comorbidity": comorbidity,
                "medication_name": medication,
            }
            for comorbidity, medications in MEDICATION_COMORBIDITY_MAP.items()
            for medication in medications
        ]
    )

    df["medication_name"] = (
        df["medication_name"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if df["medication_name"].duplicated().any():
        duplicates = sorted(
            df.loc[
                df["medication_name"].duplicated(keep=False),
                "medication_name",
            ].unique()
        )
        raise ValueError(
            f"Medications map to multiple comorbidities: {duplicates}"
        )

    return (
        df.drop_duplicates()
        .sort_values(["comorbidity", "medication_name"])
        .reset_index(drop=True)
    )


# =============================================================================
# Main
# =============================================================================
def main(
    input_directory: Path = INPUT_DIRECTORY,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> tuple[Path, Path, Path]:
    """Clean, validate, and save all comorbidity mappings."""

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    quan_path = input_directory / "charlson_icd.csv"
    snomed_path = input_directory / "charlson_snomed.xlsx"

    quan_clean = clean_quan_mapping(quan_path)
    snomed_clean = clean_snomed_mapping(snomed_path)
    medication_clean = build_medication_mapping()

    validate_comorbidity_names(
        quan_clean,
        snomed_clean,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    quan_output = (
        output_directory
        / "charlson_icd10_mapping.csv"
    )

    snomed_output = (
        output_directory
        / "charlson_snomed_mapping.csv"
    )

    medication_output = (
        output_directory
        / "charlson_medication_mapping.csv"
    )

    quan_clean.to_csv(
        quan_output,
        index=False,
    )

    snomed_clean.to_csv(
        snomed_output,
        index=False,
    )

    medication_clean.to_csv(
        medication_output,
        index=False,
    )

    print(
        f"Validated {len(EXPECTED_COMORBIDITIES)} matching comorbidity "
        "names in the ICD-10 and SNOMED mappings."
    )

    print(f"\nSaved ICD mapping to: {quan_output}")
    print(f"Saved SNOMED mapping to: {snomed_output}")
    print(f"Saved medication mapping to: {medication_output}")

    return (
        quan_output,
        snomed_output,
        medication_output,
    )


if __name__ == "__main__":
    main()

import sys
import pandas as pd
from pathlib import Path
import logging


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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / 'configs' / "config.yaml"

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("build_comorbidity_features.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

COMORBIDITY_COLUMN_MAP = {
    "problems": {
        "problem_dt_tm": "comorbidity_date",
        "problem_code": "snomed_code",
    },
    "diagnoses": {
        "diagnosis_date": "comorbidity_date",
        "diagnosis_code_icd": "icd_code",
        "diagnosis_code_snomed": "snomed_code",
        "spell_identifier": "spell_id",
    },
    "prescriptions": {
        "order_dt_tm": "comorbidity_date",
        "medication_name_short": "medication_name",
    },
}

STANDARD_COMORBIDITY_COLS = [
    "subject",
    "spell_id",
    "comorbidity_date",
    "source",
    "icd_code",
    "snomed_code",
    "medication_name",
    "event_date_imputed",
]

def load_config_paths(config_path):
    """Load paths and resolve relative values from the YAML directory."""
    import yaml

    config_path = Path(config_path).expanduser().resolve()

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise ValueError(
            f"Configuration file must contain a 'paths' mapping: {config_path}"
        )

    resolved_paths = {}

    for name, value in config["paths"].items():
        path = Path(value).expanduser()

        if not path.is_absolute():
            path = config_path.parent / path

        resolved_paths[name] = path.resolve()

    return resolved_paths

def load_datasets(config):
    """Load comorbidity sources and infection episode dates."""

    datasets = {}

    dataset_paths = {
        "problems": "cleaned_problems",
        "diagnoses": "cleaned_diagnoses",
        "prescriptions": "cleaned_prescriptions",
    }
    dataset_dtypes = {
        "problems": {
            "subject": "string",
            "problem_code": "string",
        },
        "diagnoses": {
            "subject": "string",
            "spell_identifier": "string",
            "diagnosis_code_icd": "string",
            "diagnosis_code_snomed": "string",
        },
        "prescriptions": {
            "subject": "string",
            "medication_name_short": "string",
        },
    }

    for name, config_key in dataset_paths.items():
        input_path = config[config_key]
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Configured {name} file does not exist: {input_path}"
            )

        datasets[name] = pd.read_csv(
            input_path,
            dtype=dataset_dtypes[name],
        )

    infection_eps_path = config["infection_eps"]
    if not infection_eps_path.is_file():
        raise FileNotFoundError(
            "Configured infection episode file does not exist: "
            f"{infection_eps_path}. Build infection_eps.csv before running "
            "episode-specific comorbidity feature engineering."
        )

    datasets["infection_eps"] = pd.read_csv(
        infection_eps_path,
        usecols=[
            "subject",
            "infection_id",
            "admission_date",
            "spell_id",
            "first_culture_dt",
        ],
        dtype={
            "subject": "string",
            "infection_id": "string",
            "spell_id": "string",
        },
        parse_dates=["admission_date", "first_culture_dt"],
    )

    return datasets

def load_mapping(config):
    """Load CCI mapping information."""

    mapping_paths = {
        "icd": (
            "charlson_icd10_mapping",
            {"icd_code": "string"},
        ),
        "snomed": (
            "charlson_snomed_mapping",
            {"snomed_code": "string"},
        ),
        "medication": (
            "charlson_medication_mapping",
            {"medication_name": "string"},
        ),
    }
    cci_mappings = {}

    for name, (config_key, dtypes) in mapping_paths.items():
        input_path = config[config_key]
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Configured {name} mapping does not exist: {input_path}"
            )

        cci_mappings[name] = pd.read_csv(
            input_path,
            dtype=dtypes,
        )

    return cci_mappings


def standardise_comorbidity_df(df, source, column_map):
    """Convert a comorbidity source to the common schema."""

    df = df.copy()
    df = df.rename(columns=column_map)

    df["source"] = source

    for col in STANDARD_COMORBIDITY_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[STANDARD_COMORBIDITY_COLS]


def fill_missing_diagnosis_dates(
    diagnosis_df,
    infection_eps_dates,
    admission_date_col="admission_date",
):
    """Fill missing diagnosis dates using spell admission dates."""

    diagnosis_df = diagnosis_df.copy()

    spell_dates = (
        infection_eps_dates[
            ["subject", "spell_id", admission_date_col]
        ]
        .drop_duplicates()
    )

    diagnosis_df["comorbidity_date"] = pd.to_datetime(
        diagnosis_df["comorbidity_date"],
        errors="coerce",
    )

    spell_dates[admission_date_col] = pd.to_datetime(
        spell_dates[admission_date_col],
        errors="coerce",
    )

    diagnosis_df = diagnosis_df.merge(
        spell_dates,
        on=["subject", "spell_id"],
        how="left",
    )

    diagnosis_df["event_date_imputed"] = (
        diagnosis_df["comorbidity_date"].isna()
        & diagnosis_df[admission_date_col].notna()
    ).astype(int)

    diagnosis_df["comorbidity_date"] = (
        diagnosis_df["comorbidity_date"]
        .fillna(diagnosis_df[admission_date_col])
    )

    return diagnosis_df 

def combine_comorbidity_datasets(dfs):
    """
    Long-format comorbidity evidence table.

    Output:
        One row per recorded comorbidity-related event across problems,
        diagnoses, and prescriptions.

    Columns:
        subject, spell_id, comorbidity_date, source,
        icd_code, snomed_code, medication_name, event_date_imputed

    Repeated records are retained at this stage. They are mapped to CCI
    conditions and collapsed later into one feature row per infection episode.
    """

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(
        subset=["subject", "spell_id", "comorbidity_date", "source"]
    )

    return combined_df

def map_comorbidities_to_cci(combined_df, cci_mappings):
    """Map source codes and medication names to Charlson categories.

    ICD-10 values are matched against Quan prefixes. SNOMED identifiers and
    medication names are matched exactly. Each result cell is a tuple because
    one source value can legitimately support more than one Charlson category.
    Unmatched or missing values produce an empty tuple.
    """

    mapped_df = combined_df.copy()

    icd_mapping = cci_mappings["icd"]
    snomed_mapping = cci_mappings["snomed"]
    medication_mapping = cci_mappings["medication"]

    icd_prefixes = [
        (str(code).strip().upper().replace(".", ""), comorbidity)
        for code, comorbidity in icd_mapping[
            ["icd_code", "comorbidity"]
        ].itertuples(index=False, name=None)
    ]

    def match_icd_code(value):
        if pd.isna(value):
            return ()

        observed_code = str(value).strip().upper().replace(".", "")
        categories = [
            comorbidity
            for prefix, comorbidity in icd_prefixes
            if observed_code.startswith(prefix)
        ]
        return tuple(dict.fromkeys(categories))

    def build_exact_lookup(mapping, code_column, *, lowercase=False):
        lookup = {}

        for code, comorbidity in mapping[
            [code_column, "comorbidity"]
        ].itertuples(index=False, name=None):
            key = str(code).strip()
            if lowercase:
                key = key.lower()
            lookup.setdefault(key, []).append(comorbidity)

        return {
            key: tuple(dict.fromkeys(categories))
            for key, categories in lookup.items()
        }

    def match_exact(value, lookup, *, lowercase=False):
        if pd.isna(value):
            return ()

        key = str(value).strip()
        if lowercase:
            key = key.lower()
        return lookup.get(key, ())

    snomed_lookup = build_exact_lookup(
        snomed_mapping,
        "snomed_code",
    )
    medication_lookup = build_exact_lookup(
        medication_mapping,
        "medication_name",
        lowercase=True,
    )

    mapped_df["icd_comorbidity"] = mapped_df["icd_code"].map(match_icd_code)
    mapped_df["snomed_comorbidity"] = mapped_df["snomed_code"].map(
        lambda value: match_exact(value, snomed_lookup)
    )
    mapped_df["medication_comorbidity"] = mapped_df["medication_name"].map(
        lambda value: match_exact(
            value,
            medication_lookup,
            lowercase=True,
        )
    )

    return mapped_df
def reconcile_cci_mapping(df):
    """Reconcile mapped CCI evidence without treating empty tuples as data.

    Mapping columns contain tuples: ``()`` means that the source did not map,
    while one or more values preserve legitimate one-to-many mappings. ICD and
    SNOMED evidence is compatible when the two sets overlap; compatible sets
    are combined without duplicates. Disjoint coded evidence is flagged as a
    conflict and deliberately left unresolved. Medication evidence is used
    only when neither coded source maps.
    """

    df = df.copy()

    mapping_columns = [
        "icd_comorbidity",
        "snomed_comorbidity",
        "medication_comorbidity",
    ]
    for column in mapping_columns:
        invalid = ~df[column].map(lambda value: isinstance(value, tuple))
        if invalid.any():
            raise TypeError(f"{column} must contain tuples, using () for no match")

    reconciled = []
    conflicts = []

    for icd_categories, snomed_categories, medication_categories in df[
        mapping_columns
    ].itertuples(index=False, name=None):
        if icd_categories and snomed_categories:
            if set(icd_categories).isdisjoint(snomed_categories):
                reconciled.append(())
                conflicts.append(1)
                continue

            reconciled.append(
                tuple(dict.fromkeys((*icd_categories, *snomed_categories)))
            )
            conflicts.append(0)
            continue

        if icd_categories:
            reconciled.append(icd_categories)
        elif snomed_categories:
            reconciled.append(snomed_categories)
        elif medication_categories:
            reconciled.append(medication_categories)
        else:
            reconciled.append(())

        conflicts.append(0)

    df["coding_conflict"] = pd.Series(
        conflicts,
        index=df.index,
        dtype="int64",
    )
    df["cci_condition"] = pd.Series(
        reconciled,
        index=df.index,
        dtype="object",
    )

    return df

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("build_comorbidity_features.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Loading configuration...")
    configs = load_config_paths(CONFIG_PATH)

    logging.info("Loading datasets...")
    datasets = load_datasets(configs)

    logging.info("Loading CCI mapping information...")
    cci_mappings = load_mapping(configs)

    logging.info("Standardising comorbidity datasets...")

    standardised_dfs = []

    for source, column_map in COMORBIDITY_COLUMN_MAP.items():

        df = standardise_comorbidity_df(
            datasets[source],
            source=source,
            column_map=column_map,
        )

        if source == "diagnoses":
            df = fill_missing_diagnosis_dates(
                df,
                datasets["infection_eps"],
            )

        standardised_dfs.append(df)

    logging.info("Combining comorbidity datasets...")

    combined_df = combine_comorbidity_datasets(standardised_dfs)

    logging.info("Mapping comorbidities to CCI conditions...")

    mapped_df = map_comorbidities_to_cci(combined_df, cci_mappings)

    logging.info("Reconciling CCI mappings...")

    cci_comorbidities = reconcile_cci_mapping(mapped_df)

    logging.info("Saving CCI comorbidity features...")
    cci_comorbidities.to_csv(OUTPUT_DIRECTORY / "cci_comorbidity_features.csv", index=False)


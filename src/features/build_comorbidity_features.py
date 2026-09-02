import logging
import sys
import pandas as pd
from pathlib import Path
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "synthetic_data_cleaned"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / 'configs' / "config.yaml"

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

CCI_WEIGHTS = {
    "myocardial_infarction": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease": 1,
    "dementia": 1,
    "chronic_pulmonary_disease": 1,
    "rheumatic_disease": 1,
    "peptic_ulcer_disease": 1,
    "mild_liver_disease": 1,
    "diabetes_without_complication": 1,
    "diabetes_with_complication": 2,
    "hemiplegia_or_paraplegia": 2,
    "renal_disease": 2,
    "malignancy": 2,
    "moderate_or_severe_liver_disease": 3,
    "metastatic_solid_tumor": 6,
    "aids_hiv": 6,
}

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

    diagnosis_df = diagnosis_df.drop(columns=[admission_date_col])

    return diagnosis_df 

def combine_comorbidity_datasets(dfs):
    """Combine sources into one longitudinal evidence table."""

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates()

    return combined_df

def map_comorbidities_to_cci(combined_df, cci_mappings):
    """Map clinical evidence to tuple-valued CCI categories."""

    mapped_df = combined_df.copy()

    # Prepare the three source mappings.
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
        """Match one ICD-10 code."""

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
        """Build an exact-match lookup."""

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
        """Match one exact source value."""

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

    # Map every evidence source to condition tuples.
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
    """Resolve CCI evidence and flag coding conflicts."""

    df = df.copy()

    # Require tuple-valued mappings.
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
                # Exclude unresolved disagreements from CCI flags.
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

    # Store the audit flag and resolved conditions.
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


def aggregate_comorbidities_by_infection_episode(
    cci_df,
    infection_df,
    *,
    subject_col="subject",
    infection_id_col="infection_id",
    condition_col="cci_condition",
    comorbidity_date_col="comorbidity_date",
    admission_date_col="admission_date",
):
    """Build historical CCI flags for each infection episode."""

    # Validate the required schemas.
    cci_required = {
        subject_col,
        condition_col,
        comorbidity_date_col,
    }
    infection_required = {
        subject_col,
        infection_id_col,
        admission_date_col,
    }

    missing_cci_columns = sorted(cci_required.difference(cci_df.columns))
    if missing_cci_columns:
        raise ValueError(
            "cci_df is missing required columns: "
            f"{missing_cci_columns}"
        )

    missing_infection_columns = sorted(
        infection_required.difference(infection_df.columns)
    )
    if missing_infection_columns:
        raise ValueError(
            "infection_df is missing required columns: "
            f"{missing_infection_columns}"
        )

    # Validate episode keys and cutoff dates.
    index_columns = [subject_col, infection_id_col]
    episodes = infection_df[
        [*index_columns, admission_date_col]
    ].copy()

    missing_episode_keys = episodes[index_columns].isna().any(axis=1)
    if missing_episode_keys.any():
        raise ValueError(
            f"infection_df contains missing values in {index_columns}"
        )

    duplicate_episode_keys = episodes.duplicated(index_columns, keep=False)
    if duplicate_episode_keys.any():
        duplicate_keys = (
            episodes.loc[duplicate_episode_keys, index_columns]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            "infection_df must contain one row per subject/infection_id; "
            f"duplicate keys: {duplicate_keys}"
        )

    original_admission_dates = episodes[admission_date_col]
    episodes[admission_date_col] = pd.to_datetime(
        original_admission_dates,
        errors="coerce",
        format="mixed",
    )
    invalid_admission_dates = (
        original_admission_dates.notna()
        & episodes[admission_date_col].isna()
    )
    if invalid_admission_dates.any():
        raise ValueError(
            f"infection_df contains invalid {admission_date_col} values"
        )
    if episodes[admission_date_col].isna().any():
        raise ValueError(
            f"infection_df contains missing {admission_date_col} values"
        )

    episode_index = pd.MultiIndex.from_frame(episodes[index_columns])

    # Start every episode with 17 zero flags.
    result = pd.DataFrame(
        0,
        index=episode_index,
        columns=list(CCI_WEIGHTS),
        dtype="int64",
    )
    result.columns.name = None

    events = cci_df[
        [subject_col, condition_col, comorbidity_date_col]
    ].copy()

    # Parse event dates and reject invalid values.
    original_comorbidity_dates = events[comorbidity_date_col]
    events[comorbidity_date_col] = pd.to_datetime(
        original_comorbidity_dates,
        errors="coerce",
        format="mixed",
    )
    invalid_comorbidity_dates = (
        original_comorbidity_dates.notna()
        & events[comorbidity_date_col].isna()
    )
    if invalid_comorbidity_dates.any():
        raise ValueError(
            f"cci_df contains invalid {comorbidity_date_col} values"
        )

    def normalise_conditions(value):
        """Return condition names as a tuple."""

        if isinstance(value, str):
            return (value,)
        if isinstance(value, (tuple, list, set, frozenset)):
            return tuple(value)
        if pd.isna(value):
            return ()
        raise TypeError(
            f"{condition_col} values must be condition names or iterables "
            "of condition names"
        )

    # Expand tuples; empty conflict tuples contribute no rows.
    events[condition_col] = events[condition_col].map(
        normalise_conditions
    )
    events = events.explode(condition_col)
    events = events.loc[events[condition_col].notna()].copy()

    # Validate that all conditions are strings and known CCI categories.
    non_string_conditions = ~events[condition_col].map(
        lambda value: isinstance(value, str)
    )
    if non_string_conditions.any():
        raise TypeError(f"{condition_col} must contain string condition names")

    unknown_conditions = sorted(
        set(events[condition_col]).difference(CCI_WEIGHTS)
    )
    if unknown_conditions:
        raise ValueError(
            f"cci_df contains unknown CCI conditions: {unknown_conditions}"
        )
    
    # get only events with valid subject and comorbidity dates
    dated_events = events.loc[
        events[subject_col].notna()
        & events[comorbidity_date_col].notna()
    ]
    if dated_events.empty or episodes.empty:
        return result

    # Keep each condition's first dated evidence.
    first_occurrences = (
        dated_events
        .groupby(
            [subject_col, condition_col],
            as_index=False,
            sort=False,
        )[comorbidity_date_col]
        .min()
    )

    # Keep conditions present before or on the episode admission date.
    episode_conditions = episodes.merge(
        first_occurrences,
        on=subject_col,
        how="inner",
    )
    episode_conditions = episode_conditions.loc[
        episode_conditions[comorbidity_date_col]
        <= episode_conditions[admission_date_col]
    ]
    if episode_conditions.empty:
        return result

    # Pivot eligible evidence to binary flags.
    flags = (
        episode_conditions
        .assign(_present=1)
        .pivot_table(
            index=index_columns,
            columns=condition_col,
            values="_present",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(
            index=episode_index,
            columns=list(CCI_WEIGHTS),
            fill_value=0,
        )
        .astype("int64")
    )
    flags.columns.name = None

    return flags


def calculate_cci_score(df):
    """Calculate CCI score while preserving original condition flags."""

    df = df.copy()
    scoring_df = df.copy()

    scoring_df.loc[
        scoring_df["diabetes_with_complication"] == 1,
        "diabetes_without_complication",
    ] = 0

    scoring_df.loc[
        scoring_df["moderate_or_severe_liver_disease"] == 1,
        "mild_liver_disease",
    ] = 0

    scoring_df.loc[
        scoring_df["metastatic_solid_tumor"] == 1,
        "malignancy",
    ] = 0

    df["cci_score"] = sum(
        scoring_df[condition] * weight
        for condition, weight in CCI_WEIGHTS.items()
    )

    return df


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler("build_comorbidity_features.log"),
        ],
    )

    logger.info("Loading configuration...")
    configs = load_config_paths(CONFIG_PATH)

    logger.info("Loading datasets...")
    # ``infection_eps`` contains one row per infection episode.
    # columns: subject, infection_id, admission_date, spell_id, first_culture_dt
    datasets = load_datasets(configs)

    logger.info("Loading CCI mapping information...")
    cci_mappings = load_mapping(configs)

    logger.info("Standardising comorbidity datasets...")

    standardised_dfs = []

    for source, column_map in COMORBIDITY_COLUMN_MAP.items():

        # One row per comorbidity evidence record.
        # Standard columns:
        # subject, spell_id, comorbidity_date, source,
        # icd_code, snomed_code, medication_name, event_date_imputed
        df = standardise_comorbidity_df(
            datasets[source],
            source=source,
            column_map=column_map,
        )

        if source == "diagnoses":
            # Fill missing diagnosis dates from the corresponding
            # spell admission date where available.
            df = fill_missing_diagnosis_dates(
                df,
                datasets["infection_eps"],
            )

        standardised_dfs.append(df)

    logger.info("Combining comorbidity datasets...")

    # Longitudinal evidence table:
    # one row per comorbidity evidence record across all sources.
    combined_df = combine_comorbidity_datasets(standardised_dfs)

    logger.info("Mapping comorbidities to CCI conditions...")

    # Adds tuple-valued ICD, SNOMED and medication CCI mappings.
    mapped_df = map_comorbidities_to_cci(
        combined_df,
        cci_mappings,
    )

    logger.info("Reconciling CCI mappings...")

    # Adds resolved cci_condition and coding_conflict columns.
    cci_comorbidities = reconcile_cci_mapping(mapped_df)

    logger.info("Aggregating comorbidities by infection episode...")

    # Unit of observation changes here:
    # output = one row per infection episode,
    # with 17 binary historical CCI condition features.
    cci_comorbidities = aggregate_comorbidities_by_infection_episode(
        cci_comorbidities,
        datasets["infection_eps"],
    )

    logger.info("Calculating CCI scores...")

    # Adds one cci_score column to the 17 condition features.
    cci_comorbidities = calculate_cci_score(cci_comorbidities)

    logger.info("Saving CCI comorbidity features...")

    if not OUTPUT_DIRECTORY.exists():
        logger.info(f"Creating output directory: {OUTPUT_DIRECTORY}")
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    cci_comorbidities.to_csv(
        OUTPUT_DIRECTORY / "cci_comorbidity_features.csv",
        index=True,
    )

    print(cci_comorbidities.head())

    print(cci_comorbidities.columns)

    logger.info("CCI comorbidity feature engineering complete!! 🙌🏼👌🏼😁.")
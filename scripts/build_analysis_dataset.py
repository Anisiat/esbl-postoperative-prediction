import logging
import sys
from pathlib import Path

import pandas as pd
from rich.logging import RichHandler
from rich.markup import escape

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"

SRC_DIR = PROJECT_ROOT / "src"
RICH_MARKUP = {"markup": True}

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Imports
# =============================================================================

from steps.clean_demographics import run as clean_demographics
from steps.clean_microbiology import run as clean_microbiology

from steps.clean_diagnoses import run as clean_diagnoses
from steps.clean_problems import run as clean_problems
from steps.clean_prescribing import run as clean_prescribing

from steps.assign_infection_ep_id import (
    get_infection_ep_id as assign_infection_id,
)

from steps.get_abx_exposure import (
    get_abx_exposure,
    group_prophylaxis,
)

from steps.build_ep_surgery_features import (
    get_surgery_features,
)

from steps.build_infection_eps import (
    build_infection_eps,
)

from steps.build_comorbidity_features import (
    build_comorbidity_features,
)

from steps.get_healthcare_exposure import (
    get_healthcare_exposure,
)

from steps.get_temp_crp import (
    get_temperature,
    get_crp,
)

from steps.get_colonisation import (
    get_prior_esbl_culture,
)

from utils import data_cleaning_tools as dct


# =============================================================================
# Terminal logging
# =============================================================================

def configure_terminal_logging() -> None:
    """Configure clear, colour-coded logging for this command-line script."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
            )
        ],
        # Replace any existing root handlers so this command always uses Rich.
        force=True,
    )


def log_section(title: str) -> None:
    """Write a visually distinct pipeline section heading."""

    if any(
        isinstance(handler, RichHandler)
        for handler in logging.getLogger().handlers
    ):
        logging.info(
            "[bold cyan]-- %s --[/bold cyan]",
            escape(title),
            extra=RICH_MARKUP,
        )
    else:
        logging.info("-- %s --", title)


# =============================================================================
# Pipeline
# =============================================================================

def run_steps(
    cfg_path=CONFIG_PATH,
    verbose=True,
):

    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)

    logging.info("Loading config...")

    cfg_path = Path(cfg_path)

    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = dct.load_config(cfg_path)
    paths = cfg.get("paths", {})

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # 1. Clean core source tables
    # =========================================================================

    log_section("1. Clean core source tables")
    logging.info("Cleaning demographics...")

    demographics_df = clean_demographics(
        cfg_path,
        save=True,
        return_df=True,
    )

    logging.info("Cleaning microbiology...")

    microbiology_df, screens_df = clean_microbiology(
        cfg_path,
        save=True,
        return_df=True,
    )

    # =========================================================================
    # 2. Build infection-level cohort
    # =========================================================================

    log_section("2. Build infection-level cohort")
    logging.info("Assigning infection IDs...")

    microbiology_df = assign_infection_id(
        microbiology_df
    )

    logging.info("Adding antibiotic exposure...")

    infection_abx_df = get_abx_exposure(
        mcs_df=microbiology_df,
        cfg_path=cfg_path,
    )

    logging.info("Adding surgery features...")

    infection_surgery_df = get_surgery_features(
        infection_abx_df
    )

    # Keep only postoperative infections
    infection_surgery_df = infection_surgery_df[
        infection_surgery_df["surgery_before_infection"] == 1
    ].copy()

    infection_surgery_df["prophylaxis_group"] = (
        infection_surgery_df["prophylaxis"]
        .apply(group_prophylaxis)
    )

    logging.info("Building infection episodes...")

    infection_eps = build_infection_eps(
        infection_surgery_df
    )

    # -------------------------------------------------------------------------
    # Save infection episodes before downstream feature engineering
    # -------------------------------------------------------------------------

    infection_eps_path = (
        INTERIM_DIR
        / "infection_eps.csv"
    )

    infection_eps.to_csv(
        infection_eps_path,
        index=False,
    )

    logging.info(
        "Saved infection episode table to %s",
        infection_eps_path,
    )

    # =========================================================================
    # 3. Clean comorbidity source tables
    # =========================================================================
    #
    # These remain longitudinal tables.
    # They are NOT collapsed to one row per patient here.
    # =========================================================================

    log_section("3. Clean comorbidity source tables")
    logging.info("Cleaning diagnosis table...")

    diagnoses_df = clean_diagnoses(
        cfg_path,
        save=True,
        return_df=True,
    )

    logging.info("Cleaning problem-list table...")

    problems_df = clean_problems(
        cfg_path,
        save=True,
        return_df=True,
    )

    logging.info("Cleaning prescribing table...")

    prescribing_df = clean_prescribing(
        cfg_path,
        save=True,
        return_df=True,
    )

    # =========================================================================
    # 4. Build time-aware comorbidity features
    # =========================================================================
    #
    # infection_eps is now available and provides:
    #   - subject
    #   - infection_id
    #   - admission_date
    #   - first_culture_dt
    #
    # Therefore comorbidities can be calculated relative to each infection.
    # =========================================================================

    log_section("4. Build time-aware comorbidity features")
    logging.info(
        "Building episode-specific comorbidity features..."
    )

    comorbidity_df = build_comorbidity_features(
        infection_df=infection_eps,
        diagnoses_df=diagnoses_df,
        problems_df=problems_df,
        prescribing_df=prescribing_df,
    )

    comorbidity_path = (
        INTERIM_DIR
        / "comorbidity_features.csv"
    )

    comorbidity_df.to_csv(
        comorbidity_path,
        index=False,
    )

    # =========================================================================
    # 5. Assemble analysis dataset
    # =========================================================================

    log_section("5. Assemble analysis dataset")
    logging.info(
        "Merging infection episodes, comorbidities and demographics..."
    )

    analysis_df = (
        infection_eps
        .merge(
            comorbidity_df,
            on=[
                "subject",
                "infection_id",
            ],
            how="left",
        )
        .merge(
            demographics_df,
            on="subject",
            how="left",
        )
    )

    # =========================================================================
    # 6. Add remaining clinical features
    # =========================================================================

    log_section("6. Add remaining clinical features")
    logging.info(
        "Adding prior healthcare exposure..."
    )

    analysis_df = get_healthcare_exposure(
        analysis_df,
        cfg_paths=paths,
        exposure_type="any",
    )

    logging.info(
        "Adding temperature and CRP..."
    )

    analysis_df = get_temperature(
        analysis_df,
        cfg_path=cfg_path,
    )

    analysis_df = get_crp(
        analysis_df,
        cfg_path=cfg_path,
    )

    logging.info(
        "Adding prior ESBL culture..."
    )

    analysis_df = get_prior_esbl_culture(
        analysis_df,
        cfg_path=cfg_path,
    )

    # =========================================================================
    # 7. Outcomes / final derived variables
    # =========================================================================

    log_section("7. Derive outcomes")
    analysis_df["death_date"] = pd.to_datetime(
        analysis_df["death_date"],
        errors="coerce",
    )

    analysis_df["discharge_date"] = pd.to_datetime(
        analysis_df["discharge_date"],
        errors="coerce",
    )

    analysis_df["in_hosp_mortality"] = (
        analysis_df["death_date"].notna()
        & (
            analysis_df["death_date"]
            <= analysis_df["discharge_date"]
        )
    ).astype(int)

    # Already used for cohort restriction
    analysis_df = analysis_df.drop(
        columns=["surgery_before_infection"],
        errors="ignore",
    )

    # =========================================================================
    # 8. Save final analysis dataset
    # =========================================================================

    log_section("8. Save analysis dataset")
    analysis_df_path = (
        INTERIM_DIR
        / "analysis_df.csv"
    )

    analysis_df.to_csv(
        analysis_df_path,
        index=False,
    )

    logging.info(
        "Saved final analysis dataset to %s",
        analysis_df_path,
    )

    return analysis_df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":

    configure_terminal_logging()

    logging.info(
        "[bold blue]Build ESBL analysis dataset[/bold blue]",
        extra=RICH_MARKUP,
    )
    logging.info("Configuration    : %s", CONFIG_PATH)
    logging.info("Output directory : %s", INTERIM_DIR)

    run_steps(
        cfg_path=CONFIG_PATH,
        verbose=True,
    )

    logging.info(
        "[bold green]Analysis dataset pipeline completed successfully.[/bold green]",
        extra=RICH_MARKUP,
    )

import pandas as pd
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "synthetic_data"



def generate_comorbidity_tables():
    """Generate synthetic tables for developing comorbidity feature engineering.

    Returns
    -------
    cohort_df : pd.DataFrame
        Synthetic index elective-surgery/infection episodes.

    diagnoses_df : pd.DataFrame
        Synthetic episode diagnosis records containing ICD-10 and SNOMED codes.

    problems_df : pd.DataFrame
        Synthetic historical problem-list records containing SNOMED codes.

    prescriptions_df : pd.DataFrame
        Synthetic prescribing records used as supporting medication evidence.
    """

    # -------------------------------------------------------------------------
    # 1. Index elective surgery / infection cohort
    # -------------------------------------------------------------------------

    cohort_df = pd.DataFrame(
        {
            "SUBJECT": [
                "P001",
                "P002",
                "P003",
                "P003",
                "P004",
                "P005",
            ],
            "SPELL_IDENTIFIER": [
                "S003",
                "S011",
                "S020",
                "S021",
                "S030",
                "S040",
            ],
            "ADMISSION_DATE": [
                "2022-03-10",
                "2023-06-01",
                "2021-05-01",
                "2024-01-01",
                "2023-09-10",
                "2025-02-01",
            ],
            "DISCHARGE_DATE": [
                "2022-03-25",
                "2023-06-15",
                "2021-05-12",
                "2024-01-16",
                "2023-09-20",
                "2025-02-20",
            ],
            "SAMPLE_COLLECTED_DT": [
                "2022-03-18",
                "2023-06-08",
                "2021-05-06",
                "2024-01-10",
                "2023-09-15",
                "2025-02-12",
            ],
        }
    )

    # -------------------------------------------------------------------------
    # 2. Episode diagnoses
    #
    # Deliberately includes:
    # - mostly missing diagnosis dates, as in the real table;
    # - ICD-only diagnoses;
    # - SNOMED-only diagnoses;
    # - diagnoses from the index spell;
    # - diagnoses attached to other spells;
    # - conditions that should / should not belong to Charlson categories.
    # -------------------------------------------------------------------------

    diagnoses_df = pd.DataFrame(
        {
            "SUBJECT": [
                "P001",
                "P001",
                "P001",
                "P002",
                "P002",
                "P003",
                "P003",
                "P003",
                "P004",
                "P005",
            ],
            "SPELL_IDENTIFIER": [
                "S001",
                "S002",
                "S003",
                "S010",
                "S011",
                "S019",
                "S020",
                "S021",
                "S029",
                "S040",
            ],
            "DIAGNOSIS_DATE": [
                None,
                None,
                "2022-03-15",
                None,
                None,
                "2020-06-01",
                None,
                "2024-01-08",
                None,
                None,
            ],
            "DIAGNOSIS_CODE_ICD": [
                "e119",   # Type 2 diabetes
                "i509",   # Heart failure
                "n179",   # Acute kidney injury - not chronic renal CCI
                "j449",   # COPD
                "n390",   # UTI - not a Charlson condition
                "i639",   # Stroke
                "e119",   # Diabetes
                "c787",   # Secondary malignant neoplasm
                None,
                "k704",   # Severe liver disease candidate
            ],
            "DIAGNOSIS_CODE_SNOMED": [
                "none",
                "none",
                "none",
                "none",
                "none",
                "230690007",
                "none",
                "none",
                "427571000",
                "none",
            ],
            "DIAGNOSIS_DESC_ICD": [
                "type 2 diabetes mellitus",
                "heart failure, unspecified",
                "acute kidney failure, unspecified",
                "chronic obstructive pulmonary disease",
                "urinary tract infection, site not specified",
                "cerebral infarction, unspecified",
                "type 2 diabetes mellitus",
                "secondary malignant neoplasm of liver",
                None,
                "alcoholic hepatic failure",
            ],
            "DIAGNOSIS_DESC_SNOMED": [
                "none",
                "none",
                "none",
                "none",
                "none",
                "stroke",
                "none",
                "none",
                "lumbosacral radiculoplexus neuropathy due to type 1 diabetes mellitus",
                "none",
            ],
        }
    )

    # -------------------------------------------------------------------------
    # 3. Problems / historical problem list
    #
    # PROBLEM_CODE is represented as a SNOMED CT concept ID.
    # -------------------------------------------------------------------------

    problems_df = pd.DataFrame(
        {
            "SUBJECT": [
                "P001",
                "P001",
                "P002",
                "P003",
                "P004",
                "P005",
            ],
            "PROBLEM_CODE": [
                "44054006",
                "84114007",
                "13645005",
                "230690007",
                "427571000",
                "38341003",
            ],
            "PROBLEM_DESC": [
                "type 2 diabetes mellitus",
                "heart failure",
                "chronic obstructive lung disease",
                "stroke",
                "lumbosacral radiculoplexus neuropathy due to type 1 diabetes mellitus",
                "hypertension",
            ],
            "PROBLEM_DT_TM": [
                "2019-02-01 10:00:00",
                "2020-08-12 15:30:00",
                "2020-01-01 09:00:00",
                "2022-05-01 12:00:00",
                "2022-11-10 14:00:00",
                "2024-05-20 11:00:00",
            ],
        }
    )

    # -------------------------------------------------------------------------
    # 4. Pharmacy prescribing
    #
    # Medication evidence is deliberately kept separate from formal Charlson
    # condition ascertainment.
    # -------------------------------------------------------------------------

    prescriptions_df = pd.DataFrame(
        {
            "SUBJECT": [
                "P001",
                "P001",
                "P002",
                "P003",
                "P004",
                "P005",
            ],
            "MEDICATION_NAME_SHORT": [
                "metformin",
                "furosemide",
                "salbutamol",
                "aspirin",
                "insulin soluble",
                "amlodipine",
            ],
            "THERAPEUTICAL_CLASS": [
                "metabolic agents",
                "cardiovascular agents",
                "respiratory agents",
                "cardiovascular agents",
                "metabolic agents",
                "cardiovascular agents",
            ],
            "ORDER_DT_TM": [
                "2021-10-01 08:00:00",
                "2021-11-10 09:00:00",
                "2022-12-20 10:00:00",
                "2023-06-01 12:00:00",
                "2023-01-15 13:00:00",
                "2024-09-01 11:00:00",
            ],
            "ADMISSION_MEDICINE_Y_N": [
                "Y",
                "Y",
                "Y",
                "N",
                "Y",
                "Y",
            ],
            "GP_TO_CONTINUE": [
                "Y",
                "Y",
                "Y",
                "N",
                "Y",
                "Y",
            ],
        }
    )

    # -------------------------------------------------------------------------
    # Convert temporal columns to datetime
    # -------------------------------------------------------------------------

    cohort_df["ADMISSION_DATE"] = pd.to_datetime(
        cohort_df["ADMISSION_DATE"]
    )
    cohort_df["DISCHARGE_DATE"] = pd.to_datetime(
        cohort_df["DISCHARGE_DATE"]
    )
    cohort_df["SAMPLE_COLLECTED_DT"] = pd.to_datetime(
        cohort_df["SAMPLE_COLLECTED_DT"]
    )

    diagnoses_df["DIAGNOSIS_DATE"] = pd.to_datetime(
        diagnoses_df["DIAGNOSIS_DATE"],
        errors="coerce",
    )

    problems_df["PROBLEM_DT_TM"] = pd.to_datetime(
        problems_df["PROBLEM_DT_TM"]
    )

    prescriptions_df["ORDER_DT_TM"] = pd.to_datetime(
        prescriptions_df["ORDER_DT_TM"]
    )

    return (
        cohort_df,
        diagnoses_df,
        problems_df,
        prescriptions_df,
    )

if __name__ == "__main__":
    (
        cohort_df,
        diagnoses_df,
        problems_df,
        prescriptions_df,
    ) = generate_comorbidity_tables()

    # save datasets to CSV files 
    if not OUTPUT_DIRECTORY.exists():
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    cohort_df.to_csv(f"{OUTPUT_DIRECTORY}/cohort.csv", index=False)
    diagnoses_df.to_csv(f"{OUTPUT_DIRECTORY}/diagnoses.csv", index=False)
    problems_df.to_csv(f"{OUTPUT_DIRECTORY}/problems.csv", index=False)
    prescriptions_df.to_csv(f"{OUTPUT_DIRECTORY}/prescriptions.csv", index=False)

    logging.info(f"Synthetic comorbidity tables saved to {OUTPUT_DIRECTORY}")
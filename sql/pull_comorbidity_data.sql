-- =============================================================================
-- Extract comorbidity source data for the elective surgery ESBL cohort
-- =============================================================================
--
-- Purpose:
--   Extract longitudinal clinical information required to derive historical
--   comorbidity features for patients in the elective surgery ESBL cohort.
--
-- Data sources:
--   1. ICARE_PROBLEMS_ANON
--        - Historical problem-list entries
--        - Includes coded problems and free-text descriptions
--
--   2. ICARE_EPISODES_DIAGNOSIS_ANON
--        - Diagnoses recorded during hospital episodes
--        - Includes ICD and SNOMED codes
--
--   3. ICARE_PHARMACY_PRESCRIBING_ANON
--        - Prescribed medications
--        - Used as supporting evidence for selected comorbidities
--
-- Cohort restriction:
--   Only records belonging to subjects present in
--   AT_ELECTIVE_SURGERY_MICRO_EKP are extracted.
--
-- Downstream processing:
--   The extracted tables will be processed in Python to:
--     - standardise codes and dates;
--     - identify information available before each index admission/infection;
--     - derive comorbidity flags from ICD/SNOMED codes;
--     - calculate a Quan-Charlson comorbidity score;
--     - retain medication-derived evidence separately for validation and
--       potential prediction features.
--
-- Important:
--   Comorbidity definitions and temporal filtering are NOT performed here.
--   Snowflake is used only for cohort-restricted data extraction. Clinical
--   feature engineering is performed downstream in Python.
-- =============================================================================


-- =============================================================================
-- 1. Problems / historical problem list
-- =============================================================================

SELECT
    p.SUBJECT,
    p.PROBLEM_CODE,
    p.PROBLEM_DESC,
    p.PROBLEM_DT_TM
FROM ICHT_PROD.ICARE_ICHT.ICARE_PROBLEMS_ANON p
WHERE p.SUBJECT IN (
    SELECT DISTINCT SUBJECT
    FROM ICHT_SANDBOX_PROD.SMTPATH_25046.AT_ELECTIVE_SURGERY_MICRO_EKP
);


-- =============================================================================
-- 2. Episode diagnoses
-- =============================================================================

SELECT
    d.SUBJECT,
    d.SPELL_IDENTIFIER,
    d.DIAGNOSIS_DATE,
    d.DIAGNOSIS_CODE_ICD,
    d.DIAGNOSIS_CODE_SNOMED,
    d.DIAGNOSIS_DESC_ICD,
    d.DIAGNOSIS_DESC_SNOMED
FROM ICHT_PROD.ICARE_ICHT.ICARE_EPISODES_DIAGNOSIS_ANON d
WHERE d.SUBJECT IN (
    SELECT DISTINCT SUBJECT
    FROM ICHT_SANDBOX_PROD.SMTPATH_25046.AT_ELECTIVE_SURGERY_MICRO_EKP
);


-- =============================================================================
-- 3. Pharmacy prescribing
-- =============================================================================

SELECT
    rx.SUBJECT,
    rx.MEDICATION_NAME_SHORT,
    rx.THERAPEUTICAL_CLASS,
    rx.ORDER_DT_TM,
    rx.ADMISSION_MEDICINE_Y_N,
    rx.GP_TO_CONTINUE
FROM ICHT_PROD.ICARE_ICHT.ICARE_PHARMACY_PRESCRIBING_ANON rx
WHERE rx.SUBJECT IN (
    SELECT DISTINCT SUBJECT
    FROM ICHT_SANDBOX_PROD.SMTPATH_25046.AT_ELECTIVE_SURGERY_MICRO_EKP
)
AND LOWER(rx.THERAPEUTICAL_CLASS) IN (
    'cardiovascular agents',
    'central nervous system agents',
    'coagulation modifiers',
    'gastrointestinal agents',
    'genitourinary tract agents',
    'hormones/hormone modifiers',
    'metabolic agents',
    'miscellaneous agents',
    'none',
    'nutritional products',
    'respiratory agents',
    'topical agents'
);
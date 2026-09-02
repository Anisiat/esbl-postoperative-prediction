# Comorbidity feature-building bundle

This bundle contains the synthetic source tables, mock infection episodes,
raw and cleaned Charlson mappings, cleaning scripts, feature-building script,
and focused tests needed to reproduce the current local comorbidity workflow.

## Setup

Run commands from the directory containing `configs/`, `data/`, `src/`, and
`tests/`:

```bash
python -m pip install -r requirements-comorbidity.txt
```

## Rebuild the inputs and mappings

```bash
python src/data_cleaning/clean_icd_snomed_prescribing.py
python src/data_cleaning/clean_diagnoses.py
python src/data_cleaning/clean_problems.py
python src/data_cleaning/clean_prescriptions.py
```

The mapping cleaner reads `configs/charlson_icd.csv` and
`configs/charlson_snomed.xlsx`. The three clinical cleaners read the synthetic
CSV files under `data/synthetic_data/`.

## Build the comorbidity evidence table

```bash
python src/features/build_comorbidity_features.py
```

The builder reads `configs/config.yaml`, whose relative paths are resolved from
the `configs/` directory. Keep the directory hierarchy unchanged. The included
`data/interim/infection_eps.csv` is a mock episode table for local test runs.
The current script writes
`data/synthetic_data_cleaned/cci_comorbidity_features.csv`.

## Tests

```bash
python -m pytest \
  tests/test_clean_icd_snomed.py \
  tests/test_clean_problems.py \
  tests/test_clean_prescriptions.py \
  tests/test_build_comorbidity_features.py
```

All patient records included here are synthetic. The SNOMED CT source workbook
is included for local reproducibility; check terminology licensing and workbook
metadata before distributing the bundle outside the authorised project team.

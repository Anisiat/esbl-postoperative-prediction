# ESBL Postoperative Infection Prediction

## Project overview

This project aims to develop and evaluate machine-learning models for predicting ESBL-positive postoperative Enterobacterales infections in elective surgical patients.

The work forms part of a wider research programme investigating how routinely collected electronic health-record data can support antibiotic decision-making and infection management.

## Prediction task

The primary prediction task is binary classification:

* **Outcome = 1:** ESBL-positive Enterobacterales infection
* **Outcome = 0:** non-ESBL Enterobacterales infection

The initial modelling cohort consists of adult patients undergoing elective surgery who subsequently had a culture-positive Enterobacterales infection during the defined postoperative period.

Each row in the modelling dataset represents one infection episode.

## Candidate predictors

Candidate predictors may include:

* Patient demographics
* Comorbidities
* Previous ESBL positivity
* Previous antimicrobial exposure
* Surgical specialty
* Procedure characteristics
* Perioperative antibiotic prophylaxis
* Organism
* Specimen or infection site
* Relevant healthcare exposure variables

Predictor availability and timing will be reviewed carefully to avoid data leakage.

## Planned models

Initial models will include:

* Logistic regression
* Regularised logistic regression
* Decision tree
* Random forest
* Gradient boosting
* XGBoost

## Evaluation

Models will be evaluated using measures including:

* Area under the receiver operating characteristic curve
* Area under the precision-recall curve
* Sensitivity
* Specificity
* Positive predictive value
* Negative predictive value
* Calibration
* Brier score

Model performance will be assessed using an appropriately separated training, validation and test strategy. The final splitting strategy will account for repeated observations belonging to the same patient.

## Repository structure

```text
data/          Generated or processed data
notebooks/     Exploratory analyses
src/           Reusable project code
tests/         Automated code tests
outputs/       Model results, figures and saved artefacts
```

## Data governance

The real study data are held within a secure data environment and are not included in this repository.

Local development will use synthetic data designed to reproduce the structure, data types, missingness patterns and approximate relationships present in the secure dataset without reproducing identifiable patient information.

Scripts developed locally will subsequently be transferred into the secure data environment and run against the real study data.

## Project status

The project is currently under development. The initial phase involves:

1. Defining the prediction target and prediction time point.
2. Creating a synthetic development dataset.
3. Building a reproducible preprocessing pipeline.
4. Establishing baseline models.
5. Evaluating and comparing candidate models.


## Synthetic data generation

[`src/generate_synthetic_data.py`](src/generate_synthetic_data.py) creates a
local development dataset from aggregate, non-patient-level summaries of the
real iCARE modelling data. The summaries are produced inside the secure iCARE
environment using [`scripts/summarise_real_data.py`](scripts/summarise_real_data.py).
Data summaries are placed in `data/real_data_summary/`.

The generator uses:

* `dataset_summary.csv` for the number of rows, patients and ESBL-positive
  outcomes;
* `missingness_summary.csv` for column order, data types and missing-value
  counts;
* `continuous_summary.csv` for ranges and quantiles of continuous variables;
* `categorical_summary.csv` for category frequencies;
* `dependency_correlations.csv` for Spearman correlations between continuous
  and binary predictors, including correlations between missingness indicators;
  and
* `logistic_coefficients.csv` to reproduce approximate fitted relationships
  between the predictors and `esbl_status`.

Continuous values are sampled from piecewise-linear distributions defined by
the reported quantiles, while categorical values and missing values are created
to match the aggregate counts. A simple Gaussian rank-copula then reorders the
generated numeric and binary values so their rank correlations and joint
missingness approximate those observed in iCARE, without changing any marginal
counts or distributions. Synthetic patient identifiers, infection identifiers
and admission dates are generated independently. The logistic coefficients are
then used to rank noisy latent risks, with the published case count fixing the
final outcome prevalence.

The dependency step is deliberately limited to relationships supported by the
aggregate export: it approximates numeric/binary correlations and missingness,
but does not claim to reproduce every higher-order or categorical relationship.
If `dependency_correlations.csv` is absent, the generator remains compatible
with older summary bundles and samples predictors independently.

Real and synthetic data are summarised by separate scripts because the local
environment does not contain the iCARE configuration and utility modules:

```bash
# Run inside iCARE against the real modelling data
python scripts/summarise_real_data.py

# Run locally against the generated data (no iCARE config or utilities needed)
python scripts/summarise_synthetic_data.py
```

The synthetic script uses `data/real_data_summary/` only to identify which
variables are continuous, binary and categorical. It reads
`data/synthetic_data.csv` and writes to `outputs/synthetic_data_summary/` by
default; all three paths can be overridden with command-line arguments.

From the project root, generate the default dataset with:

```bash
python -m src.generate_synthetic_data
```

This reads from `data/real_data_summary/`, writes
`data/synthetic_data.csv`, and uses a fixed random seed (`2026`) so the output
is reproducible. Input/output locations, the seed, coefficient file and date
range can be changed with command-line options; run
`python -m src.generate_synthetic_data --help` for details.
